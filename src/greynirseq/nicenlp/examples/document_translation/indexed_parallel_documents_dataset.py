import json
import logging
from typing import List, Any, Optional, Callable

import numpy
import torch
from fairseq import utils
from fairseq.data import BaseWrapperDataset, Dictionary, data_utils
from fairseq.data.language_pair_dataset import collate
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset
from fairseq.tasks.translation_from_pretrained_bart import (
    TranslationFromPretrainedBARTTask
)
from torch.utils.data import Dataset
import datasets as hf_datasets
from datasets import Dataset as HFDataset
from fairseq.data import Dictionary

from fairseq.data import FairseqDataset, LanguagePairDataset

from icecream import ic


logger = logging.getLogger(__name__)


class KEYS:
    DOCUMENT = "document"
    SENTENCE_WEIGHTS = "document.sentence_weights"
    DOCUMENT_IDS = "document.ids"
    DOCUMENT_WEIGHT = "document.weight"
    UUID_PREFIX = "document.weight"
    DOCUMENT_INDEX = "document_index"
    UUID = "uuid"
    LANG1 = "1"
    LANG2 = "2"
    PARAGRAPH_INDEX = "paragraph_index"
    LENGTH = "length"
    INDICES = "indices"


class IndexedParallelDocumentsDataset(LanguagePairDataset):
    def __init__(
        self,
        document_dataset,
        index_dataset,
        length_dataset,
        dictionary,
        encode_fn: Optional[Callable] = None,
        append_source_id=None,
        append_target_id=None,
    ):
        super().__init__(None, 0, dictionary)
        self.document_dataset = document_dataset
        self.index_dataset = index_dataset
        self.length_dataset = length_dataset
        self.dictionary = dictionary
        # ConcatDataset expects a numpy array or list
        self.sizes = self.index_dataset[KEYS.LENGTH]
        self.append_source_id = append_source_id
        self.append_target_id = append_target_id
        self.encode_fn = encode_fn
        self.tgt_eos = (
            self.dictionary.eos()
            if self.append_target_id is None
            else self.append_target_id
        )
        # this is compatibility with LanguagePairDataset collater and its teacher forcing adjustments
        self.src_dict = self.dictionary
        self.left_pad_source = False  # expected by bart model
        self.left_pad_target = False  # expected by bart model
        self.src_lang_id = (
            None
        )  # fairseq 0.10.2 accesses these in LanguagePairDataset.collater (so this attribute must exist)
        self.tgt_lang_id = None
        self.src_sizes = self.sizes
        self.tgt_sizes = None
        self._dataset_ntokens = None

    def __len__(self):
        return len(self.index_dataset)

    def size(self, index):
        return self.index_dataset[int(index)][KEYS.LENGTH]

    @property
    def dataset_ntokens(self):
        if self._dataset_ntokens is None:
            self._dataset_ntokens = sum(
                self.length_dataset[f"{KEYS.DOCUMENT_WEIGHT}.{KEYS.LANG1}"]
            )
        return self._dataset_ntokens

    def __getitem__(self, index):
        item = self.index_dataset[index]
        doc_idx = item[KEYS.DOCUMENT_INDEX]
        src_doc = self.document_dataset[doc_idx][f"{KEYS.DOCUMENT}.{KEYS.LANG1}"]
        src_segments = [
            src_doc[item[KEYS.PARAGRAPH_INDEX]][idx] for idx in item[KEYS.INDICES]
        ]
        tgt_doc = self.document_dataset[doc_idx][f"{KEYS.DOCUMENT}.{KEYS.LANG2}"]
        tgt_segments = [
            tgt_doc[item[KEYS.PARAGRAPH_INDEX]][idx] for idx in item[KEYS.INDICES]
        ]
        # src_affix = torch.tensor(
        #     [] if self.append_source_id is None else [self.append_source_id]
        # )
        # src_segments.append(src_affix)
        if self.append_source_id is not None:
            src_segments.append(self.append_source_id)
        # tgt_segments.append(torch.tensor([self.tgt_eos]))
        tgt_segments.append(self.tgt_eos)

        # ic((self.encode_fn(src_segments).numel(), self.encode_fn(tgt_segments).numel()))
        return {
            "id": index,
            "source": self.encode_fn(src_segments)
            if self.encode_fn is not None
            else src_segments,
            "target": self.encode_fn(tgt_segments)
            if self.encode_fn is not None
            else tgt_segments,
        }

    @classmethod
    def from_parallel_jsonl_many(
        cls,
        src_paths: List[str],
        tgt_paths: List[str],
        bpe_encoder,
        dictionary: Dictionary,
        encode_fn: Optional[Callable] = None,
        max_seq_len: int = None,
        append_source_id: int = None,
        append_target_id: int = None,
        max_merges: Optional[int] = 10,
        load_from_cache_file: bool = True,
        num_proc: int = 16,
    ):
        logger.info(f"Loading src_dataset: {src_paths}")
        src_dataset = hf_datasets.load_dataset(
            # "json", data_files=src_paths, split="train[:10]"
            "json", data_files=src_paths, split="train"
        )
        logger.info(f"Loading tgt_dataset: {src_paths}")
        tgt_dataset = hf_datasets.load_dataset(
            # "json", data_files=tgt_paths, split="train[:10]"
            "json", data_files=tgt_paths, split="train"
        )
        logger.info(f"Merging src and tgt documents")
        document_dataset = merge_exactly_parallel_document_datasets(
            src_dataset, tgt_dataset, check_exactly_parallel=False, num_proc=num_proc
        )

        logger.info(f"Computing sentence lengths of src documents")
        src_sent_lens = get_mono_document_sentence_lengths_dataset(
            src_dataset,
            bpe_encoder,
            dictionary,
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
        )
        logger.info(f"Computing sentence lengths of tgt documents")
        tgt_sent_lens = get_mono_document_sentence_lengths_dataset(
            tgt_dataset,
            bpe_encoder,
            dictionary,
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
        )
        logger.info(f"Merging src and tgt lengths")
        length_dataset = merge_exactly_parallel_length_datasets(
            src_sent_lens, tgt_sent_lens, num_proc=num_proc
        )
        logger.info(f"Computing contiguous training indices")
        index_dataset = length_dataset_to_contiguous_indices(
            length_dataset,
            max_seq_len=max_seq_len,
            max_merges=max_merges,
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
        )

        index_dataset.set_format(
            type="numpy", columns=KEYS.LENGTH, output_all_columns=True
        )

        return cls(
            document_dataset,
            index_dataset,
            length_dataset,
            dictionary,
            encode_fn=encode_fn,
            append_source_id=append_source_id,
            append_target_id=append_target_id,
        )


def get_mono_document_sentence_lengths_dataset(
    dataset: HFDataset,
    bpe_encoder,
    dictionary: Dictionary,
    load_from_cache_file=True,
    num_proc: int = 4,
):
    """Encode documents with bpe encoder and fairseq dictionary and remove text afterwards"""
    keep_columns = [KEYS.UUID, KEYS.SENTENCE_WEIGHTS, KEYS.DOCUMENT_WEIGHT]
    remove_columns = [k for k in dataset.column_names if k not in keep_columns]
    dataset = dataset.map(
        _mono_document_to_sentence_lengths,
        fn_kwargs={"bpe_encoder": bpe_encoder, "dictionary": dictionary},
        remove_columns=remove_columns,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )
    return dataset


def _mono_document_to_sentence_lengths(
    example, bpe_encoder=None, dictionary: Dictionary = None
):
    sent_lens_per_pg = []
    for paragraph in example[KEYS.DOCUMENT]:
        sent_lens_per_pg.append(
            [
                len(
                    dictionary.encode_line(
                        bpe_encoder.encode(segment),
                        append_eos=False,
                        add_if_not_exist=False,
                    )
                )
                for segment in paragraph
            ]
        )
    example[KEYS.SENTENCE_WEIGHTS] = sent_lens_per_pg
    example[KEYS.DOCUMENT_WEIGHT] = sum(sum(pg) for pg in sent_lens_per_pg)
    return example


def merge_exactly_parallel_length_datasets(
    lang1_dset: HFDataset, lang2_dset: HFDataset, num_proc: int = 4
):
    lang1_remapped = lang1_dset.rename_columns(
        {key: f"{key}.{KEYS.LANG1}" for key in lang1_dset.column_names}
    )
    lang2_remapped = lang2_dset.rename_columns(
        {key: f"{key}.{KEYS.LANG2}" for key in lang1_dset.column_names}
    )
    merged = hf_datasets.concatenate_datasets([lang1_remapped, lang2_remapped], axis=1)
    merged.map(check_merged_exactly_parallel_documents_match, num_proc=num_proc)
    return merged


def check_merged_exactly_parallel_documents_match(example):
    # assert (
    #     example[f"{KEYS.UUID}.{KEYS.LANG1}"] == example[f"{KEYS.UUID}.{KEYS.LANG2}"]
    # ), "Expected document id's to match"
    for paragraph1, paragraph2 in zip(
        example[f"{KEYS.SENTENCE_WEIGHTS}.{KEYS.LANG1}"],
        example[f"{KEYS.SENTENCE_WEIGHTS}.{KEYS.LANG2}"],
    ):
        assert len(paragraph1) == len(
            paragraph2
        ), "Expected number of sentences in paragraph to match"

    return example


def length_dataset_to_contiguous_indices(
    dataset: HFDataset,
    max_seq_len: int = None,
    max_merges=None,
    load_from_cache_file=True,
    num_proc: int = 4,
):
    dataset = dataset.map(
        _insert_document_idx,
        with_indices=True,
        load_from_cache_file=load_from_cache_file,
    )
    keep_columns = [
        KEYS.DOCUMENT_INDEX,
        KEYS.PARAGRAPH_INDEX,
        KEYS.INDICES,
        KEYS.LENGTH,
    ]
    remove_columns = [col for col in dataset.column_names if col not in keep_columns]
    dataset = dataset.map(
        _lengths_to_contiguous_indices,
        batched=True,
        batch_size=1,
        fn_kwargs={"max_seq_len": max_seq_len, "max_merges": max_merges},
        remove_columns=remove_columns,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )
    return dataset


def _lengths_to_contiguous_indices(
    example_soa, max_seq_len: int = 2 ** 63 - 1, max_merges: Optional[int] = None
):
    doc_indices = example_soa[KEYS.DOCUMENT_INDEX]
    max_merges = max_merges or 10
    example_out_soa = {
        KEYS.DOCUMENT_INDEX: [],
        KEYS.PARAGRAPH_INDEX: [],
        KEYS.INDICES: [],
        KEYS.LENGTH: [],
    }
    # breakpoint()
    for doc_idx, doc1, doc2 in zip(
        doc_indices,
        example_soa[f"{KEYS.SENTENCE_WEIGHTS}.{KEYS.LANG1}"],
        example_soa[f"{KEYS.SENTENCE_WEIGHTS}.{KEYS.LANG2}"],
    ):
        for pg_idx, (pg1, pg2) in enumerate(zip(doc1, doc2)):
            longer_seq_lens = [max(seg1, seg2) for seg1, seg2 in zip(pg1, pg2)]
            smaller_seq_lens = [min(seg1, seg2) for seg1, seg2 in zip(pg1, pg2)]
            curr = []
            for idx, (length, smaller_len) in enumerate(
                zip(longer_seq_lens, smaller_seq_lens)
            ):
                curr = [
                    (idxs + [idx], clen + length)
                    for (idxs, clen) in curr
                    if (smaller_len > 0)
                    and (clen + length < max_seq_len)
                    and (len(idxs) < max_merges)
                ]
                if length < max_seq_len:
                    curr.append(([idx], length))
                if curr:
                    example_out_soa[KEYS.INDICES].extend(
                        [idxs for idxs, length in curr]
                    )
                    example_out_soa[KEYS.LENGTH].extend(
                        [length for idxs, length in curr]
                    )
                    example_out_soa[KEYS.DOCUMENT_INDEX].extend([doc_idx] * len(curr))
                    example_out_soa[KEYS.PARAGRAPH_INDEX].extend([pg_idx] * len(curr))
    # cannot return this as a tensor or list of tensors, hf_datasets expects list or array
    example_out_soa[KEYS.DOCUMENT_INDEX] = numpy.array(
        example_out_soa[KEYS.DOCUMENT_INDEX]
    )
    return example_out_soa


def merge_exactly_parallel_document_datasets(
    lang1_dset: HFDataset,
    lang2_dset: HFDataset,
    check_exactly_parallel=True,
    num_proc: int = 4,
):
    lang1_remapped = lang1_dset.rename_columns(
        {key: f"{key}.{KEYS.LANG1}" for key in lang1_dset.column_names}
    )
    lang2_remapped = lang2_dset.rename_columns(
        {key: f"{key}.{KEYS.LANG2}" for key in lang1_dset.column_names}
    )
    merged = hf_datasets.concatenate_datasets([lang1_remapped, lang2_remapped], axis=1)
    if check_exactly_parallel:
        # merged.map(check_parallel_documents_match)
        pass
    return merged


def _insert_document_idx(example, document_index: int):
    example[KEYS.DOCUMENT_INDEX] = document_index
    return example
