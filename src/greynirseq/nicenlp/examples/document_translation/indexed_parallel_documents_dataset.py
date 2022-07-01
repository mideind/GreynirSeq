import json
import logging
from typing import List, Any, Optional, Callable, Union

import numpy as np
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
from datasets import Dataset as HFDataset, Sequence, Value
from fairseq.data import Dictionary

from fairseq.data import FairseqDataset, LanguagePairDataset

from icecream import ic
from fairseq.data import encoders

from .encoders import Encoder


logger = logging.getLogger(__name__)

_DOCUMENT_JSONL_FEATURE_DICT = {
    "document": Sequence(
        feature=Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
        length=-1,
        id=None,
    ),
    "domains": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
    "lang": Value(dtype="string", id=None),
    "title": Value(dtype="string", id=None),
    "uuid": Value(dtype="string", id=None),
}

_ALIGNMENTS_JSONL_FEATURE_DICT = {
    "langs": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
    "alignments": Sequence(
        feature=Sequence(
            feature=Sequence(
                feature=Sequence(
                    feature=Value(dtype="int64", id=None), length=-1, id=None
                ),
                length=-1,
                id=None,
            ),
            length=-1,
            id=None,
        ),
        length=-1,
        id=None,
    ),
    "uuid": Value(dtype="string", id=None),
}


class KEYS:
    DOCUMENT = "document"
    SENTENCE_WEIGHTS = "document.sentence_weights"
    DOCUMENT_IDS = "document.ids"
    DOCUMENT_WEIGHT = "document.weight"
    ALIGNMENTS = "alignments"
    DOCUMENT_INDEX = "document_index"
    UUID = "uuid"
    LANG1 = "1"
    LANG2 = "2"
    LANGS = "langs"
    LANG = "lang"
    PARAGRAPH_INDEX = "paragraph_index"
    LENGTH = "length"
    INDICES = "indices"


class IndexedParallelDocumentsDataset(LanguagePairDataset):
    def __init__(
        self,
        args,
        document_dataset,
        index_dataset,
        dictionary,
        encoder: Encoder,
        append_source_id=None,
        append_target_id=None,
        seed: int = 1,
    ):
        super().__init__(None, 0, dictionary)
        self.args = args
        self.document_dataset = document_dataset
        self.index_dataset = index_dataset
        self.dictionary = dictionary
        self.seed = seed
        # ConcatDataset expects a numpy array or list
        self.sizes = np.array(self.index_dataset[KEYS.LENGTH])
        self.append_source_id = append_source_id
        self.append_target_id = append_target_id
        self.encoder = encoder
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
        self._sorted_indices = self.sizes.argsort()
        self._sorted_lengths = self.sizes[self._sorted_indices]
        self._bpe = None
        self.epoch = 1

    def __len__(self):
        return len(self.index_dataset)

    def size(self, index):
        return self.index_dataset[int(index)][KEYS.LENGTH]

    @property
    def dataset_ntokens(self):
        if self._dataset_ntokens is None:
            self._dataset_ntokens = sum(
                self.document_dataset[f"{KEYS.DOCUMENT_WEIGHT}.{KEYS.LANG1}"]
            )
        return self._dataset_ntokens

    def __getitem__(self, index):
        item = self.index_dataset[int(index)]
        doc_idx = item[KEYS.DOCUMENT_INDEX]
        src_doc = self.document_dataset[doc_idx][f"{KEYS.DOCUMENT}.{KEYS.LANG1}"]
        tgt_doc = self.document_dataset[doc_idx][f"{KEYS.DOCUMENT}.{KEYS.LANG2}"]

        src_segments = [src_doc[item[KEYS.PARAGRAPH_INDEX]][idx] for idx in item[f"{KEYS.INDICES}.{KEYS.LANG1}"]]
        tgt_segments = [tgt_doc[item[KEYS.PARAGRAPH_INDEX]][idx] for idx in item[f"{KEYS.INDICES}.{KEYS.LANG2}"]]

        with data_utils.numpy_seed(self.seed, self.epoch, index):
            insert_sep = np.random.randint(2, dtype=np.bool)
        if insert_sep and len(src_segments) > 1:
            bos = torch.tensor([self.dictionary.bos()])
            src_out = [bos] * (len(src_segments) * 2 - 1)
            src_out[0::2] = [self.encoder.encode(seg) for seg in src_segments]
            tgt_out = [bos] * (len(tgt_segments) * 2 - 1)
            tgt_out[0::2] = [self.encoder.encode(seg) for seg in tgt_segments]
        else:
            src_out = [self.encoder.encode(seg) for seg in src_segments]
            tgt_out = [self.encoder.encode(seg) for seg in tgt_segments]

        src_affix = ([self.dictionary.eos()] if self.append_source_id is None else [self.dictionary.eos(), self.append_source_id])
        tgt_affix = ([self.dictionary.eos()] if self.append_target_id is None else [self.dictionary.eos(), self.append_target_id])
        src_out = torch.cat(src_out + [torch.tensor(src_affix)])
        tgt_out = torch.cat(tgt_out + [torch.tensor(tgt_affix)])

        return {"id": index, "source": src_out, "target": tgt_out}

    @classmethod
    def from_parallel_jsonl_many(
        cls,
        args,
        src_paths: List[str],
        tgt_paths: List[str],
        bpe_encoder,
        dictionary: Dictionary,
        encoder: Encoder,
        max_seq_len: int = None,
        append_source_id: int = None,
        append_target_id: int = None,
        max_merges: Optional[int] = 10,
        load_from_cache_file: bool = True,
        num_proc: int = 1,
        align_paths: Optional[str] = None,
        seed: int = 1,
    ):
        features = hf_datasets.Features(_DOCUMENT_JSONL_FEATURE_DICT)

        logger.info(f"Loading src_dataset: {src_paths}")
        src_dataset = hf_datasets.Dataset.from_json(
            src_paths, split="train", chunksize=40 << 20, features=features
        )
        logger.info(f"Loading tgt_dataset: {src_paths}")
        tgt_dataset = hf_datasets.Dataset.from_json(
            tgt_paths, split="train", chunksize=40 << 20, features=features
        )

        src_lang = src_dataset[0][KEYS.LANG]
        tgt_lang = tgt_dataset[0][KEYS.LANG]
        flip_alignment = False
        align_dataset = None
        if align_paths is not None:
            align_dataset = hf_datasets.Dataset.from_json(
                align_paths,
                split="train",
                chunksize=40 << 20,
                features=hf_datasets.Features(_ALIGNMENTS_JSONL_FEATURE_DICT),
            )
            assert set(align_dataset[0][KEYS.LANGS]) == set([src_lang, tgt_lang])
            if align_dataset[KEYS.LANGS] != [src_lang, tgt_lang]:
                flip_alignment = True

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
        length_dataset = merge_length_datasets(
            src_sent_lens,
            tgt_sent_lens,
            num_proc=num_proc,
            check_lengths=False,
            alignment=align_dataset,
        )
        logger.info(f"Computing contiguous training indices")
        index_dataset = length_dataset_to_contiguous_indices(
            length_dataset,
            max_seq_len=max_seq_len,
            max_merges=max_merges,
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
            flip_alignment=flip_alignment,
        )

        logger.info(f"Merging src and tgt documents")
        document_dataset = merge_document_datasets(
            src_dataset, tgt_dataset,
        )
        document_dataset = hf_datasets.concatenate_datasets(
            [document_dataset, length_dataset], axis=1
        )
        index_dataset.set_format(
            type="numpy", columns=KEYS.LENGTH, output_all_columns=True
        )

        return cls(
            args,
            document_dataset,
            index_dataset,
            dictionary,
            encoder=encoder,
            append_source_id=append_source_id,
            append_target_id=append_target_id,
            seed=seed,
        )

    def ordered_sizes(self):
        return self._sorted_lengths

    def ordered_indices(self):
        return self._sorted_indices

    def set_epoch(self, epoch: int):
        self.epoch = epoch


def get_mono_document_sentence_lengths_dataset(
    dataset: HFDataset,
    bpe_encoder,
    dictionary: Dictionary,
    load_from_cache_file=True,
    num_proc: int = 4,
):
    """Encode documents with bpe encoder and fairseq dictionary and remove text afterwards"""
    keep_columns = [KEYS.SENTENCE_WEIGHTS, KEYS.DOCUMENT_WEIGHT]
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
                        bpe_encoder.encode(segment.strip()),
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


def merge_length_datasets(
    lang1_dset: HFDataset,
    lang2_dset: HFDataset,
    num_proc: int = 4,
    alignment: Optional[HFDataset] = None,
    check_lengths: bool = False,
):
    lang1_remapped = lang1_dset.rename_columns(
        {key: f"{key}.{KEYS.LANG1}" for key in lang1_dset.column_names}
    )
    lang2_remapped = lang2_dset.rename_columns(
        {key: f"{key}.{KEYS.LANG2}" for key in lang1_dset.column_names}
    )
    to_be_merged = [lang1_remapped, lang2_remapped]
    if alignment:
        to_be_merged.append(alignment)
    merged = hf_datasets.concatenate_datasets(to_be_merged, axis=1)
    if check_lengths:
        merged.map(check_merged_documents_match, num_proc=num_proc)
    return merged


def check_merged_documents_match(example):
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
    flip_alignment: bool = False,
):
    dataset = dataset.map(
        _insert_document_idx,
        with_indices=True,
        load_from_cache_file=load_from_cache_file,
    )
    allow_columns = [
        KEYS.DOCUMENT_INDEX,
        KEYS.PARAGRAPH_INDEX,
        f"{KEYS.INDICES}.{KEYS.LANG1}",
        f"{KEYS.INDICES}.{KEYS.LANG2}",
        KEYS.LENGTH,
    ]
    remove_columns = [col for col in dataset.column_names if col not in allow_columns]
    fn = (
        _lengths_to_contiguous_indices_w_align
        if KEYS.ALIGNMENTS in dataset.column_names
        else _lengths_to_contiguous_indices
    )
    fn_kwargs = {"max_seq_len": max_seq_len, "max_merges": max_merges}
    if flip_alignment:
        fn_kwargs["flip_alignment"] = flip_alignment
    dataset = dataset.map(
        fn,
        batched=True,
        batch_size=1,
        fn_kwargs=fn_kwargs,
        remove_columns=remove_columns,
        load_from_cache_file=True,
        num_proc=num_proc,
    )
    return dataset


def _lengths_to_contiguous_indices(
    example_in, max_seq_len: int = 2 ** 63 - 1, max_merges: Optional[int] = None
):
    doc_indices = example_in[KEYS.DOCUMENT_INDEX]
    max_merges = max_merges or 10
    assert KEYS.ALIGNMENTS not in example_in
    # this is structure of arrays (as opposed to array of structures)
    example_out = {
        KEYS.DOCUMENT_INDEX: [],
        KEYS.PARAGRAPH_INDEX: [],
        f"{KEYS.INDICES}.{KEYS.LANG1}": [],
        KEYS.LENGTH: [],
    }
    for doc_idx, doc1, doc2 in zip(
        doc_indices,
        example_in[f"{KEYS.SENTENCE_WEIGHTS}.{KEYS.LANG1}"],
        example_in[f"{KEYS.SENTENCE_WEIGHTS}.{KEYS.LANG2}"],
    ):
        for pg_idx, (pg1, pg2) in enumerate(zip(doc1, doc2)):
            longer_seq_lens = [max(seg1, seg2) for seg1, seg2 in zip(pg1, pg2)]
            shorter_seq_lens = [min(seg1, seg2) for seg1, seg2 in zip(pg1, pg2)]
            new_extended_indices = []
            for idx, (long_len, short_len) in enumerate(
                zip(longer_seq_lens, shorter_seq_lens)
            ):
                # try to add to current sequences, filter them out if they aren't to long
                new_extended_indices = [
                    (curr_sent_idxs + [idx], curr_len + long_len)
                    for (curr_sent_idxs, curr_len) in new_extended_indices
                    if (short_len > 0)
                    and (curr_len + long_len < max_seq_len)
                    and (len(curr_sent_idxs) < max_merges)
                ]
                # if we are allowed to add this sentence in isolation then we are also
                # allowed to add the extensions it may have caused
                if 0 < short_len < max_seq_len and long_len < max_seq_len:
                    new_extended_indices.append(([idx], long_len))
                if new_extended_indices:
                    # fmt: off
                    # example_out[KEYS.INDICES].extend([idxs for idxs, long_len in new_extended_indices])
                    example_out[f"{KEYS.INDICES}.{KEYS.LANG1}"].extend([idxs for idxs, long_len in new_extended_indices])
                    example_out[KEYS.LENGTH].extend([long_len for idxs, long_len in new_extended_indices])
                    example_out[KEYS.DOCUMENT_INDEX].extend([doc_idx] * len(new_extended_indices))
                    example_out[KEYS.PARAGRAPH_INDEX].extend([pg_idx] * len(new_extended_indices))
                    # fmt: on

    example_out[f"{KEYS.INDICES}.{KEYS.LANG2}"] = example_out[
        f"{KEYS.INDICES}.{KEYS.LANG1}"
    ]

    # cannot return this as a tensor or list of tensors, hf_datasets expects list or array
    # example_out[KEYS.DOCUMENT_INDEX] = np.array(example_out[KEYS.DOCUMENT_INDEX])
    return example_out


def _lengths_to_contiguous_indices_w_align(
    example_in,
    max_seq_len: int = 2 ** 63 - 1,
    max_merges: Optional[int] = None,
    flip_alignment: bool = False,
):
    # this function is a near copy of _lengths_to_contiguous_indices_w_align, the key
    # difference is that it iterates over alignments as units (which can be more than
    # one sentence), instead of individual sentences
    assert KEYS.ALIGNMENTS in example_in
    max_merges = max_merges or 10
    # this is structure of arrays (as opposed to array of structures)
    example_out = {
        KEYS.DOCUMENT_INDEX: [],
        KEYS.PARAGRAPH_INDEX: [],
        KEYS.LENGTH: [],
        f"{KEYS.INDICES}.{KEYS.LANG1}": [],
        f"{KEYS.INDICES}.{KEYS.LANG2}": [],
    }

    for rel_doc_idx, (abs_doc_idx, alignments) in enumerate(
        zip(example_in[KEYS.DOCUMENT_INDEX], example_in[KEYS.ALIGNMENTS])
    ):
        weights1 = example_in[f"{KEYS.SENTENCE_WEIGHTS}.{KEYS.LANG1}"][rel_doc_idx]
        weights2 = example_in[f"{KEYS.SENTENCE_WEIGHTS}.{KEYS.LANG2}"][rel_doc_idx]
        last_pg_idx = -1
        new_extended_indices = []
        for pair_idx, (src_side, tgt_side) in enumerate(alignments):
            if flip_alignment:
                src_side, tgt_side = tgt_side, src_side
            pg_idx = src_side[0][0]
            if tgt_side[0][0] != pg_idx:
                breakpoint()
                print()
            assert tgt_side[0][0] == pg_idx
            if len(src_side) > 1:
                assert all(item[0] == pg_idx for item in src_side)
            if len(tgt_side) > 1:
                assert all(item[0] == pg_idx for item in tgt_side)
            src_len = sum(weights1[pg_idx][item[1]] for item in src_side)
            tgt_len = sum(weights2[pg_idx][item[1]] for item in tgt_side)
            short_len = min(src_len, tgt_len)
            long_len = max(src_len, tgt_len)
            new_extended_indices = [
                (curr_sent_idxs + [pair_idx], curr_len + long_len)
                for (curr_sent_idxs, curr_len) in new_extended_indices
                if (short_len > 0)
                and (curr_len + long_len < max_seq_len)
                and (len(curr_sent_idxs) < max_merges)
                and (pg_idx == last_pg_idx)
            ]
            if 0 < short_len < max_seq_len and long_len < max_seq_len:
                new_extended_indices.append(([pair_idx], long_len))
            if new_extended_indices:
                _LANG1, _LANG2 = 0, 1
                if flip_alignment:
                    _LANG1, _LANG2 = _LANG2, _LANG1
                example_out[f"{KEYS.INDICES}.{KEYS.LANG1}"].extend(
                    [
                        [
                            item[1]
                            for pair_idx in pair_idxs
                            for item in alignments[pair_idx][_LANG1]
                        ]
                        for (pair_idxs, _) in new_extended_indices
                    ]
                )
                example_out[f"{KEYS.INDICES}.{KEYS.LANG2}"].extend(
                    [
                        [
                            item[1]
                            for pair_idx in pair_idxs
                            for item in alignments[pair_idx][_LANG2]
                        ]
                        for (pair_idxs, _) in new_extended_indices
                    ]
                )
                example_out[KEYS.LENGTH].extend(
                    [long_len for idxs, long_len in new_extended_indices]
                )
                example_out[KEYS.DOCUMENT_INDEX].extend(
                    [abs_doc_idx] * len(new_extended_indices)
                )
                example_out[KEYS.PARAGRAPH_INDEX].extend(
                    [pg_idx] * len(new_extended_indices)
                )
            last_pg_idx = pg_idx

    # example_out[KEYS.LENGTH] = np.array(example_out[KEYS.LENGTH])
    # example_out[KEYS.PARAGRAPH_INDEX] = np.array(example_out[KEYS.PARAGRAPH_INDEX])
    # cannot return this as a tensor or list of tensors, hf_datasets expects list or array
    # example_out[KEYS.DOCUMENT_INDEX] = np.array(example_out[KEYS.DOCUMENT_INDEX])

    return example_out


def merge_document_datasets(
    lang1_dset: HFDataset, lang2_dset: HFDataset
):
    keep_columns = [KEYS.DOCUMENT]
    remove_columns = [k for k in lang1_dset.column_names if k not in keep_columns]
    lang1_remapped = lang1_dset.remove_columns(remove_columns).rename_columns(
        {
            key: f"{key}.{KEYS.LANG1}"
            for key in lang1_dset.column_names
            if key in keep_columns
        }
    )
    lang2_remapped = lang2_dset.remove_columns(remove_columns).rename_columns(
        {
            key: f"{key}.{KEYS.LANG2}"
            for key in lang1_dset.column_names
            if key in keep_columns
        }
    )
    merged = hf_datasets.concatenate_datasets([lang1_remapped, lang2_remapped], axis=1)
    return merged


def _insert_document_idx(example, document_index: int):
    example[KEYS.DOCUMENT_INDEX] = document_index
    return example
