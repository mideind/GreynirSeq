# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import logging
from typing import List

import datasets as hf_datasets
import numpy as np
import torch
from fairseq.data import Dictionary, LanguagePairDataset, data_utils

from greynirseq.nicenlp.data.encoders import Encoder
from greynirseq.nicenlp.data.parallel_documents.indexed_parallel_documents_dataset import (
    KEYS,
    IndexedParallelDocumentsDataset,
    lengths_to_offsets,
    merge_adjacent_sentences,
)

logger = logging.getLogger(__name__)


class IndexedParallelBTDocumentsDataset(LanguagePairDataset):
    def __init__(
        self,
        parallel_datasets: List[IndexedParallelDocumentsDataset],
        bt_datasets: List[IndexedParallelDocumentsDataset],
        dictionary: Dictionary,
        encoder: Encoder,
        parallel_prob: float = 1.0,
        seed: int = 1,
        max_seq_len: int = 1024,
        num_proc: int = 4,
        max_merges: int = 10,
        no_merge_prob: float = 0.1,
    ):
        super().__init__(None, 0, dictionary)
        self.dictionary = dictionary
        self.parallel_prob = parallel_prob
        self.mixture_ratios = [self.parallel_prob, 1 - self.parallel_prob]
        self.encoder = encoder
        self.seed = seed
        self.max_seq_len = max_seq_len
        self.max_merges = max_merges
        self.no_merge_prob = no_merge_prob
        self.num_proc = num_proc

        assert parallel_datasets or bt_datasets
        all_datasets = parallel_datasets + bt_datasets

        self.flat_src: hf_datasets.Dataset = hf_datasets.concatenate_datasets(
            [d.flat_src for d in all_datasets],
            axis=0,
        )

        self.flat_tgt: hf_datasets.Dataset = hf_datasets.concatenate_datasets(
            [d.flat_tgt for d in all_datasets],
            axis=0,
        )

        self.bt_src_start = (
            np.cumsum([len(dset.flat_src) for dset in parallel_datasets])[-1] if parallel_datasets else 0
        )
        self.bt_tgt_start = (
            np.cumsum([len(dset.flat_tgt) for dset in parallel_datasets])[-1] if parallel_datasets else 0
        )

        nparallel = len(parallel_datasets)
        doc_src_offsets = lengths_to_offsets([len(d.flat_src) for d in all_datasets])
        doc_tgt_offsets = lengths_to_offsets([len(d.flat_tgt) for d in all_datasets])

        # create the offsets for the combined dataset by concatenating the offsets of the datasets and repeating them
        # creates a list of (n_i, 2) arrays where n_i is the number of alignments/pairs in the i-th dataset
        all_offsets = [
            np.tile([src_offset, tgt_offset], len(dset.flat_align)).reshape(-1, 2)
            for (dset, src_offset, tgt_offset) in zip(all_datasets, doc_src_offsets, doc_tgt_offsets)
        ]
        parallel_offsets, bt_offsets = all_offsets[:nparallel], all_offsets[nparallel:]
        empty_array = np.array([], dtype=np.int64).reshape(0, 2)
        # split the offsets into parallel and backtranslation
        # The reason we keep them separated is that they are super/sub-sampled during the interleave
        # which is discarded after each epoch (with a new seed)
        p_all_offsets = np.concatenate(parallel_offsets) if parallel_offsets else empty_array
        b_all_offsets = np.concatenate(bt_offsets) if bt_offsets else empty_array

        # keep in mind that the flat_align dataset has the following columns:
        # - document_index: int, the index of the document in the original dataset
        # - paragraph_index: int, the index of the paragraph in the document
        # - weight: int, the max of src and tgt number of BPEs in the segment pair
        # - source_indices: list of ints, the indices of the source segments in the pair
        # - target_indices: list of ints, the indices of the target segments in the pair
        # - skip: bool, whether to skip this pair
        # we then add two columns for the offsets of the source and target segments
        # those values are "global" offsets, i.e. they are the offsets in the concatenated dataset
        # so that we can index into the concatenated flat_src and flat_tgt datasets with the source indices + offsets
        self.flat_align_parallel = (
            hf_datasets.concatenate_datasets([d.flat_align for d in parallel_datasets], axis=0)  # type: ignore
            .add_column(KEYS.SOURCE_OFFSETS, column=p_all_offsets[:, 0])
            .add_column(KEYS.TARGET_OFFSETS, column=p_all_offsets[:, 1])
        )
        self.flat_align_bt = (
            hf_datasets.concatenate_datasets([d.flat_align for d in bt_datasets], axis=0)  # type: ignore
            .add_column(KEYS.SOURCE_OFFSETS, column=b_all_offsets[:, 0])
            .add_column(KEYS.TARGET_OFFSETS, column=b_all_offsets[:, 1])
        )

        # this gets set after set_epoch or interleave_indices is called
        self.index_dataset = None
        self.epoch = 0
        self._interleave_seed = None
        # definitions for langpairdataset functionality
        # ConcatDataset expects a numpy array or list
        self._sizes = None
        # this is compatibility with LanguagePairDataset collater and its teacher forcing adjustments
        self.src_dict = self.dictionary
        self.left_pad_source = False  # expected by bart model
        self.left_pad_target = False  # expected by bart model
        self.src_lang_id = (
            None  # fairseq 0.10.2 accesses these in LanguagePairDataset.collater (so this attribute must exist)
        )
        self.tgt_lang_id = None
        self.src_sizes = self.sizes
        self.tgt_sizes = None
        self._dataset_ntokens = None
        self._sorted_indices = None
        self._sorted_lengths = None
        self.num_parallel_skipped = sum(self.flat_align_parallel[KEYS.SKIP])
        self.num_bt_skipped = sum(self.flat_align_bt[KEYS.SKIP])

    def __getitem__(self, index):
        assert (
            self.index_dataset is not None
        ), "You must call the interleave_indices() on this dataset before accessing items"
        item = self.index_dataset[int(index)]
        is_bt = any(item[KEYS.SOURCE_INDICES] >= self.bt_src_start)
        maybe_noised_encode_fn = self.encoder.encode_noisy if is_bt else self.encoder.encode
        src_segments: List[str] = [self.flat_src[int(i)]["segment"] for i in item[KEYS.SOURCE_INDICES]]
        tgt_segments: List[str] = [self.flat_tgt[int(i)]["segment"] for i in item[KEYS.TARGET_INDICES]]
        src_langs: List[str] = [self.flat_src[int(i)]["lang"] for i in item[KEYS.SOURCE_INDICES]]
        tgt_langs: List[str] = [self.flat_tgt[int(i)]["lang"] for i in item[KEYS.TARGET_INDICES]]
        if len(set(src_langs)) != 1:
            self.log_example(index=index)
            raise ValueError("source segments must be from the same language")
        if len(set(tgt_langs)) != 1:
            self.log_example(index=index)
            raise ValueError("target segments must be from the same language")

        # Experimental: add BT information
        bt_info = self.encoder.encode("BT") if is_bt else torch.tensor([], dtype=torch.long)
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            insert_sep = np.random.randint(2, dtype=bool)

            assert KEYS.EXACT_ALIGNMENT in item or not insert_sep  # insert_sep implies exact_alignment
            if insert_sep and len(src_segments) > 1 and np.all(item[KEYS.EXACT_ALIGNMENT]):
                # only insert separator when alignment is *exact*
                bos = torch.tensor([self.dictionary.bos()])
                src_out = [bos] * (len(src_segments) * 2 - 1)
                src_out[0::2] = [maybe_noised_encode_fn(seg) for seg in src_segments]
                tgt_out = [bos] * (len(tgt_segments) * 2 - 1)
                tgt_out[0::2] = [self.encoder.encode(seg) for seg in tgt_segments]
            else:
                src_out = [maybe_noised_encode_fn(seg) for seg in src_segments]
                tgt_out = [self.encoder.encode(seg) for seg in tgt_segments]

        # This language code handling is like the mBart-50 model and nllb-200
        src_out = torch.cat(
            [torch.tensor([self.dictionary.index(src_langs[0])])]
            + src_out
            + [torch.tensor([self.dictionary.eos()]), bt_info]
        )
        tgt_out = torch.cat(
            [torch.tensor([self.dictionary.index(tgt_langs[0])])] + tgt_out + [torch.tensor([self.dictionary.eos()])]
        )

        if len(src_out) > self.max_seq_len or len(tgt_out) > self.max_seq_len:
            logger.warning(
                f"Truncating example at index={index} because it is too long: src={len(src_out)}, tgt={len(tgt_out)}"
            )
            self.log_example(index=index)
            # We take the first 510 tokens and the last 510 tokens
            half_seq_len = self.max_seq_len // 2
            src_out = torch.cat([src_out[:half_seq_len], src_out[-half_seq_len:]])
            tgt_out = torch.cat([tgt_out[:half_seq_len], tgt_out[-half_seq_len:]])

        example = {
            "id": index,
            "source": src_out,
            "target": tgt_out,
        }

        return example

    def decode(self, example):
        src_string = self.src_dict.string(example["source"])
        tgt_string = self.src_dict.string(example["target"])
        print(f"{self.encoder.bpe.decode(src_string)}")
        print(f"{self.encoder.bpe.decode(tgt_string)}")
        print()

    def log_example(self, index: int):
        """For debugging"""
        item = self.index_dataset[int(index)]
        logger.error(f"index={index}")
        print(f"item={item}")
        is_bt = any(item[KEYS.SOURCE_INDICES] >= self.bt_src_start)
        logger.error(f"is_bt={is_bt}")
        logger.error([self.flat_src[int(i)]["segment"] for i in item[KEYS.SOURCE_INDICES]])
        logger.error([self.flat_tgt[int(i)]["segment"] for i in item[KEYS.TARGET_INDICES]])
        logger.error([self.flat_src[int(i)]["lang"] for i in item[KEYS.SOURCE_INDICES]])
        logger.error([self.flat_tgt[int(i)]["lang"] for i in item[KEYS.TARGET_INDICES]])

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        logger.info(f"Preparing epoch {epoch}")
        self.interleave_indices()
        logger.info(f"Done preparing epoch {epoch}")

    def _interleave_indices_inner(self):
        parallel_merged = None
        bt_merged = None

        if self.flat_align_parallel is not None:
            logger.info(f"Merging adjacent parallel using {self.num_proc} workers")
            parallel_merged = merge_adjacent_sentences(
                self.flat_align_parallel,
                num_proc=self.num_proc,
                no_merge_prob=self.no_merge_prob,
                max_seq_len=self.max_seq_len,
                max_merges=self.max_merges,
                seed=(self.seed, self.epoch),
            )
        if self.flat_align_bt is not None:
            logger.info(f"Merging adjacent bt using {self.num_proc} workers")
            bt_merged = merge_adjacent_sentences(
                self.flat_align_bt,
                num_proc=self.num_proc,
                no_merge_prob=self.no_merge_prob,
                max_seq_len=self.max_seq_len,
                max_merges=self.max_merges,
                seed=(self.seed, self.epoch),
            )
        if parallel_merged is not None and bt_merged is not None:
            logger.info("Interleaving parallel and bt")
            self.index_dataset = hf_datasets.interleave_datasets(
                [parallel_merged, bt_merged],
                seed=self.epoch,
                probabilities=self.mixture_ratios,
            )

        else:
            self.index_dataset = parallel_merged or bt_merged
        assert self.index_dataset is not None

        self._interleave_seed = self.epoch
        logger.info("Sorting index dataset on lengths")
        lengths = np.array(self.index_dataset[KEYS.WEIGHT])
        self._sorted_indices = lengths.argsort()
        self._sorted_lengths = lengths[self._sorted_indices]
        self._sizes = self._sorted_lengths

    def interleave_indices(self):
        if self.epoch != self._interleave_seed:
            logger.info(
                f"Interleaving parallel and bt datasets with ratios={self.mixture_ratios}, at epoch {self.epoch}"
            )
            self._interleave_indices_inner()

    def __len__(self):
        return len(self.index_dataset) if self.index_dataset is not None else 0

    def ordered_sizes(self):
        return self._sorted_lengths

    def ordered_indices(self):
        return self._sorted_indices

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

    def __str__(self) -> str:
        return (
            f"ParallelBTDataset(num_parallel_alignment_pairs={len(self.flat_align_parallel)}, "
            f"num_bt_alignment_pairs={len(self.flat_align_bt)}, num_parallel_skipped={self.num_parallel_skipped}, "
            f"num_bt_skipped={self.num_bt_skipped}, num_training_pairs={len(self.index_dataset)})"
        )
