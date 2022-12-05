# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import logging
import multiprocessing
import time
from pathlib import Path
from typing import List

import datasets as hf_datasets
import numpy as np
import torch
from fairseq.data import LanguagePairDataset, data_utils

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
        dictionary,
        encoder: Encoder,
        append_source_id=None,
        append_target_id=None,
        parallel_prob: float = 1.0,
        seed: int = 1,
        max_seq_len=None,
        num_proc: int = 4,
        max_merges=10,
        passthrough_prob=0.1,
    ):
        super().__init__(None, 0, dictionary)
        self.dictionary = dictionary
        self.parallel_prob = parallel_prob
        self.mixture_ratios = [self.parallel_prob, 1 - self.parallel_prob]
        self.encoder = encoder
        self.seed = seed
        self.max_seq_len = max_seq_len
        self.max_merges = max_merges
        self.passthrough_prob = passthrough_prob
        self.num_proc = num_proc

        assert parallel_datasets or bt_datasets
        self.parent_parallel_datasets = parallel_datasets
        self.parent_bt_datasets = bt_datasets
        all_datasets = parallel_datasets + bt_datasets
        self.foo = all_datasets[0]

        self.flat_src = hf_datasets.concatenate_datasets(
            [d.flat_src for d in all_datasets],
            axis=0,
        )
        # logger.info(f"Loaded {len(self.flat_src)} source segments")
        # self.flat_src.to_parquet("/data/scratch/haukur/document_translation/parquet/flat_src.parquet")
        # self.flat_src = hf_datasets.Dataset.from_parquet(
        #     "/data/scratch/haukur/document_translation/parquet/flat_src.parquet"
        # )

        self.flat_tgt = hf_datasets.concatenate_datasets(
            [d.flat_tgt for d in all_datasets],
            axis=0,
        )
        # logger.info(f"Loaded {len(self.flat_tgt)} target segments")
        # self.flat_tgt.to_parquet("/data/scratch/haukur/document_translation/parquet/flat_tgt.parquet")
        # self.flat_tgt = hf_datasets.Dataset.from_parquet(
        #     "/data/scratch/haukur/document_translation/parquet/flat_tgt.parquet"
        # )

        self.bt_src_start = (
            np.cumsum([len(dset.flat_src) for dset in parallel_datasets])[-1] if parallel_datasets else 0
        )
        self.bt_tgt_start = (
            np.cumsum([len(dset.flat_tgt) for dset in parallel_datasets])[-1] if parallel_datasets else 0
        )

        nparallel = len(parallel_datasets)
        doc_src_offsets = lengths_to_offsets([len(d.flat_src) for d in all_datasets])
        doc_tgt_offsets = lengths_to_offsets([len(d.flat_tgt) for d in all_datasets])

        all_offsets = [
            np.tile([src, tgt], len(dset.flat_align)).reshape(-1, 2)
            for (dset, src, tgt) in zip(all_datasets, doc_src_offsets, doc_tgt_offsets)
        ]
        parallel_offsets, bt_offsets = all_offsets[:nparallel], all_offsets[nparallel:]
        empty_array = np.array([], dtype=np.int64).reshape(0, 2)
        p_all_offsets = np.concatenate(parallel_offsets) if parallel_offsets else empty_array
        b_all_offsets = np.concatenate(bt_offsets) if bt_offsets else empty_array

        # The reason we keep them separated is that they are super/sub-sampled during the interleave
        # which is discarded after each epoch (with a new seed)
        self.flat_align_parallel = (
            hf_datasets.concatenate_datasets([d.flat_align for d in parallel_datasets], axis=0)
            .add_column(KEYS.SOURCE_OFFSETS, column=p_all_offsets[:, 0])
            .add_column(KEYS.TARGET_OFFSETS, column=p_all_offsets[:, 1])
        )
        self.flat_align_bt = (
            hf_datasets.concatenate_datasets([d.flat_align for d in bt_datasets], axis=0)
            .add_column(KEYS.SOURCE_OFFSETS, column=b_all_offsets[:, 0])
            .add_column(KEYS.TARGET_OFFSETS, column=b_all_offsets[:, 1])
        )

        # this gets set after set_epoch or interleave_indices is called
        self.index_dataset = None
        self.epoch = None
        self._interleave_seed = None
        # definitions for langpairdataset functionality
        # ConcatDataset expects a numpy array or list
        self._sizes = None
        self.append_source_id = append_source_id
        self.append_target_id = append_target_id
        self.tgt_eos = self.dictionary.eos() if self.append_target_id is None else self.append_target_id
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

    def __getitem__(self, index):
        assert (
            self.index_dataset is not None
        ), "You must call the interleave_indices() on this dataset before accessing items"
        item = self.index_dataset[int(index)]
        is_bt = any(item[KEYS.SOURCE_INDICES] >= self.bt_src_start)
        maybe_noised_encode_fn = self.encoder.encode_noisy if is_bt else self.encoder.encode
        src_segments = [self.flat_src[int(i)]["segment"] for i in item[KEYS.SOURCE_INDICES]]
        tgt_segments = [self.flat_tgt[int(i)]["segment"] for i in item[KEYS.TARGET_INDICES]]

        with data_utils.numpy_seed(self.seed, self.epoch, index):
            insert_sep = np.random.randint(2, dtype=np.bool)

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

        src_affix = (
            [self.dictionary.eos()] if self.append_source_id is None else [self.dictionary.eos(), self.append_source_id]
        )
        tgt_affix = (
            [self.dictionary.eos()] if self.append_target_id is None else [self.dictionary.eos(), self.append_target_id]
        )
        src_out = torch.cat(src_out + [torch.tensor(src_affix)])
        tgt_out = torch.cat(tgt_out + [torch.tensor(tgt_affix)])

        if len(src_out) > 1020 or len(tgt_out) > 1020:
            assert False

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

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        logger.info(f"Preparing next epoch")
        self.interleave_indices()
        logger.info(f"Done preparing epoch")

    def _interleave_indices_inner(self):
        parallel_merged = None
        bt_merged = None

        if self.flat_align_parallel is not None:
            logger.info(f"Merging adjacent parallel using {self.num_proc} workers")
            parallel_merged = merge_adjacent_sentences(
                self.flat_align_parallel,
                num_proc=self.num_proc,
                passthrough_prob=self.passthrough_prob,
                max_seq_len=self.max_seq_len,
                max_merges=self.max_merges,
                seed=self.seed,
            )
        if self.flat_align_bt is not None:
            logger.info(f"Merging adjacent bt using {self.num_proc} workers")
            bt_merged = merge_adjacent_sentences(
                self.flat_align_bt,
                num_proc=self.num_proc,
                passthrough_prob=self.passthrough_prob,
                max_seq_len=self.max_seq_len,
                max_merges=self.max_merges,
                seed=self.seed,
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

        self._interleave_seed = self.epoch
        logger.info("Sorting index dataset on lengths")
        lengths = np.array(self.index_dataset[KEYS.WEIGHT])
        self._sorted_indices = lengths.argsort()
        self._sorted_lengths = lengths[self._sorted_indices]
        self._sizes = self._sorted_lengths

        logger.info(f"Caching index_dataset to disk at epoch {self.epoch} to {self.get_index_cache_path()}")
        _ = self.index_dataset.save_to_disk(str(self.get_index_cache_path()))

    def get_index_cache_path(self):
        # TODO: add a fingerprint to this path
        document_translation_cache_dir = hf_datasets.config.HF_DATASETS_CACHE / f"document_translation"
        document_translation_cache_dir.mkdir(exist_ok=True)
        return document_translation_cache_dir / f"index_dataset.epoch.{self.epoch}"

    def interleave_indices(self):
        if self.epoch != self._interleave_seed:
            logger.info(
                f"Interleaving parallel and bt datasets with ratios={self.mixture_ratios}, at epoch {self.epoch}"
            )

            index_cache_path = self.get_index_cache_path()
            lockfile_path = Path(f"{index_cache_path}.lock")
            # TODO: write main process PID into lockfile so that we can detect if the lockfile is stale
            while True:
                try:
                    with lockfile_path.open(mode="x") as _lockfile:
                        # mode=exclusive creation
                        pass
                    # we have the lock now

                    ### BEGIN CRITICAL SECTION ###
                    if multiprocessing.get_context().parent_process() is None:
                        # erum main process, þ.e.a.s. ekki worker
                        self._interleave_indices_inner()
                    elif index_cache_path.exists():
                        # cache til og multiprocessing er í gangi
                        # gerum ekki neitt
                        logger.info(f"Found matching cached index_dataset, skipping interleave at epoch {self.epoch}")
                    else:
                        # cache ekki til
                        # Eyðum gomlum indexum TODO XXX: delete older epochs
                        # búum til cache-ið
                        self._interleave_indices_inner()

                    ### END CRITICAL SECTION ###

                    # sleppa las
                    lockfile_path.unlink(missing_ok=False)  # we want to crash if this fails

                    # og loada gognum
                    logger.info(f"Memory mapping indices at epoch {self.epoch}")
                    self.index_dataset = hf_datasets.load_from_disk(str(index_cache_path))
                    logger.info(f"Memory mapped {len(self.index_dataset)} at {self.epoch}")
                    lengths = np.array(self.index_dataset[KEYS.WEIGHT])
                    self._sorted_indices = lengths.argsort()
                    self._sorted_lengths = lengths[self._sorted_indices]
                    self._sizes = self._sorted_lengths
                    break

                except FileExistsError as e:
                    time.sleep(0.25)

    def __len__(self):
        return len(self.index_dataset) if self.index_dataset is not None else 0

    def ordered_sizes(self):
        return self._sorted_lengths

    def ordered_indices(self):
        return self._sorted_indices

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False
