# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

# Based on fairseq/tasks/translation.py and fairseq/tasks/translation_from_pretrained_bart.py
# that has the following license
#
#    Copyright (c) Facebook, Inc. and its affiliates.
#
#    This source code is licensed under the MIT license found in the
#    LICENSE file in the root directory of this source tree.

import json
import logging
from typing import Any, List, Optional, Union

import numpy
import torch
from fairseq.data import encoders
from fairseq import utils
from fairseq.data import BaseWrapperDataset, Dictionary, data_utils, FairseqDataset, iterators
from fairseq.data.language_pair_dataset import collate
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset
from fairseq.tasks.translation_from_pretrained_bart import (
    TranslationFromPretrainedBARTTask
)
from fairseq.data import ConcatDataset

from torch.utils.data import Dataset
import datasets as hf_datasets

from icecream import ic

from .fragment_noise import FragmentNoiser
from .word_noise import WordNoiser
from .spm_segmentation_noise import SpmNoiser


logger = logging.getLogger(__name__)


@register_task("document_translation_from_pretrained_bart")
class DocumentTranslationFromPretrainedBART(TranslationFromPretrainedBARTTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--langs',  type=str, metavar='LANG',
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')
        parser.add_argument('--prepend-bos', action='store_true',
                            help='prepend bos token to each sentence, which matches '
                                 'mBART pretraining')
        parser.add_argument('--max-sentences', type=int, default=100)
        parser.add_argument('--bt-subset', type=str, default="")
        parser.add_argument('--align-subset', type=str, default="", help="The subset of parallel data that has requires an alignment jsonl file")
        parser.add_argument('--sentencepiece-alpha', type=float, default=1.00, help="Parameter for segmentation distribution, this is NOT a probability")
        parser.add_argument('--parallel-prob', type=float, help="Probability of sampling parallel data if bt data is included (Note: NOT sample weight)", default=0.33)
        parser.add_argument('--word-noise-prob', type=float, default=0.01)
        parser.add_argument('--fragment-noise-prob', type=float, default=0.01)
        parser.add_argument('--max-shuffle-dist', type=int, default=3)
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.langs = args.langs.split(",")
        for dict_ in [src_dict, tgt_dict]:
            for lang in self.langs:
                dict_.add_symbol("[{}]".format(lang))
            dict_.add_symbol("<mask>")

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # this is for sharding
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_dir_path = paths[(epoch - 1) % len(paths)]
        # this is for different datasets that comprise the training set
        dataset_names = split.split(",")
        bt_dataset_names = self.args.bt_subset.split(",")
        align_dataset_names = self.args.align_subset.split(",")
        parallel_dataset_names = [
            # XXX:
            name for name in dataset_names if name not in bt_dataset_names and name not in align_dataset_names
        ]
        bt_dataset_names = [n for n in bt_dataset_names if n not in align_dataset_names]
        assert parallel_dataset_names, "Expected at least one sentence parallel dataset"

        # infer langcode and translation direction
        src, tgt = self.args.source_lang, self.args.target_lang
        direction = f"{src}-{tgt}"

        from .sentencepiece_bpe_sampling import SentencepieceBPESampled
        import copy

        _args_w_bpe_sampling = copy.deepcopy(self.args)
        _args_w_bpe_sampling.bpe = "sentencepiece_sampled"
        spm_w_noise = encoders.build_bpe(_args_w_bpe_sampling)
        word_noise_prob = self.args.max_shuffle_dist
        max_shuffle_dist = self.args.max_shuffle_dist
        fragment_noise_prob = self.args.fragment_noise_prob
        word_noiser = WordNoiser(word_noise_prob, max_shuffle_dist)
        noisy_subword_enc = SpmNoiser(self.src_dict, spm_w_noise)
        fragment_noiser = FragmentNoiser(fragment_noise_prob, min_val=self.src_dict.nspecial, max_val=len(self.src_dict) - 1 - len(self.langs))
        # XXX: since this is at document level, we probably dont want to apply this too aggressively
        # XXX: e.g. only enable a noiser with some probability

        bpe = encoders.build_bpe(self.args)
        from .indexed_parallel_documents_dataset import IndexedParallelDocumentsDataset
        from .indexed_parallel_bt_documents_dataset import IndexedParallelBTDocumentsDataset

        from .encoders import Encoder
        my_enc = Encoder(self.args, self.src_dict, min_val=self.src_dict.nspecial, max_val=len(self.src_dict) - 1 - len(self.langs))

        src_paths = [
            f"{data_dir_path}/{name}.{direction}.{src}.jsonl"
            for name in parallel_dataset_names
        ]
        tgt_paths = [
            f"{data_dir_path}/{name}.{direction}.{tgt}.jsonl"
            for name in parallel_dataset_names
        ]
        max_seq_len = int(self.args.max_source_positions * 0.45)  # to account for segmentation noise

        parallel_dataset = IndexedParallelDocumentsDataset.from_parallel_jsonl_many(
            self.args,
            src_paths,
            tgt_paths,
            bpe,
            self.src_dict,
            encoder=my_enc,
            max_seq_len=max_seq_len,
            append_source_id=self.src_dict.index("[{}]".format(self.args.source_lang)),  # 250_004 for english
            append_target_id=self.tgt_dict.index("[{}]".format(self.args.target_lang)),  # 250_012 for icelandic
            seed=self.args.seed,
        )

        valid_dataset_names = self.args.valid_subset
        if split in valid_dataset_names:
            self.datasets[split] = parallel_dataset
            return parallel_dataset

        parallel_datasets = [parallel_dataset]
        if align_dataset_names:
            src_paths = [
                f"{data_dir_path}/{name}.{direction}.{src}.jsonl"
                for name in align_dataset_names
            ]
            tgt_paths = [
                f"{data_dir_path}/{name}.{direction}.{tgt}.jsonl"
                for name in align_dataset_names
            ]
            align_paths = [
                f"{data_dir_path}/{name}.{direction}.align.jsonl"
                for name in align_dataset_names
            ]
            aligned_dataset = IndexedParallelDocumentsDataset.from_parallel_jsonl_many(
                self.args,
                src_paths,
                tgt_paths,
                bpe,
                self.src_dict,
                encoder=my_enc,
                max_seq_len=max_seq_len,
                append_source_id=self.src_dict.index("[{}]".format(self.args.source_lang)),  # 250_004 for english
                append_target_id=self.tgt_dict.index("[{}]".format(self.args.target_lang)),  # 250_012 for icelandic
                align_paths=align_paths,
                seed=self.args.seed,
            )
            parallel_datasets.append(aligned_dataset)

        src_paths = [
            f"{data_dir_path}/{name}.{direction}.{src}.jsonl"
            for name in bt_dataset_names
        ]
        tgt_paths = [
            f"{data_dir_path}/{name}.{direction}.{tgt}.jsonl"
            for name in bt_dataset_names
        ]
        bt_dataset = IndexedParallelDocumentsDataset.from_parallel_jsonl_many(
            self.args,
            src_paths,
            tgt_paths,
            bpe,
            self.src_dict,
            encoder=my_enc,
            max_seq_len=max_seq_len,
            append_source_id=self.src_dict.index("[{}]".format(self.args.source_lang)),
            append_target_id=self.tgt_dict.index("[{}]".format(self.args.target_lang)),
            seed=self.args.seed,
        )

        dataset = IndexedParallelBTDocumentsDataset(
            parallel_datasets,
            bt_dataset,
            self.src_dict,
            encoder=my_enc,
            append_source_id=self.src_dict.index("[{}]".format(self.args.source_lang)),
            append_target_id=self.tgt_dict.index("[{}]".format(self.args.target_lang)),
            parallel_prob=self.args.parallel_prob,
            seed=self.args.seed,
        )
        dataset.set_epoch(1)

        # def decode(example):
        #     src_string = self.src_dict.string(example["source"])
        #     tgt_string = self.src_dict.string(example["target"])
        #     # decoded = bpe.decode(spm_string)
        #     print()
        #     print(bpe.decode(src_string))
        #     print()
        #     print(bpe.decode(tgt_string))
        #     print()
        #     print()

        self.datasets[split] = dataset
        return dataset

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
        data_buffer_size=0,
        disable_iterator_cache=False,
    ):
        if not hasattr(dataset, "ordered_sizes"):
            return super().get_batch_iterator(
                dataset,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_positions=max_positions,
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=num_workers,
                epoch=epoch,
                data_buffer_size=data_buffer_size,
                disable_iterator_cache=disable_iterator_cache,
            )
        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(
            dataset
        )
        if can_reuse_epoch_itr and dataset in self.dataset_to_epoch_iter:
            logger.debug("reusing EpochBatchIterator for epoch {}".format(epoch))
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()
        lengths = dataset.ordered_sizes()

        logger.debug("Batching by size...")
        from .batch_sampler import batch_by_size
        with data_utils.numpy_seed(seed, epoch):
            batch_sampler = batch_by_size(indices, lengths, max_tokens, max_sentences)
        logger.debug("Done")

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

    def can_reuse_epoch_itr(self, dataset):
        return False
