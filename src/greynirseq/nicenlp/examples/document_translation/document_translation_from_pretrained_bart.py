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

import logging

from fairseq import utils
from fairseq.data import FairseqDataset, data_utils, encoders, iterators
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.tasks.translation_from_pretrained_bart import TranslationFromPretrainedBARTTask
from icecream import ic

from .encoders import Encoder
from .indexed_parallel_bt_documents_dataset import IndexedParallelBTDocumentsDataset
from .indexed_parallel_documents_dataset import IndexedParallelDocumentsDataset

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
        parser.add_argument('--sentencepiece-alpha', type=float, default=1.00,
                            help='Parameter for segmentation distribution, '
                                 'this is NOT a probability')
        parser.add_argument('--parallel-prob', type=float, default=0.33,
                            help='Probability of sampling parallel data if bt data '
                                 'is included (Note: NOT sample weight)')
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
        parallel_dataset_names = [name for name in dataset_names if name not in bt_dataset_names]
        assert parallel_dataset_names, "Expected at least one parallel dataset"

        # infer langcode and translation direction
        src, tgt = self.args.source_lang, self.args.target_lang
        direction = f"{src}-{tgt}"

        from .sentencepiece_bpe_sampling import SentencepieceBPESampled  # noqa

        bpe = encoders.build_bpe(self.args)

        # XXX: since this is at document level, we probably dont want to apply this too aggressively
        # XXX: e.g. only enable a noiser with some probability
        my_enc = Encoder(
            self.args, self.src_dict, min_val=self.src_dict.nspecial, max_val=len(self.src_dict) - 1 - len(self.langs)
        )

        src_paths = [f"{data_dir_path}/{name}.{direction}.{src}.jsonl" for name in parallel_dataset_names]
        tgt_paths = [f"{data_dir_path}/{name}.{direction}.{tgt}.jsonl" for name in parallel_dataset_names]
        max_seq_len = int(self.args.max_source_positions * 0.45)  # to account for segmentation noise
        ic(split)
        parallel_dataset = IndexedParallelDocumentsDataset.from_parallel_jsonl_many(
            self.args,
            src_paths,
            tgt_paths,
            bpe,
            self.src_dict,
            encoder=my_enc,
            max_seq_len=max_seq_len,
            # 250_004 for english
            # 250_012 for icelandic
            append_source_id=self.src_dict.index("[{}]".format(self.args.source_lang)),
            append_target_id=self.tgt_dict.index("[{}]".format(self.args.target_lang)),
        )

        if "valid" in split or "test" in split or not bt_dataset_names:
            self.datasets[split] = parallel_dataset
            return parallel_dataset

        src_paths = [f"{data_dir_path}/{name}.{direction}.{src}.jsonl" for name in bt_dataset_names]
        tgt_paths = [f"{data_dir_path}/{name}.{direction}.{tgt}.jsonl" for name in bt_dataset_names]
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
        )

        dataset = IndexedParallelBTDocumentsDataset(
            parallel_dataset,
            bt_dataset,
            self.src_dict,
            encoder=my_enc,
            append_source_id=self.src_dict.index("[{}]".format(self.args.source_lang)),
            append_target_id=self.tgt_dict.index("[{}]".format(self.args.target_lang)),
            parallel_prob=self.args.parallel_prob,
        )
        dataset.set_epoch(1)

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
        can_reuse_epoch_itr = not disable_iterator_cache and self.can_reuse_epoch_itr(dataset)
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

        from .batch_sampler import batch_by_size

        with data_utils.numpy_seed(seed, epoch):
            batch_sampler = batch_by_size(indices, lengths, max_tokens, max_sentences)

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
