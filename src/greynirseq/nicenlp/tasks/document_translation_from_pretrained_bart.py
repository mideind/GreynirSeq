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
from fairseq.data import Dictionary, FairseqDataset, data_utils, encoders, iterators
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask
from fairseq.tasks.translation_from_pretrained_bart import TranslationFromPretrainedBARTTask

from greynirseq.nicenlp.data.batch_sampler import batch_by_size
from greynirseq.nicenlp.data.parallel_documents.indexed_parallel_bt_documents_dataset import (
    IndexedParallelBTDocumentsDataset,
)
from greynirseq.nicenlp.data.parallel_documents.indexed_parallel_documents_dataset import (
    IndexedParallelDocumentsDataset,
)

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
        parser.add_argument('--max-sentences', type=int, default=100
        )
        parser.add_argument('--max-sequence-length', type=int, default=int(1024*0.75))
        parser.add_argument('--num-preprocess-workers', type=int, default=2)
        parser.add_argument('--bt-subset', type=str, default="")
        parser.add_argument('--align-subset', type=str, default="", help="The subset of parallel data that has requires an alignment jsonl file")
        parser.add_argument('--sentencepiece-alpha', type=float, default=1.00, help="Parameter for segmentation distribution, this is NOT a probability")
        parser.add_argument('--parallel-prob', type=float, help="Probability of sampling parallel data if bt data is included (Note: NOT sample weight)", default=0.33)
        parser.add_argument('--word-noise-prob', type=float, default=0.01)
        parser.add_argument('--fragment-noise-prob', type=float, default=0.01)
        parser.add_argument('--max-merges', type=int, default=10, help="How many segments are at most merged into a single training example.")
        parser.add_argument('--max-shuffle-dist', type=int, default=3)
        ### character noising
        parser.add_argument('--char-swap-prob', type=float, default=0.01)
        parser.add_argument('--char-delete-prob', type=float, default=0.01)
        parser.add_argument('--char-insert-prob', type=float, default=0.01)
        parser.add_argument('--char-duplicate-prob', type=float, default=0.01)
        parser.add_argument('--char-case-prob', type=float, default=0.01)
        parser.add_argument('--char-substitution-prob', type=float, default=0.01)
        parser.add_argument('--seq-lower-prob', type=float, default=0.01)
        parser.add_argument('--seq-upper-prob', type=float, default=0.01)
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.src_dict: Dictionary = self.src_dict
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
        all_dataset_names = sorted(set(split.split(",")))  # split.split(",")
        bt_dataset_names = self.args.bt_subset.split(",")
        align_dataset_names = self.args.align_subset.split(",")
        parallel_dataset_names = [
            name for name in all_dataset_names if name not in bt_dataset_names and name not in align_dataset_names
        ]

        # infer langcode and translation direction
        src, tgt = self.args.source_lang, self.args.target_lang
        direction = f"{src}-{tgt}"

        # XXX: since this is at document level, we probably dont want to apply this too aggressively
        # XXX: e.g. only enable a noiser with some probability
        assert (
            self.args.max_sequence_length <= self.args.max_source_positions
        ), "The maximum training sequence length should be lesser than the positional encoding."
        max_seq_len = self.args.max_sequence_length

        logger.info(f"Max sequence length={max_seq_len}")
        logger.info(f"Max merges={self.args.max_merges}")

        bpe = encoders.build_bpe(self.args)
        from greynirseq.nicenlp.data.encoders import Encoder

        my_enc = Encoder(
            self.args,
            self.src_dict,
            min_val=self.src_dict.nspecial,
            max_val=len(self.src_dict) - 1 - len(self.langs),
        )

        def decode(example):
            src_string = self.src_dict.string(example["source"])
            tgt_string = self.src_dict.string(example["target"])
            # decoded = bpe.decode(spm_string)
            print()
            print(bpe.decode(src_string))
            print()
            print(bpe.decode(tgt_string))
            print()

        def create_path(name, lang, align=False):
            return f"{data_dir_path}/{name}.{direction}.{lang if not align else 'align'}.jsonl"

        logger.info(f"Split name {split}")

        if split in self.args.valid_subset:
            src_paths = [f"{data_dir_path}/{name}.{direction}.{src}.jsonl" for name in parallel_dataset_names]
            tgt_paths = [f"{data_dir_path}/{name}.{direction}.{tgt}.jsonl" for name in parallel_dataset_names]
            parallel_dataset = IndexedParallelDocumentsDataset.from_parallel_jsonl_many(
                src_paths,
                tgt_paths,
                bpe,
                self.src_dict,
                encoder=my_enc,
                max_seq_len=max_seq_len,
                max_merges=self.args.max_merges,
                append_source_id=self.src_dict.index("[{}]".format(self.args.source_lang)),  # 250_004 for english
                append_target_id=self.tgt_dict.index("[{}]".format(self.args.target_lang)),  # 250_012 for icelandic
                num_proc=self.args.num_preprocess_workers,
                seed=self.args.seed,
            )
            self.datasets[split] = parallel_dataset
            return parallel_dataset

        bt_datasets = []
        parallel_datasets = []
        for dataset_name in all_dataset_names:
            if dataset_name == "":
                continue
            alignment_path = None
            src_path = create_path(name=dataset_name, lang=src)
            tgt_path = create_path(name=dataset_name, lang=tgt)
            if dataset_name in align_dataset_names:
                alignment_path = create_path(lang="none", name=dataset_name, align=True)

            dataset = IndexedParallelDocumentsDataset.from_parallel_jsonl_many(
                src_path,
                tgt_path,
                bpe,
                self.src_dict,
                encoder=my_enc,
                max_seq_len=max_seq_len,
                max_merges=self.args.max_merges,
                append_source_id=self.src_dict.index("[{}]".format(self.args.source_lang)),  # 250_004 for english
                append_target_id=self.tgt_dict.index("[{}]".format(self.args.target_lang)),  # 250_012 for icelandic
                align_paths=alignment_path,
                num_proc=self.args.num_preprocess_workers,
                seed=self.args.seed,
            )
            if dataset_name in bt_dataset_names:
                bt_datasets.append(dataset)
            else:
                parallel_datasets.append(dataset)

        # we already handled this
        valid_dataset_names = self.args.valid_subset
        if split in valid_dataset_names:
            assert False

        dataset = IndexedParallelBTDocumentsDataset(
            parallel_datasets,
            bt_datasets,
            self.src_dict,
            encoder=my_enc,
            append_source_id=self.src_dict.index("[{}]".format(self.args.source_lang)),
            append_target_id=self.tgt_dict.index("[{}]".format(self.args.target_lang)),
            parallel_prob=self.args.parallel_prob,
            seed=self.args.seed,
            max_seq_len=max_seq_len,
            max_merges=self.args.max_merges,
            num_proc=self.args.num_preprocess_workers,
        )
        # often a trainer will check to see if dataset is empty before training/validation
        dataset.set_epoch(1)

        logger.info("Dataset loading done.")
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

        logger.debug("Batching by size...")

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
