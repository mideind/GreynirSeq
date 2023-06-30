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
from dataclasses import dataclass, field
from typing import cast

from fairseq import utils
from fairseq.data import Dictionary, FairseqDataset, data_utils, iterators
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE, SentencepieceConfig
from fairseq.tasks import register_task
from omegaconf import II

from greynirseq.nicenlp.data.batch_sampler import batch_by_size
from greynirseq.nicenlp.data.char_noise import CharacterNoiserConfig
from greynirseq.nicenlp.data.parallel_documents.indexed_parallel_bt_documents_dataset import (
    IndexedParallelBTDocumentsDataset,
)
from greynirseq.nicenlp.data.parallel_documents.indexed_parallel_documents_dataset import (
    IndexedParallelDocumentsDataset,
)
from greynirseq.nicenlp.data.word_noise import WordNoiserConfig

from .translation_from_pretrained_bart import TranslationFromPretrainedBARTConfig, TranslationFromPretrainedBARTTask

logger = logging.getLogger(__name__)


@dataclass
class DocumentTranslationFromPretrainedBARTConfig(TranslationFromPretrainedBARTConfig):
    max_sequence_length: int = field(
        default=int(1024 * 0.75),
        metadata={"help": "max sequence length"},
    )
    num_preprocess_workers: int = field(
        default=2,
        metadata={"help": "number of workers to preprocess the data"},
    )
    bt_subset: str = field(
        default="",
        metadata={"help": "comma separated list of subsets to use for backtranslation"},
    )
    align_subset: str = field(
        default="",
        metadata={"help": "The subset of parallel data that has requires an alignment jsonl file"},
    )
    parallel_prob: float = field(
        default=0.33,
        metadata={"help": "Probability of sampling parallel data if bt data is included (Note: NOT sample weight)"},
    )
    fragment_noise_prob: float = field(
        default=0.01,
        metadata={"help": "Probability of fragment noise"},
    )
    max_merges: int = field(
        default=10,
        metadata={"help": "How many segments are at most merged into a single training example."},
    )
    global_skip_noise_prob: float = field(
        default=0.10,
        metadata={"help": "Probability of skipping a segment"},
    )
    word_noise_config: WordNoiserConfig = field(
        default=WordNoiserConfig(),
        metadata={"help": "Word noising config"},
    )
    char_noise_config: CharacterNoiserConfig = field(
        default=CharacterNoiserConfig(),
        metadata={"help": "Character noising config"},
    )
    spm_model: str = II("bpe.sentencepiece_model")
    valid_subset: str = II("dataset.valid_subset")
    seed: int = II("common.seed")


@register_task("document_translation_from_pretrained_bart", dataclass=DocumentTranslationFromPretrainedBARTConfig)
class DocumentTranslationFromPretrainedBART(TranslationFromPretrainedBARTTask):
    """Task for training multi sentence translation models from pre-trained BART models."""

    def __init__(self, cfg: DocumentTranslationFromPretrainedBARTConfig, src_dict: Dictionary, tgt_dict: Dictionary):
        super().__init__(cfg, src_dict=src_dict, tgt_dict=tgt_dict)
        # this is for typing only
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict

    def load_dataset(self, split: str, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): The value of --train-subset OR
                --valid-subset which has been split on "," OR
                --test-subset which has been split on "," CLI args.
                Each is a comma-separated list of dataset names.
                This method is called separately for each subset.
        """
        logger.info(f"Split name {split}")
        self.cfg = cast(DocumentTranslationFromPretrainedBARTConfig, self.cfg)
        # this is for sharding
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_dir_path = paths[(epoch - 1) % len(paths)]

        # all datasets in this split
        split_dataset_names = sorted(set(split.split(",")))
        # all bt datasets defined for this task
        bt_dataset_names = self.cfg.bt_subset.split(",")
        # all alignment datasets defined for this task
        align_dataset_names = self.cfg.align_subset.split(",")
        # the parallel (1-to-1) datasets are the ones that are not bt or alignment datasets
        parallel_dataset_names = [
            name for name in split_dataset_names if name not in bt_dataset_names and name not in align_dataset_names
        ]

        # langcode and translation direction
        src, tgt = self.cfg.source_lang, self.cfg.target_lang
        direction = f"{src}-{tgt}"
        if src is None or tgt is None:
            raise ValueError("--source-lang and --target-lang must be set")

        # sanity checks
        assert (
            self.cfg.max_sequence_length <= self.cfg.max_source_positions
        ), "The maximum training sequence length should be lesser than the positional encoding."
        max_seq_len = self.cfg.max_sequence_length

        logger.info(f"Max sequence length={max_seq_len}")
        logger.info(f"Max merges={self.cfg.max_merges}")
        print(self.cfg)

        bpe = SentencepieceBPE(SentencepieceConfig(sentencepiece_model=self.cfg.spm_model))
        noisy_bpe = SentencepieceBPE(
            SentencepieceConfig(sentencepiece_model=self.cfg.spm_model, sentencepiece_enable_sampling=True)
        )
        from greynirseq.nicenlp.data.encoders import Encoder

        my_enc = Encoder(
            dictionary=self.src_dict,
            bpe=bpe,
            noisy_bpe=noisy_bpe,
            allowed_dictionary_min=self.src_dict.nspecial,
            allowed_dictionary_max=len(self.src_dict) - 1 - len(self.langs),
            fragment_noise_prob=self.cfg.fragment_noise_prob,
            global_skip_noise_prob=self.cfg.global_skip_noise_prob,
            word_noise_config=self.cfg.word_noise_config,
            char_noise_config=self.cfg.char_noise_config,
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

        def create_path(name: str, lang: str, align=False) -> str:
            return f"{data_dir_path}/{name}.{direction}.{lang if not align else 'align'}.jsonl"

        if split in self.cfg.valid_subset:
            src_paths = [f"{data_dir_path}/{name}.{direction}.{src}.jsonl" for name in parallel_dataset_names]
            tgt_paths = [f"{data_dir_path}/{name}.{direction}.{tgt}.jsonl" for name in parallel_dataset_names]
            parallel_dataset = IndexedParallelDocumentsDataset.from_parallel_jsonl_many(
                src_paths,
                tgt_paths,
                bpe,
                self.src_dict,
                encoder=my_enc,
                max_seq_len=max_seq_len,
                max_merges=self.cfg.max_merges,
                append_source_id=self.src_dict.index("[{}]".format(self.cfg.source_lang)),  # 250_004 for english
                append_target_id=self.tgt_dict.index("[{}]".format(self.cfg.target_lang)),  # 250_012 for icelandic
                num_proc=self.cfg.num_preprocess_workers,
                seed=self.cfg.seed,
            )
            self.datasets[split] = parallel_dataset
            return parallel_dataset

        bt_datasets = []
        parallel_datasets = []
        for dataset_name in split_dataset_names:
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
                max_merges=self.cfg.max_merges,
                append_source_id=self.src_dict.index("[{}]".format(self.cfg.source_lang)),  # 250_004 for english
                append_target_id=self.tgt_dict.index("[{}]".format(self.cfg.target_lang)),  # 250_012 for icelandic
                align_paths=alignment_path,
                num_proc=self.cfg.num_preprocess_workers,
                seed=self.cfg.seed,
            )
            if dataset_name in bt_dataset_names:
                bt_datasets.append(dataset)
            else:
                parallel_datasets.append(dataset)

        # we already handled this
        valid_dataset_names = self.cfg.valid_subset
        if split in valid_dataset_names:
            assert False

        dataset = IndexedParallelBTDocumentsDataset(
            parallel_datasets,
            bt_datasets,
            self.src_dict,
            encoder=my_enc,
            append_source_id=self.src_dict.index("[{}]".format(self.cfg.source_lang)),
            append_target_id=self.tgt_dict.index("[{}]".format(self.cfg.target_lang)),
            parallel_prob=self.cfg.parallel_prob,
            seed=self.cfg.seed,
            max_seq_len=max_seq_len,
            max_merges=self.cfg.max_merges,
            num_proc=self.cfg.num_preprocess_workers,
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
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        max_sentences = max_sentences or self.cfg.max_sentences
        logger.info(f"S Batching by size... with max_tokens={max_tokens} and max_sentences={max_sentences}")
        if not hasattr(dataset, "ordered_sizes"):
            logger.info("FOOFOO")
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
                skip_remainder_batch=skip_remainder_batch,
                grouped_shuffling=grouped_shuffling,
                update_epoch_batch_itr=update_epoch_batch_itr,
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

        logger.info(f"Batching by size... with max_tokens={max_tokens} and max_sentences={max_sentences}")
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
            skip_remainder_batch=skip_remainder_batch,
            grouped_shuffling=grouped_shuffling,
        )

        if can_reuse_epoch_itr:
            self.dataset_to_epoch_iter[dataset] = epoch_iter

        return epoch_iter

    def can_reuse_epoch_itr(self, dataset):
        return False
