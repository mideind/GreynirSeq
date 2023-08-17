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

import copy
import glob
import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Dict, cast

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
    max_sentences: int = II("dataset.batch_size")
    no_merge_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of not merging a segment"},
    )
    dict_path: str = field(default="", metadata={"help": "Path to the dictionary"})
    data_language_mappings: str = field(
        default="",
        metadata={
            "help": "Comma separated list of language mappings from the JSONL language to the language string expected by the model, e.g. 'is:is_IS,en:en_XX'"
        },
    )


@register_task("document_translation_from_pretrained_bart", dataclass=DocumentTranslationFromPretrainedBARTConfig)
class DocumentTranslationFromPretrainedBART(TranslationFromPretrainedBARTTask):
    """Task for training multi sentence translation models from pre-trained BART models."""

    def __init__(
        self, cfg: DocumentTranslationFromPretrainedBARTConfig, the_dict: Dictionary, language_mappings: Dict[str, str]
    ):
        super().__init__(cfg, src_dict=the_dict, tgt_dict=copy.deepcopy(the_dict))
        # this is for typing only
        self.the_dict = the_dict
        # TODO: This is a temp hack for NLLB-200
        # the_dict.add_symbol("<mask1>")
        # the_dict.add_symbol("<mask2>")
        # self.tgt_dict = the_dict
        # Hack done
        self.language_mappings = language_mappings

    @classmethod
    def setup_task(cls, cfg: DocumentTranslationFromPretrainedBARTConfig, **kwargs):
        """Setup the task (e.g., load dictionaries).

        Args:
            args (argparse.Namespace): parsed command-line arguments
        """
        if cfg.dict_path == "":
            raise ValueError("Must specify a dictionary path")
        if cfg.data_language_mappings == "":
            raise ValueError("Must specify languages to train on")
        the_dict = cls.load_dictionary(cfg.dict_path)
        logger.info("dictionary: {} types".format(len(the_dict)))
        # langcode and translation direction
        language_mappings_pairs = cfg.data_language_mappings.split(",")
        language_mappings = {}
        for pair in language_mappings_pairs:
            parts = pair.split(":")
            if len(parts) != 2:
                raise ValueError("Invalid language mapping: {}".format(pair))
            language_mappings[parts[0]] = parts[1]

        return cls(cfg, the_dict, language_mappings)

    def load_dataset(self, split: str, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        This function is called once per train split (e.g., "train") and multiple times per validation split.
        This function has deviated a lot from the original fairseq implementation.

        Args:
            split (str): The value of --train-subset OR
                --valid-subset which has been split on "," OR
                --test-subset which has been split on "," CLI args.
                Each is a comma-separated list of dataset names.
                This method is called separately for each subset.
        """
        self.cfg = cast(DocumentTranslationFromPretrainedBARTConfig, self.cfg)
        # this is for sharding
        paths = utils.split_paths(self.cfg.data)  # type: ignore
        assert len(paths) > 0
        data_dir_path = paths[(epoch - 1) % len(paths)]

        # if a split contains a comma, we should crash - since that is no longer supported
        assert "," not in split, "Split should not contain a comma"
        # now lets list all the 'jsonl' files in the data dir
        jsonl_files = glob.glob(os.path.join(data_dir_path, "*.jsonl"))
        jsonl_files_paths = [pathlib.Path(path) for path in jsonl_files]
        # now we gather all the files which start with the same string as the split
        split_files = [path for path in jsonl_files_paths if path.stem.startswith(split)]
        for path in split_files:
            logger.info(f"Found file {path}")

        def metadata_from_filename(path: pathlib.Path):
            """Extract translation metadata from filename.

            There are 3 types of files:
            1. {name}.{lang1}-{lang2}.{lang1}.jsonl
            2. {name}.{lang1}-{lang2}.{lang2}.jsonl
            3. {name}.{lang1}-{lang2}.align.jsonl [optional]
            """
            name, direction, file_type = path.stem.split(".")  # .stem removes the jsonl extension
            lang1, lang2 = direction.split("-")
            assert lang1 in self.language_mappings
            assert lang2 in self.language_mappings
            assert file_type in [lang1, lang2, "align"]
            print(self.the_dict.index(self.language_mappings[lang1]), lang1)
            print(self.the_dict.index(self.language_mappings[lang2]), lang2)
            if file_type == "align":
                file_type = "align"
            elif file_type == lang1:
                file_type = "src"
            elif file_type == lang2:
                file_type = "tgt"
            return {
                "name": name,
                "direction": direction,
                "type": file_type,
                "path": str(path),
            }

        datasets_metadata = [metadata_from_filename(path) for path in split_files]
        # group the datasets by name and direction
        datasets_by_name_and_direction = {}
        for dataset_metadata in datasets_metadata:
            name = dataset_metadata["name"]
            direction = dataset_metadata["direction"]
            if name + direction not in datasets_by_name_and_direction:
                datasets_by_name_and_direction[name + direction] = []
            datasets_by_name_and_direction[name + direction].append(dataset_metadata)

        bt_dataset_names = self.cfg.bt_subset.split(",")

        logger.info(datasets_by_name_and_direction)
        # We combine the lists into a single entity with the same name and direction
        datasets_for_loading = {}
        for name_direction, datasets in datasets_by_name_and_direction.items():
            assert len(datasets) <= 3, "There should be at most 3 files per dataset"
            assert len(datasets) >= 2, "There should be at least 2 files per dataset"
            src_dataset = [dataset for dataset in datasets if dataset["type"] == "src"][0]
            tgt_dataset = [dataset for dataset in datasets if dataset["type"] == "tgt"][0]
            align_datasets = [dataset for dataset in datasets if dataset["type"] == "align"]
            if len(align_datasets) == 0:
                align_dataset = None
            else:
                align_dataset = align_datasets[0]

            datasets_for_loading[name_direction] = {
                "name": datasets[0]["name"],
                "src_path": src_dataset["path"],
                "tgt_path": tgt_dataset["path"],
                "align_path": align_dataset["path"] if align_dataset is not None else None,
                "is_bt": datasets[0]["name"] in bt_dataset_names,
            }

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
            dictionary=self.the_dict,
            bpe=bpe,
            noisy_bpe=noisy_bpe,
            allowed_dictionary_min=self.the_dict.nspecial,
            allowed_dictionary_max=len(self.the_dict) - 1 - len(self.langs),  # type: ignore
            fragment_noise_prob=self.cfg.fragment_noise_prob,
            global_skip_noise_prob=self.cfg.global_skip_noise_prob,
            word_noise_config=self.cfg.word_noise_config,
            char_noise_config=self.cfg.char_noise_config,
        )

        def decode(example):
            src_string = self.the_dict.string(example["source"])
            tgt_string = self.the_dict.string(example["target"])
            # decoded = bpe.decode(spm_string)
            print()
            print(bpe.decode(src_string))
            print()
            print(bpe.decode(tgt_string))
            print()

        datasets = []
        for _, dataset_values in datasets_for_loading.items():
            dataset = IndexedParallelDocumentsDataset.from_parallel_jsonl(
                name=dataset_values["name"],
                is_bt=dataset_values["is_bt"],
                src_path=dataset_values["src_path"],
                tgt_path=dataset_values["tgt_path"],
                bpe_encoder=bpe,
                dictionary=self.the_dict,
                encoder=my_enc,
                data_language_mapper=self.language_mappings,
                max_seq_len=max_seq_len,
                max_merges=self.cfg.max_merges,
                align_path=dataset_values["align_path"],
                num_proc=self.cfg.num_preprocess_workers,
                seed=self.cfg.seed,
            )
            datasets.append(dataset)

        if len(datasets) != 1:
            parallel_datasets = [dataset for dataset in datasets if not dataset.is_bt]
            bt_datasets = [dataset for dataset in datasets if dataset.is_bt]

            dataset = IndexedParallelBTDocumentsDataset(
                parallel_datasets,
                bt_datasets,
                self.the_dict,
                encoder=my_enc,
                parallel_prob=self.cfg.parallel_prob,
                seed=self.cfg.seed,
                max_seq_len=max_seq_len,
                max_merges=self.cfg.max_merges,
                num_proc=self.cfg.num_preprocess_workers,
                no_merge_prob=self.cfg.no_merge_prob,
            )
        else:
            dataset = datasets[0]

        dataset.set_epoch(1)
        logger.info("Dataset loading done.")
        self.datasets[split] = dataset
        logger.info(f"split dataset: {dataset}")
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
        logger.info(f"Batching by size... with max_tokens={max_tokens} and max_sentences={max_sentences}")
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
        lengths = dataset.ordered_sizes()  # type: ignore

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
