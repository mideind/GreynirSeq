# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import argparse
import logging
import os
from pathlib import Path

import numpy as np
from fairseq.data import (
    Dictionary,
    IdDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    data_utils,
)
from fairseq.tasks import FairseqTask, register_task

from greynirseq.nicenlp.data.datasets import (
    NestedDictionaryDatasetFix,
    NoBosEosDataset,
    NumWordsDataset,
    WordEndMaskDataset,
)
from greynirseq.nicenlp.data.encoding import get_word_beginnings

logger = logging.getLogger(__name__)


@register_task("multi_class_token_classification_task")
class MultiClassTokenClassificationTask(FairseqTask):
    def __init__(
        self,
        args: argparse.Namespace,
        data_dictionary: Dictionary,
        label_dictionary: Dictionary,
        is_word_initial: dict,
    ):
        super().__init__(args)
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary

        if not hasattr(args, "max_positions"):
            self._max_positions = (args.max_source_positions, args.max_target_positions)
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions
        self.is_word_initial = is_word_initial
        self.num_labels = len(self._label_dictionary.symbols)

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", metavar="FILE", help="file prefix for data")

    @classmethod
    def setup_task(cls, args: argparse.Namespace, **kwargs):
        data_dict = Dictionary.load(os.path.join(args.data, "dict.txt"))
        data_dict.add_symbol("<mask>")
        logger.info("[input] dictionary: {} types".format(len(data_dict)))
        is_word_initial = get_word_beginnings(args, data_dict)
        term_dict = Dictionary.load(os.path.join(args.data, "dict_term.txt"))
        logger.info("[label] dictionary: {} types".format(len(term_dict)))
        return MultiClassTokenClassificationTask(args, data_dict, term_dict, is_word_initial)

    def load_dataset(self, split: str, combine: bool = False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        inputs_path = Path(self.args.data) / "{split}".format(split=split)
        src_tokens = data_utils.load_indexed_dataset(
            str(inputs_path), self.source_dictionary, self.args.dataset_impl, combine=combine
        )
        assert src_tokens is not None, "could not find dataset: {}".format(inputs_path)

        src_tokens = PrependTokenDataset(src_tokens, self.source_dictionary.bos())
        targets_path = Path(self.args.data) / "{}.term".format(split)
        labels = data_utils.load_indexed_dataset(
            str(targets_path), self._label_dictionary, self.args.dataset_impl, combine=combine
        )
        assert labels is not None, "could not find labels: {}".format(targets_path)
        clean_labels = NoBosEosDataset(labels, self.label_dictionary)
        word_mask = WordEndMaskDataset(src_tokens, self.dictionary, self.is_word_initial, bos_value=0, eos_value=0)
        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(src_tokens, pad_idx=self.source_dictionary.pad()),
                "nsrc_tokens": NumelDataset(src_tokens),
                "word_mask": RightPadDataset(word_mask, pad_idx=0),  # pad is zero since mask
            },
            "target_attrs": RightPadDataset(clean_labels, pad_idx=self.label_dictionary.pad()),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "nwords": NumWordsDataset(src_tokens, self.dictionary, self.is_word_initial),
        }

        nested_dataset = NestedDictionaryDatasetFix(dataset, sizes=[src_tokens.sizes])

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        dataset = SortDataset(nested_dataset, sort_order=[shuffle])

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))
        self.datasets[split] = dataset
        return self.datasets[split]

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
