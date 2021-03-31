# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

# flake8: noqa

import json
import logging
import os
from collections import namedtuple
from pathlib import Path

import numpy as np
import torch
from fairseq.data import (
    Dictionary,
    IdDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TruncateDataset,
    data_utils,
    encoders,
)
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.sentence_prediction import SentencePredictionTask

from greynirseq.nicenlp.data.datasets import (
    DynamicLabelledSpanDataset,
    LabelledSpanDataset,
    NestedDictionaryDatasetFix,
    NumSpanDataset,
    NumWordsDataset,
    ProductSpanDataset,
    WordSpanDataset,
)
from greynirseq.nicenlp.utils.label_schema.label_schema import (
    label_schema_as_dictionary,
    make_group_masks,
    make_vec_idx_to_dict_idx,
    parse_label_schema,
)

logger = logging.getLogger(__name__)


@register_task("multi_span_prediction")
class MultiSpanPredictionTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument(
            "--label-schema",
            metavar="FILE",
            help="json file providing label-set and label-groups \
                (mutually exclusive labels)",
            required=True,
        )
        parser.add_argument("--no-shuffle", action="store_true", default=False)

    def __init__(self, args, data_dictionary, label_dictionary, is_word_initial, label_schema):
        super().__init__(args)
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary
        if not hasattr(args, "max_positions"):
            self._max_positions = (args.max_source_positions, args.max_target_positions)
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions
        self.is_word_initial = is_word_initial

        self.label_schema = label_schema
        self.num_labels = len(self.label_schema.labels)

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_dict = cls.load_dictionary(args, os.path.join(args.data, "dict.txt"))
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        is_word_initial = cls.get_word_beginnings(args, data_dict)

        label_dict, label_schema = cls.load_label_dictionary(args, args.nonterm_schema)
        logger.info("[label] dictionary: {} types".format(len(label_dict)))
        return MultiSpanPredictionTask(args, data_dict, label_dict, is_word_initial, label_schema)

    @classmethod
    def load_label_dictionary(cls, args, filename, **kwargs):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
        """
        label_schema = parse_label_schema(filename)
        label_dict = Dictionary()

        labels = list(label_schema.labels)
        assert labels[0] == "NULL", "Expected label at index 0 to be 'NULL'"
        assert len(labels) == len(set(labels))

        for label in labels:
            label_dict.add_symbol(label)

        assert label_dict.symbols[label_dict.nspecial] == "NULL", "Expected first nonspecial token to be 'NULL'"
        return label_dict, label_schema

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def get_word_beginnings(cls, args, dictionary):
        bpe = encoders.build_bpe(args)
        if bpe is not None:

            def is_beginning_of_word(i):
                if i < dictionary.nspecial:
                    return True
                tok = dictionary[i]
                if tok.startswith("madeupword"):
                    return True
                try:
                    return bpe.is_beginning_of_word(tok)
                except ValueError:
                    return True

            is_word_initial = {}
            for i in range(len(dictionary)):
                is_word_initial[i] = int(is_beginning_of_word(i))
            return is_word_initial
        return None

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        inputs_path = Path(self.args.data) / "{split}".format(split=split)
        src_tokens = data_utils.load_indexed_dataset(
            str(inputs_path),
            self.source_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        assert src_tokens is not None, "could not find dataset: {}".format(inputs_path)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        src_tokens = PrependTokenDataset(src_tokens, self.source_dictionary.bos())

        targets_path = Path(self.args.data) / "{}.nonterm".format(split)
        labelled_spans = data_utils.load_indexed_dataset(
            str(targets_path),
            self.label_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        assert labelled_spans is not None, "could not find labels: {}".format(targets_path)

        raise NotImplementedError

        target_spans = LabelledSpanDataset(labelled_spans, return_spans=True)
        labels = LabelledSpanDataset(labelled_spans, return_spans=False)

        # all possible word spans in each sequence
        word_spans = WordSpanDataset(src_tokens, self.source_dictionary, self.is_word_initial)
        all_spans = ProductSpanDataset(word_spans)

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(src_tokens, pad_idx=self.source_dictionary.pad()),
                "nsrc_tokens": NumelDataset(src_tokens),
                "src_spans": RightPadDataset(all_spans, pad_idx=self.label_dictionary.pad()),
                "nsrc_spans": NumSpanDataset(all_spans),
            },
            "targets": RightPadDataset(labels, pad_idx=self.label_dictionary.pad()),
            "target_spans": RightPadDataset(target_spans, pad_idx=self.label_dictionary.pad()),
            "ntargets": NumelDataset(labels),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "nwords": NumWordsDataset(src_tokens, self.dictionary, self.is_word_initial),
            "word_spans": RightPadDataset(word_spans, pad_idx=self.label_dictionary.pad()),
        }

        nested_dataset = NestedDictionaryDatasetFix(dataset, sizes=[src_tokens.sizes])

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(nested_dataset, sort_order=[shuffle])

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)
        model.task = self

        raise NotImplementedError
        # TODO: This is incorrect
        # model.register_classification_head(
        #     "multi_span_classification",
        #     # num_cats=len(self.label_schema.label_categories),
        #     num_cats=None,
        #     num_labels=self.num_labels,
        # )

        return model

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
