import logging
import os
from pathlib import Path
import json
from collections import namedtuple

import numpy as np

import torch
from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    NumSamplesDataset,
    NumelDataset,
    RightPadDataset,
    SortDataset,
    TruncateDataset,
    PrependTokenDataset,
)
from fairseq.data import encoders
from fairseq.tasks import FairseqTask, register_task
from fairseq.tasks.sentence_prediction import SentencePredictionTask

try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from greynirseq.nicenlp.data.datasets import (
    DynamicLabelledSpanDataset,
    LabelledSpanDataset,
    SpanDataset,
    SparseProductSpanDataset,
    ProductSpanDataset,
    NumSpanDataset,
    NestedDictionaryDatasetFix,
    LossMaskDataset,
    NumWordsDataset,
)
import greynirseq.nicenlp.utils.greynir.greynir_utils as greynir_utils

logger = logging.getLogger(__name__)


def parse_label_schema(path):
    LabelSchema = namedtuple(
        "LabelSchema",
        [
            "labels",
            "group_name_to_labels",
            "label_categories",
            "category_to_group_names",
            "separator",
            "group_names",
            "null",
            "null_leaf",
        ],
    )
    with open(path, "r") as fp:
        j_obj = json.load(fp)
    return LabelSchema(**j_obj)


@register_task("multi_span_prediction")
class MultiSpanPredictionTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument(
            "--nonterm-schema",
            metavar="FILE",
            help="json file providing label-set and label-groups \
                (mutually exclusive labels)",
            required=True,
        )
        # parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument("--no-shuffle", action="store_true", default=False)

    def __init__(
        self, args, data_dictionary, label_dictionary, is_word_initial, label_schema
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

        self.label_schema = label_schema
        self.num_labels = len(self.label_schema.labels)

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_dict = cls.load_dictionary(args, os.path.join(args.data, "dict.txt"))
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        is_word_initial = cls.get_word_beginnings(args, data_dict)

        label_dict, label_schema = cls.load_label_dictionary(args, args.nonterm_schema)
        logger.info("[label] dictionary: {} types".format(len(label_dict)))
        return MultiSpanPredictionTask(
            args, data_dict, label_dict, is_word_initial, label_schema
        )

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

        assert (
            label_dict.symbols[label_dict.nspecial] == "NULL"
        ), "Expected first nonspecial token to be 'NULL'"
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

        # TODO: it does not make sense to parse partial sequences
        # if self.args.truncate_sequence:
        #     src_tokens = TruncateDataset(src_tokens, self.args.max_positions)

        targets_path = Path(self.args.data) / "{}.nonterm".format(split)
        labelled_spans = data_utils.load_indexed_dataset(
            str(targets_path),
            self.label_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        assert labelled_spans is not None, "could not find labels: {}".format(
            targets_path
        )

        target_spans = DynamicLabelledSpanDataset(
            labelled_spans,
            self.label_dictionary,
            rebinarize_fn=greynir_utils.rebinarize,
            seed=self.args.seed,
            return_spans=True,
        )
        labels = DynamicLabelledSpanDataset(
            labelled_spans,
            self.label_dictionary,
            rebinarize_fn=greynir_utils.rebinarize,
            seed=self.args.seed,
            return_spans=False,
        )

        # all possible word spans in each sequence
        word_spans = SpanDataset(src_tokens, self.is_word_initial)
        all_spans = ProductSpanDataset(word_spans)
        # all_spans = SparseProductSpanDataset(spans)

        # loss_masks = LossMaskDataset(
        #     labels,
        #     {
        #         idx: torch.Tensor(mask)
        #         for (idx, mask) in LABEL_IDX_TO_GROUP_MASK.items()
        #     },
        # )

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens, pad_idx=self.source_dictionary.pad()
                ),
                "nsrc_tokens": NumelDataset(src_tokens),
                "src_spans": RightPadDataset(
                    all_spans, pad_idx=self.label_dictionary.pad()
                ),
                "nsrc_spans": NumSpanDataset(all_spans),
            },
            "targets": RightPadDataset(labels, pad_idx=self.label_dictionary.pad()),
            "target_spans": RightPadDataset(
                target_spans, pad_idx=self.label_dictionary.pad()
            ),
            "ntargets": NumelDataset(labels),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "nwords": NumWordsDataset(src_tokens, is_word_initial=self.is_word_initial),
            "word_spans": RightPadDataset(
                word_spans, pad_idx=self.label_dictionary.pad()
            ),
            # "loss_masks": loss_masks,
        }

        nested_dataset = NestedDictionaryDatasetFix(dataset, sizes=[src_tokens.sizes])

        if self.args.no_shuffle or True:
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

        # num_classes_mutex = len(self._label_dictionary)
        # num_classes_binary = 0

        # if hasattr(self.args, "num_classes_mutex"):
        #     num_classes_mutex = self.args.num_classes_mutex

        # if hasattr(self.args, "num_classes_binary"):
        #     num_classes_binary = self.args.num_classes_binary

        model.register_classification_head(
            "multi_span_classification",
            # num_cats=len(self.label_schema.label_categories),
            num_cats=None,
            num_labels=self.num_labels,
        )

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
