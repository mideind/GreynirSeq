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
    POSDataset,
    WordEndMaskDataset,
    GroupMaskDataset,
    RightPad2dDataset,
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


def label_schema_as_dictionary(label_schema):
    label_dict = Dictionary()

    labels = list(label_schema.labels)
    assert len(labels) == len(set(labels))

    for label in labels:
        label_dict.add_symbol(label)

    return label_dict


def make_group_masks(schema, dictionary, device="cpu"):
    num_groups = len(schema.group_names)
    label_shift = dictionary.nspecial
    num_labels = len(dictionary) - label_shift
    ret_mask = torch.zeros(num_labels, num_groups, dtype=torch.int64, device=device)
    for cat, cat_group_names in schema.category_to_group_names.items():
        cat_label_idx = dictionary.index(cat)
        cat_vec_idx = schema.label_categories.index(cat)
        for group_name in cat_group_names:
            ret_mask[cat_vec_idx, schema.group_names.index(group_name)] = 1
        assert cat_label_idx != dictionary.unk()
    for cat in schema.label_categories:
        cat_label_idx = dictionary.index(cat)
        assert cat_label_idx != dictionary.unk()
    return ret_mask


@register_task("pos_ice")
class POSTask(FairseqTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument(
            "--term-schema",
            metavar="FILE",
            help="json file providing label-set and label-groups",
            required=True,
        )
        parser.add_argument("--no-shuffle", action="store_true", default=False)

    def __init__(
        self, args, data_dictionary, label_dictionary, is_word_initial, label_schema
    ):
        super().__init__(args)
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary
        self._label_dictionary.sep = lambda: self._label_dictionary.index(
            label_schema.separator
        )
        assert self._label_dictionary.index("<mask>") == self._label_dictionary.unk()
        assert self._label_dictionary.sep() != self._label_dictionary.unk()
        if not hasattr(args, "max_positions"):
            self._max_positions = (args.max_source_positions, args.max_target_positions)
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions
        self.is_word_initial = is_word_initial

        self.label_schema = label_schema
        self.num_cats = len(self.label_schema.label_categories)
        self.num_groups = len(self.label_schema.group_name_to_labels.keys())
        self.num_labels = len(self.label_schema.labels)

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_dict = cls.load_dictionary(args, os.path.join(args.data, "dict.txt"))
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        is_word_initial = cls.get_word_beginnings(args, data_dict)
        term_dict = cls.load_dictionary(
            args, os.path.join(args.data, "dict_term.txt"), add_mask=False
        )

        # label_dict, label_schema = cls.load_label_dictionary(args, args.term_schema)
        _, label_schema = cls.load_label_dictionary(args, args.term_schema)
        logger.info("[label] dictionary: {} types".format(len(term_dict)))

        seen = set()
        for idx, lbl in enumerate(term_dict.symbols):
            exists = lbl in label_schema.labels
            seen.add(lbl)
            if not exists and idx > term_dict.nspecial and lbl != "<mask>":
                assert False, "Unexpected POS label item in term_dict.txt: {}".format(
                    lbl
                )
        for lbl in label_schema.labels:
            if lbl in seen:
                continue
            assert False, "Unexpected POS label item in label_schema {}".format(lbl)

        return POSTask(args, data_dict, term_dict, is_word_initial, label_schema)

    @classmethod
    def load_label_dictionary(cls, args, filename, **kwargs):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
        """
        label_schema = parse_label_schema(filename)

        return label_schema_as_dictionary(label_schema), label_schema

    @classmethod
    def load_dictionary(cls, args, filename, add_mask=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        if add_mask:
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

        targets_path = Path(self.args.data) / "{}.term".format(split)
        term_labels = data_utils.load_indexed_dataset(
            str(targets_path),
            self.label_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        assert term_labels is not None, "could not find labels: {}".format(targets_path)

        term_cats, term_attrs = POSDataset.make_both(
            term_labels,
            self.label_dictionary,
        )

        word_masks_w_bos = WordEndMaskDataset(
            src_tokens,
            self.is_word_initial,
            include_bos=True,
        )

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens, pad_idx=self.source_dictionary.pad()
                ),
                "nsrc_tokens": NumelDataset(src_tokens),
                "word_mask_w_bos": RightPadDataset(
                    word_masks_w_bos,
                    pad_idx=0,
                ),
            },
            "target_cats": RightPadDataset(
                term_cats, pad_idx=self.label_dictionary.pad()
            ),
            "target_attrs": RightPad2dDataset(term_attrs, pad_idx=0),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "nwords": NumWordsDataset(src_tokens, is_word_initial=self.is_word_initial),
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

        model.register_classification_head(
            "pos_ice",
            num_cats=self.num_cats,
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
