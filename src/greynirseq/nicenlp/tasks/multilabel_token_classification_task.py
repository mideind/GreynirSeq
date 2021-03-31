# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
from fairseq.data import (
    Dictionary,
    IdDataset,
    ListDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    data_utils,
)
from fairseq.tasks import FairseqTask, register_task

from greynirseq.nicenlp.data.datasets import (
    IgnoreLabelsDataset,
    NestedDictionaryDatasetFix,
    NumWordsDataset,
    POSDataset,
    RightPad2dDataset,
    WordEndMaskDataset,
)
from greynirseq.nicenlp.data.encoding import get_word_beginnings
from greynirseq.nicenlp.utils.label_schema.label_schema import (
    label_schema_as_dictionary,
    make_dict_idx_to_vec_idx,
    make_group_masks,
    make_group_name_to_group_attr_vec_idxs,
    make_vec_idx_to_dict_idx,
    parse_label_schema,
)

logger = logging.getLogger(__name__)


@register_task("multi_label_token_classification_task")
class MultiLabelTokenClassificationTask(FairseqTask):
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
        self,
        args: argparse.Namespace,
        data_dictionary: Dictionary,
        label_dictionary: Dictionary,
        is_word_initial: torch.Tensor,
        label_schema,
    ):
        super().__init__(args)
        torch.autograd.set_detect_anomaly(True)
        self.dictionary = data_dictionary
        self._label_dictionary = label_dictionary
        self._label_dictionary.sep = lambda: self._label_dictionary.index(label_schema.separator)
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
        self.ignore_cats = [self._label_dictionary.index(c) for c in self.label_schema.ignore_categories]

    @classmethod
    def setup_task(cls, args: argparse.Namespace, **kwargs):
        data_dict = cls.load_dictionary(args, os.path.join(args.data, "dict.txt"))
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        is_word_initial = get_word_beginnings(args, data_dict)
        term_dict = cls.load_dictionary(args, os.path.join(args.data, "dict_term.txt"), add_mask=False)

        # label_dict, label_schema = cls.load_label_dictionary(args, args.term_schema)
        _, label_schema = cls.load_label_dictionary(args, args.term_schema)
        logger.info("[label] dictionary: {} types".format(len(term_dict)))

        seen = set()
        for idx, lbl in enumerate(term_dict.symbols):
            exists = lbl in label_schema.labels
            seen.add(lbl)
            if (
                (not exists)
                and (idx > term_dict.nspecial)  # ignore bos, eos, etc
                and (lbl != "<mask>")
                and (lbl.startswith("madeupword"))  # ignore vocabulary padding
            ):
                assert False, "Unexpected POS label item in term_dict.txt: {}".format(lbl)
        for lbl in label_schema.labels:
            if lbl in seen:
                continue
            assert False, "Unexpected POS label item in label_schema {}".format(lbl)

        return MultiLabelTokenClassificationTask(args, data_dict, term_dict, is_word_initial, label_schema)

    @classmethod
    def load_label_dictionary(cls, args: argparse.Namespace, filename, **kwargs):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
        """
        label_schema = parse_label_schema(filename)

        return label_schema_as_dictionary(label_schema), label_schema

    @classmethod
    def load_dictionary(cls, args: argparse.Namespace, filename: str, add_mask: bool = True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        if add_mask:
            dictionary.add_symbol("<mask>")
        return dictionary

    def load_dataset(self, split: str, combine: bool = False, **kwargs):
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

        term_cats, term_attrs = POSDataset.make_both(term_labels, self.dictionary, self.label_dictionary)

        def print_terms(term_cats, term_attrs):
            # Debug function
            cat_labels = [self.label_dictionary[t] for t in term_cats]
            attr_data = [t.nonzero().T for t in term_attrs if t.numel()]
            attr_labels = []
            for word_attr in attr_data:
                if not word_attr.numel():
                    attr_labels.append([])
                    continue
                attr_labels.append([self.label_dictionary[t + self.label_dictionary.nspecial] for t in word_attr[0]])
            return cat_labels, attr_labels

        word_mask = WordEndMaskDataset(src_tokens, self.dictionary, self.is_word_initial, bos_value=0, eos_value=0)

        exclude_cats_mask = IgnoreLabelsDataset(term_cats, self.ignore_cats)

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(src_tokens, pad_idx=self.source_dictionary.pad()),
                "nsrc_tokens": NumelDataset(src_tokens),
                "word_mask": RightPadDataset(word_mask, pad_idx=0),
            },
            "exclude_cats_mask": RightPadDataset(exclude_cats_mask, pad_idx=1),
            "target_cats": RightPadDataset(term_cats, pad_idx=self.label_dictionary.pad()),
            "target_attrs": RightPad2dDataset(term_attrs, pad_idx=0),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "nwords": NumWordsDataset(src_tokens, self.dictionary, self.is_word_initial),
        }

        nested_dataset = NestedDictionaryDatasetFix(dataset, sizes=[src_tokens.sizes])

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(nested_dataset, sort_order=[shuffle])

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))
        self.datasets[split] = dataset
        return self.datasets[split]

    def prepare_tokens(self, tokens: torch.Tensor):
        sizes = [len(seq) for seq in tokens]
        src_tokens = ListDataset(tokens, sizes=sizes)
        src_tokens = RightPadDataset(src_tokens, pad_idx=self.source_dictionary.pad())

        word_mask = WordEndMaskDataset(src_tokens, self.dictionary, self.is_word_initial, bos_value=0, eos_value=0)

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": src_tokens,
                "nsrc_tokens": NumelDataset(src_tokens),
                "word_mask": RightPadDataset(word_mask, pad_idx=0),
            },
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "nwords": NumWordsDataset(src_tokens, self.dictionary, self.is_word_initial),
            "nsentences": NumSamplesDataset(),
        }
        dataset = NestedDictionaryDatasetFix(dataset, sizes=[src_tokens.sizes])
        return dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary

    @property
    def group_name_to_group_attr_vec_idxs(self):
        return make_group_name_to_group_attr_vec_idxs(self.label_dictionary, self.label_schema)

    @property
    def cat_dict_idx_to_vec_idx(self):
        return make_dict_idx_to_vec_idx(self.label_dictionary, self.label_schema.label_categories)

    @property
    def cat_vec_idx_to_dict_idx(self):
        return make_vec_idx_to_dict_idx(self.label_dictionary, self.label_schema.label_categories)

    @property
    def group_mask(self):
        return make_group_masks(self.label_dictionary, self.label_schema)

    def logits_to_labels(
        self,
        cat_logits: torch.Tensor,
        attr_logits: torch.Tensor,
        word_mask: torch.Tensor,
    ):
        # logits: Batch x Time x Labels
        bsz, _, num_cats = cat_logits.shape
        _, _, num_attrs = attr_logits.shape
        nwords = word_mask.sum(-1)
        assert num_attrs == len(self.label_schema.labels)
        assert num_cats == len(self.label_schema.label_categories)

        batch_cats = []
        batch_attrs = []
        for seq_idx in range(bsz):
            seq_nwords = nwords[seq_idx]
            pred_cat_vec_idxs = cat_logits[seq_idx, :seq_nwords].max(dim=-1).indices
            pred_cats = self.cat_vec_idx_to_dict_idx[pred_cat_vec_idxs]

            group_mask = self.group_mask[pred_cat_vec_idxs]
            offset = self.label_dictionary.nspecial
            pred_attrs = []
            for group_idx, group_name in enumerate(self.label_schema.group_names):
                group_vec_idxs = self.group_name_to_group_attr_vec_idxs[group_name]
                # logits: (bsz * nwords) x labels
                group_logits = attr_logits[seq_idx, :seq_nwords, group_vec_idxs]
                if len(group_vec_idxs) == 1:
                    group_pred = group_logits.sigmoid().ge(0.5).long()
                    group_pred_dict_idxs = (group_pred.squeeze() * (group_vec_idxs.item() + offset)).T.to(
                        "cpu"
                    ) * group_mask[:, group_idx]
                else:
                    group_pred_vec_idxs = group_logits.max(dim=-1).indices
                    group_pred_dict_idxs = (group_vec_idxs[group_pred_vec_idxs] + offset) * group_mask[:, group_idx]
                pred_attrs.append(group_pred_dict_idxs)

            pred_attrs = torch.stack([p.squeeze() for p in pred_attrs]).t()

            batch_cats.append(pred_cats)
            batch_attrs.append(pred_attrs)

        predictions = list(
            [
                _clean_cats_attrs(
                    self.label_dictionary,
                    self.label_schema,
                    seq_cats,
                    seq_attrs,
                )
                for seq_cats, seq_attrs in zip(batch_cats, batch_attrs)
            ]
        )

        return predictions


def _clean_cats_attrs(ldict: Dictionary, schema, pred_cats: torch.Tensor, pred_attrs: torch.Tensor):
    cats = ldict.string(pred_cats).split(" ")
    attrs = []

    if len(pred_attrs.shape) == 1:
        split_pred_attrs = [pred_attrs]
    else:
        split_pred_attrs = pred_attrs.split(1, dim=0)
    for (_cat_idx, attr_idxs) in zip(pred_cats.tolist(), split_pred_attrs):
        seq_attrs = [lbl for lbl in ldict.string((attr_idxs.squeeze())).split(" ")]
        if not any(it for it in seq_attrs):
            seq_attrs = []
        attrs.append(seq_attrs)
    return list(zip(cats, attrs))
