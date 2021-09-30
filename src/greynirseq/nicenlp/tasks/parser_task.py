# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.


import argparse
import logging
import os
from pathlib import Path
from typing import List

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
    encoders,
)
from fairseq.tasks import FairseqTask, register_task

from greynirseq.nicenlp.data.datasets import (
    DynamicLabelledSpanDataset,
    NestedDictionaryDatasetFix,
    NumWordsDataset,
    WordEndMaskDataset,
)
from greynirseq.nicenlp.utils.constituency import token_utils
from greynirseq.nicenlp.utils.label_schema.label_schema import label_schema_as_dictionary, parse_label_schema

logger = logging.getLogger(__name__)


@register_task("parser")
class ParserTask(FairseqTask):
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
        parser.add_argument(
            "--nonterm-schema",
            metavar="FILE",
            help="json file providing label-set and label-groups \
                (mutually exclusive labels)",
            required=True,
        )
        parser.add_argument("--no-shuffle", action="store_true", default=False)

    def __init__(
        self,
        args: argparse.Namespace,
        data_dictionary: Dictionary,
        nterm_dict: Dictionary,
        nterm_schema,
        is_word_initial,
    ):
        super().__init__(args)
        self.dictionary = data_dictionary

        if not hasattr(self, "args") and hasattr(self, "cfg"):
            self.args = self.cfg

        self.nterm_dictionary = nterm_dict
        self.nterm_schema = nterm_schema

        if not hasattr(args, "max_positions"):
            self._max_positions = (args.max_source_positions, args.max_target_positions)
        else:
            self._max_positions = args.max_positions
        args.tokens_per_sample = self._max_positions
        self.is_word_initial = is_word_initial

        self.num_nterm_cats = len(self.nterm_schema.label_categories)
        self.num_nterm_groups = NotImplemented
        self.num_nterm_labels = len(self.nterm_schema.labels)

    @classmethod
    def setup_task(cls, args: argparse.Namespace, **kwargs):
        data_dict = cls.load_dictionary(args, os.path.join(args.data, "dict.txt"))
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        is_word_initial = cls.get_word_beginnings(args, data_dict)

        # assert labels[0] == "NULL", "Expected label at index 0 to be 'NULL'"
        nterm_dict, nterm_schema = cls.load_label_dictionary(args, args.nonterm_schema)
        logger.info("[nterm] dictionary: {} types".format(len(nterm_dict)))
        nterm_dict.null = nterm_dict.index(nterm_schema.null)
        nterm_dict.leaf_index = nterm_dict.index(nterm_schema.null_leaf)

        return ParserTask(
            args,
            data_dict,
            nterm_dict=nterm_dict,
            nterm_schema=nterm_schema,
            is_word_initial=is_word_initial,
        )

    @classmethod
    def load_label_dictionary(cls, args: argparse.Namespace, filename: str, **kwargs):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
        """
        assert Path(filename).exists(), f"Expected label_schema file at {filename}"
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

    @classmethod
    def get_word_beginnings(cls, args: argparse.Namespace, dictionary: Dictionary):
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

    def load_dataset(self, split: str, combine: bool = False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        inputs_path = Path(self.args.data) / f"{split}.text"
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
        word_masks_w_bos = WordEndMaskDataset(
            src_tokens, self.dictionary, self.is_word_initial, bos_value=1, eos_value=0
        )

        nterm_targets_path = Path(self.args.data) / "{}.nonterm".format(split)
        labelled_spans = data_utils.load_indexed_dataset(
            str(nterm_targets_path),
            self.nterm_dictionary,
            self.args.dataset_impl,
            combine=combine,
        )
        assert labelled_spans is not None, "could not find nonterminal labels: {}".format(nterm_targets_path)
        target_spans, nterm_cats = DynamicLabelledSpanDataset.make_both(
            labelled_spans,
            self.nterm_dictionary,
            seed=self.args.seed,
        )

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(src_tokens, pad_idx=self.source_dictionary.pad()),
                "nsrc_tokens": NumelDataset(src_tokens),
                "word_mask_w_bos": RightPadDataset(word_masks_w_bos, pad_idx=0),
            },
            "target_span_labels": RightPadDataset(nterm_cats, pad_idx=self.nterm_dictionary.pad()),
            "target_spans": RightPadDataset(target_spans, pad_idx=0),
            "ntarget_span_labels": NumelDataset(nterm_cats),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "nwords": NumWordsDataset(src_tokens, self.dictionary, self.is_word_initial),
        }

        nested_dataset = NestedDictionaryDatasetFix(dataset, sizes=[src_tokens.sizes])

        dataset = SortDataset(nested_dataset, sort_order=[shuffle])

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def prepare_sentences(self, sentences: List[str]):
        tokens = [self.encode(token_utils.tokenize_to_string(sentence)) for sentence in sentences]
        return self.task.prepare_tokens(tokens)

    def prepare_tokens(self, tokens: torch.Tensor):
        sizes = [len(seq) for seq in tokens]
        src_tokens = ListDataset(tokens, sizes=sizes)
        src_tokens = RightPadDataset(src_tokens, pad_idx=self.source_dictionary.pad())

        word_masks_w_bos = WordEndMaskDataset(
            src_tokens, self.dictionary, self.is_word_initial, bos_value=1, eos_value=0
        )

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": src_tokens,
                "nsrc_tokens": NumelDataset(src_tokens),
                "word_mask_w_bos": RightPadDataset(word_masks_w_bos, pad_idx=0),
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
