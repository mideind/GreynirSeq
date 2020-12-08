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
    ListDataset,
)
from fairseq.data import encoders
from fairseq.tasks import FairseqTask, register_task

from greynirseq.nicenlp.data.datasets import (
    POSDataset,
    WordEndMaskDataset,
    RightPad2dDataset,
    NestedDictionaryDatasetFix,
    NumWordsDataset,
)
from greynirseq.nicenlp.utils.label_schema.label_schema import label_schema_as_dictionary, parse_label_schema
from greynirseq.nicenlp.utils.constituency import token_utils

logger = logging.getLogger(__name__)


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
            self.dictionary,
            self.label_dictionary,
        )

        word_mask = WordEndMaskDataset(
            src_tokens,
            self.dictionary,
            self.is_word_initial,
            bos_value=0,
            eos_value=0,
        )

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens, pad_idx=self.source_dictionary.pad()
                ),
                "nsrc_tokens": NumelDataset(src_tokens),
                "word_mask": RightPadDataset(
                    word_mask,
                    pad_idx=0,
                ),
            },
            "target_cats": RightPadDataset(
                term_cats, pad_idx=self.label_dictionary.pad()
            ),
            "target_attrs": RightPad2dDataset(term_attrs, pad_idx=0),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "nwords": NumWordsDataset(
                src_tokens, self.dictionary, self.is_word_initial
            ),
        }

        nested_dataset = NestedDictionaryDatasetFix(dataset, sizes=[src_tokens.sizes])

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(nested_dataset, sort_order=[shuffle])

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def prepare_tokens(self, tokens):
        sizes = [len(seq) for seq in tokens]
        src_tokens = ListDataset(tokens, sizes=sizes)
        src_tokens = RightPadDataset(src_tokens, pad_idx=self.source_dictionary.pad())

        word_mask = WordEndMaskDataset(
            src_tokens, self.dictionary, self.is_word_initial, bos_value=0, eos_value=0
        )

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": src_tokens,
                "nsrc_tokens": NumelDataset(src_tokens),
                "word_mask": RightPadDataset(word_mask, pad_idx=0),
            },
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "nwords": NumWordsDataset(
                src_tokens, self.dictionary, self.is_word_initial
            ),
            "nsentences": NumSamplesDataset(),
        }
        dataset = NestedDictionaryDatasetFix(dataset, sizes=[src_tokens.sizes])
        return dataset

    def prepare_sentences(self, sentences):
        tokens = [
            self.encode(token_utils.tokenize_to_string(sentence))
            for sentence in sentences
        ]
        return self.task.prepare_tokens(tokens)

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def label_dictionary(self):
        return self._label_dictionary
