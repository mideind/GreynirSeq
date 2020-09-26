import logging
import os

import numpy as np

from fairseq.data import (
    data_utils,
    Dictionary,
    IdDataset,
    NumSamplesDataset,
    NumelDataset,
    RightPadDataset,
    SortDataset,
    TruncateDataset,
)
from fairseq.data import encoders
from fairseq.tasks import register_task
from fairseq.tasks.sentence_prediction import SentencePredictionTask

from greynirseq.nicenlp.data import (
    LookupDataset,
    NestedDictionaryDatasetFix,
    MutexBinaryDataset,
)

logger = logging.getLogger(__name__)


@register_task("multi_label_word_classification")
class WordPredictionTask(SentencePredictionTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument("data", metavar="FILE", help="file prefix for data")
        parser.add_argument(
            "--num-classes-mutex",
            type=int,
            default=-1,
            help="Number of mutually exclusive classes, only one such",
        )
        parser.add_argument(
            "--num-classes-binary",
            type=int,
            default=-1,
            help="Number of binary classes, non mutuall exclusive",
        )
        parser.add_argument(
            "--separator-token",
            type=int,
            default="<SEP>",
            help="add separator token between inputs",
        )
        parser.add_argument("--no-shuffle", action="store_true", default=False)
        parser.add_argument(
            "--truncate-sequence",
            action="store_true",
            default=False,
            help="truncate sequence to max-positions",
        )

    def __init__(self, args, data_dictionary, label_dictionary, token_begins_word):
        super().__init__(args, data_dictionary, label_dictionary)
        self.is_word_begin = token_begins_word
        self.span_separator = -1
        self.bpe = encoders.build_bpe(args)

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, "input0", "dict.txt"),
            source=True,
        )
        logger.info("[input] dictionary: {} types".format(len(data_dict)))

        label_dict = cls.load_label_dictionary(
            args,
            os.path.join(args.data, "labels0", "dict.txt"),
            source=False,
        )
        logger.info("[label] dictionary: {} types".format(len(label_dict)))
        is_word_begin = cls.get_word_beginnings(args, data_dict)

        return WordPredictionTask(args, data_dict, label_dict, is_word_begin)

    @classmethod
    def load_label_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        return dictionary

    @classmethod
    def get_word_beginnings(cls, args, dictionary):
        bpe = encoders.build_bpe(args)
        if bpe is not None:

            # Punctuation is BOW ?
            bow_ids = bpe.encode(".,:;").split()

            def is_beginning_of_word(i):
                if i < dictionary.nspecial:
                    return True
                tok = dictionary[i]
                # if tok in bow_ids:
                #    return True
                if tok.startswith("madeupword"):
                    return True
                try:
                    return bpe.is_beginning_of_word(tok)
                except ValueError:
                    return True

            is_word_begin = {}
            for i in range(len(dictionary)):
                is_word_begin[i] = int(is_beginning_of_word(i))
            return is_word_begin
        return None

    def debug_dataset(self, dataset, idx):
        ni = dataset["net_input"]
        src_tokens = ni["src_tokens"]
        src_spans = ni["src_spans"]
        target = dataset["target"][idx]
        debug_data = [
            self.bpe.decode(self.dictionary.symbols[v.item()])
            for v in src_tokens[idx][:-1]
        ]  # Need to unpad
        debug_data_tar = [
            self.label_dictionary.symbols[v.item()] for v in target if v > -1
        ]
        print("".join(debug_data))
        print(debug_data)
        print(src_tokens[idx], src_tokens[idx].shape)
        print(src_spans[idx], src_tokens[idx].shape)
        print(debug_data_tar)
        print(
            [
                (x, y.tolist(), z.item())
                for x, y, z in zip(debug_data, src_tokens[idx], src_spans[idx])
            ]
        )
        print(target, target.shape)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_dataset(type, dictionary):
            split_path = get_path(type, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                dictionary,
                combine=combine,
            )
            return dataset

        src_tokens = make_dataset("input0", self.source_dictionary)
        assert src_tokens is not None, "could not find dataset: {}".format(
            get_path(type, split)
        )

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        if self.args.truncate_sequence:
            src_tokens = TruncateDataset(src_tokens, self.args.max_positions)

        labels = make_dataset("labels0", self.label_dictionary)
        assert labels is not None, "could not find labels: {}".format(
            get_path(type, split)
        )

        mutex_binary_labels = MutexBinaryDataset(
            labels,
            num_mutex_classes=self.args.num_classes_mutex,
            skip_n=self.label_dictionary.nspecial,  # + 1,  # Add one to count label ?
            separator=self.span_separator,
        )

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                "src_spans": RightPadDataset(
                    LookupDataset(src_tokens, self.is_word_begin),
                    pad_idx=self.label_dictionary.pad(),
                ),
                "nspans": NumelDataset(mutex_binary_labels),
            },
            "target": RightPadDataset(
                mutex_binary_labels,
                pad_idx=self.label_dictionary.pad(),
            ),
            "ntargets": NumelDataset(labels, reduce=True),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
        }

        # self.debug_dataset(dataset, 5)
        # import pdb; pdb.set_trace()

        nested_dataset = NestedDictionaryDatasetFix(
            dataset,
            sizes=[src_tokens.sizes],
        )

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                sort_order=[shuffle],
            )

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models

        model = models.build_model(args, self)

        num_classes_mutex = len(self._label_dictionary)
        num_classes_binary = 0

        if hasattr(self.args, "num_classes_mutex"):
            num_classes_mutex = self.args.num_classes_mutex

        if hasattr(self.args, "num_classes_binary"):
            num_classes_binary = self.args.num_classes_binary

        model.register_classification_head(
            "multi_label_word_classification",
            num_classes_mutex=num_classes_mutex,
            num_classes_binary=num_classes_binary,
        )

        return model
