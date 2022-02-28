# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import logging
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Any, Callable, Optional
import json
from functools import lru_cache

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
from fairseq.dataclass import FairseqDataclass
from fairseq.data import (
    BaseWrapperDataset,
    Dictionary,
    LRUCacheDataset,
    NestedDictionaryDataset,
    data_utils,
    RightPadDataset,
)

from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch import Tensor

from omegaconf import MISSING, II

from greynirseq.nicenlp.data.datasets import (
    DynamicLabelledSpanDataset,
    NestedDictionaryDatasetFix,
    NumWordsDataset,
    WordEndMaskDataset,
    RightPad2dDataset,
)
from greynirseq.nicenlp.utils.constituency import token_utils
from greynirseq.nicenlp.utils.label_schema.label_schema import label_schema_as_dictionary, parse_label_schema
import greynirseq.nicenlp.utils.constituency.greynir_utils as greynir_utils

from greynirseq.nicenlp.utils.constituency.incremental_parsing import get_incremental_parse_actions, ParseAction, ROOT_LABEL

from icecream import ic


logger = logging.getLogger(__name__)


class TextEncodingDataset(BaseWrapperDataset):
    # temporary placement
    def __init__(self, dataset: Dataset, dictionary: Dictionary, bpe: Any, prepend_token: Optional[int] = None, add_prefix_space=True):
        super().__init__(dataset)
        self.dictionary = dictionary
        self.bpe = bpe
        self._sizes = None
        self.prepend_tensor = None
        if prepend_token is not None:
            self.prepend_tensor = torch.tensor([prepend_token])
        self.add_prefix_space = add_prefix_space

    def __getitem__(self, index: int):
        text = self.dataset[index]
        if self.add_prefix_space and not text[0] == " ":
            text = " " + text
        hf_ids_string = self.bpe.encode(text)
        output_ids = self.dictionary.encode_line(hf_ids_string)
        if self.prepend_tensor is not None:
            output_ids = torch.cat([self.prepend_tensor, output_ids])
        return output_ids

    def get_sizes(self):
        sizes = torch.zeros(len(self.dataset))
        for idx in range(len(self.dataset)):
            sizes[idx] = len(self[idx])
        return sizes

    @property
    def sizes(self):
        if self._sizes is not None:
            return self._sizes
        sizes = torch.zeros(len(self.dataset), dtype=torch.long)
        for idx in range(len(self.dataset)):
            sizes[idx] = len(self[idx])
        self._sizes = sizes
        return self._sizes


class _LambdaDataset(BaseWrapperDataset):
    # temporary placement
    def __init__(self, dataset: Dataset, lambda_fn):
        super().__init__(dataset)
        self.lambda_fn = lambda_fn

    def __getitem__(self, index: int):
        return self.lambda_fn(self.dataset[index])


class GreynirTreeJSONLDataset(BaseWrapperDataset):
    # temporary placement
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    @lru_cache(maxsize=64)
    def __getitem__(self, index: int):
        tree = greynir_utils.Node.from_json(self.dataset[index])
        tree.wrap_bare_terminals()
        return tree

    @classmethod
    def from_path(cls, path):
        with open(str(path), "r") as in_fh:
            line_list = in_fh.readlines()
        return cls(line_list)


class GreynirParsingDataset(BaseWrapperDataset):
    # temporary placement
    def __init__(
        self,
        greynirtree_dataset: Dataset,
        label_dictionary: Dictionary,
        source_dictionary: Dictionary,
        bpe: Any,
        prepend_token: Optional[int] = None,
    ):
        super().__init__(greynirtree_dataset)
        self.label_dictionary = label_dictionary
        self.source_dictionary = source_dictionary
        self.bpe = bpe
        self.padding_idx = label_dictionary.pad()
        self.prepend_tensor = None
        if prepend_token is not None:
            self.prepend_tensor = torch.tensor([prepend_token])
        self._sizes = None

    @lru_cache(maxsize=64)
    def __getitem__(self, index: int):
        tree = self.dataset[index]
        action_seq, preorder_list = get_incremental_parse_actions(tree, collapse=False)

        ret = {
            "inputs": GreynirParsingDataset._encode_inputs(
                preorder_list, action_seq, self.label_dictionary, self.padding_idx
            ),
            "targets": GreynirParsingDataset._encode_targets(action_seq, self.label_dictionary, self.padding_idx),
        }
        ret["inputs"]["_src_tokens"] = GreynirParsingDataset._encode_text(
            tree.text, self.source_dictionary, self.bpe, self.prepend_tensor
        )
        return ret

    @staticmethod
    def _encode_text(text: str, source_dictionary: Dictionary, bpe: Any, prepend_tensor: Optional[Tensor] = None):
        hf_ids_string = bpe.encode(text)
        output_ids = source_dictionary.encode_line(hf_ids_string)
        if prepend_tensor is not None:
            output_ids = torch.cat([prepend_tensor, output_ids])
        return output_ids

    @staticmethod
    def _encode_inputs(
        preorder_list: List[greynir_utils.Node],
        action_seq: List[ParseAction],
        label_dictionary: Dictionary,
        padding_idx: int,
    ):
        preorder_nts = torch.tensor(
            [label_dictionary.index(node.label_without_flags) for node in preorder_list], dtype=torch.long
        )
        preorder_spans = torch.tensor([node.span for node in preorder_list], dtype=torch.long)
        padded_flags = pad_sequence(
            [
                torch.tensor([label_dictionary.index(f) for f in node.label_flags], dtype=torch.long)
                if node.label_flags
                else torch.tensor([padding_idx], dtype=torch.long)
                for node in preorder_list
            ],
            batch_first=True,
            padding_value=padding_idx,
        )
        preorder_mask = torch.zeros(len(action_seq), len(preorder_list), dtype=torch.bool)
        chain_mask = torch.zeros(len(action_seq), len(preorder_list), dtype=torch.bool)
        for step, action in enumerate(action_seq):
            preorder_mask[step, action.preorder_indices] = 1
            chain_mask[step, action.right_chain_indices] = 1
        nwords_per_step = torch.tensor([node.nwords for node in action_seq], dtype=torch.long)
        return {
            "preorder_nts": preorder_nts,
            "preorder_spans": preorder_spans,
            "preorder_flags": padded_flags,
            "preorder_mask": preorder_mask,
            "chain_mask": chain_mask,
            "nwords_per_step": nwords_per_step,
        }

    @staticmethod
    def _encode_targets(action_seq: List[ParseAction], label_dictionary: Dictionary, padding_idx: int):
        depths = torch.tensor([node.depth for node in action_seq], dtype=torch.long)
        target_parents = torch.tensor(
            [label_dictionary.index(action.parent.label_without_flags) for action in action_seq], dtype=torch.long
        )
        target_preterms = torch.tensor(
            [label_dictionary.index(action.preterminal.label_without_flags) for action in action_seq], dtype=torch.long
        )
        target_parent_flags = pad_sequence(
            [
                torch.tensor([label_dictionary.index(f) for f in action.parent.label_flags], dtype=torch.long)
                if action.parent.label_flags
                else torch.tensor([padding_idx], dtype=torch.long)
                for action in action_seq
            ],
            batch_first=True,
            padding_value=padding_idx,
        )
        target_preterm_flags = pad_sequence(
            [
                torch.tensor([label_dictionary.index(f) for f in action.preterminal.label_flags], dtype=torch.long)
                if action.preterminal.label_flags
                else torch.tensor([padding_idx], dtype=torch.long)
                for action in action_seq
            ],
            batch_first=True,
            padding_value=padding_idx,
        )
        return {
            "target_depths": depths,
            "target_parents": target_parents,
            "target_padding_mask": torch.zeros_like(target_preterms, dtype=torch.bool),
            "target_preterminals": target_preterms,
            "target_parent_flags": target_parent_flags,
            "target_preterm_flags": target_preterm_flags,
        }

    @property
    def sizes(self):
        if self._sizes is not None:
            return self._sizes
        sizes = torch.zeros(len(self.dataset), dtype=torch.long)
        for idx in range(len(self.dataset)):
            sizes[idx] = len(self[idx]["inputs"]["preorder_nts"])
        self._sizes = sizes
        return self._sizes


@dataclass
class ParserHydraConfig(FairseqDataclass):
    data: Optional[Any] = field(default=None, metadata={"help": "Data directory, it should also contain dict.txt file"})
    nonterm_schema: Optional[str] = field(default=None, metadata={"help": "Hierarchical label-schema for nonterminals"})
    term_schema: Optional[str] = field(default=None, metadata={"help": "Hierarchical label-schema for terminals"})
    label_file: Optional[str] = field(default=None, metadata={"help": "Label dictionary file, analogous to fairseqs dict.txt"})
    # borrow from top level
    _bpe: Any = II("bpe")
    _dataset: Optional[Any] = II("dataset")
    _seed: int = II("common.seed")


@register_task("parser_hydra", dataclass=ParserHydraConfig)
class ParserHydraTask(FairseqTask):
    cfg: ParserHydraConfig

    def __init__(
        self,
        cfg: ParserHydraConfig,
        data_dictionary: Dictionary,
        label_dictionary: Dictionary,
        nterm_dict: Dictionary,
        nterm_schema,
        is_word_initial,
    ):
        super().__init__(cfg)
        self.dictionary = data_dictionary

        if not hasattr(self, "args") and hasattr(self, "cfg"):
            self.args = self.cfg

        self.nterm_dictionary = nterm_dict
        self.nterm_schema = nterm_schema
        self.label_dictionary = label_dictionary

        self.is_word_initial = is_word_initial

        self.num_nterm_cats = len(self.nterm_schema.label_categories)
        self.num_nterm_groups = NotImplemented
        self.num_nterm_labels = len(self.nterm_schema.labels)
        self.num_labels = len(self.label_dictionary)

    @classmethod
    def setup_task(cls, cfg: ParserHydraConfig, **kwargs):
        data_dict = cls.load_dictionary(cls, os.path.join(cfg.data, "dict.txt"))
        logger.info("[input] dictionary: {} types".format(len(data_dict)))
        label_dict = cls.load_dictionary(cls, cfg.label_file)
        logger.info("[label] dictionary: {} types".format(len(label_dict)))

        assert cfg._bpe._name == cfg._parent.bpe._name

        is_word_initial = cls.get_word_beginnings(cfg._parent.bpe, data_dict)

        nterm_dict, nterm_schema = cls.load_label_dictionary(cfg, cfg.nonterm_schema)
        logger.info("[nterm] dictionary: {} types".format(len(nterm_dict)))
        nterm_dict.null = nterm_dict.index(nterm_schema.null)
        nterm_dict.leaf_index = nterm_dict.index(nterm_schema.null_leaf)

        return ParserHydraTask(
            cfg,
            data_dict,
            label_dictionary=label_dict,
            nterm_dict=nterm_dict,
            nterm_schema=nterm_schema,
            is_word_initial=is_word_initial,
        )

    @classmethod
    def load_label_dictionary(cls, cfg: ParserHydraConfig, filename: str, **kwargs):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
        """
        assert Path(filename).exists(), f"Expected label_schema file at {filename}"
        label_schema = parse_label_schema(filename)

        return label_schema_as_dictionary(label_schema), label_schema

    @classmethod
    def load_dictionary(cls, cfg: ParserHydraConfig, filename: str, add_mask: bool = True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        if add_mask:
            dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def get_word_beginnings(cls, cfg: FairseqDataclass, dictionary: Dictionary):
        # Note that cfg here is some 'BPEConfig' and not ParserHydraConfig itself
        bpe = encoders.build_bpe(cfg)
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
        greynirtrees_dataset = GreynirTreeJSONLDataset.from_path(Path(self.cfg.data) / f"{split}.jsonl")

        bpe = encoders.build_bpe(self.cfg._bpe)
        greynirparsing_dataset = GreynirParsingDataset(
            greynirtrees_dataset,
            label_dictionary=self.label_dictionary,
            source_dictionary=self.source_dictionary,
            bpe=bpe,
            prepend_token=self.source_dictionary.bos(),
        )

        src_tokens = TextEncodingDataset(
            _LambdaDataset(greynirtrees_dataset, lambda_fn=lambda x: x.text),
            self.source_dictionary,
            bpe=bpe,
            prepend_token=self.source_dictionary.bos(),
        )
        word_masks_w_bos = WordEndMaskDataset(
            src_tokens, self.source_dictionary, self.is_word_initial, bos_value=1, eos_value=0
        )

        with data_utils.numpy_seed(self.cfg._seed):
            shuffle = np.random.permutation(len(src_tokens))

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(src_tokens, pad_idx=self.source_dictionary.pad()),
                "word_masks_w_bos": RightPadDataset(word_masks_w_bos, pad_idx=0),
                "preorder_nts": RightPadDataset(
                    _LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["inputs"]["preorder_nts"]),
                    pad_idx=self.label_dictionary.pad(),
                ),
                "preorder_mask": RightPad2dDataset(
                    _LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["inputs"]["preorder_mask"]), pad_idx=0
                ),
                "chain_mask": RightPad2dDataset(
                    _LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["inputs"]["chain_mask"]), pad_idx=0
                ),
                "preorder_spans": RightPad2dDataset(
                    _LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["inputs"]["preorder_spans"]), pad_idx=0
                ),
                "preorder_flags": RightPad2dDataset(
                    _LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["inputs"]["preorder_flags"]),
                    pad_idx=self.label_dictionary.pad(),
                ),
                "nwords_per_step": RightPadDataset(
                    _LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["inputs"]["nwords_per_step"]),
                    pad_idx=0,
                ),
            },
            "target_depths": RightPadDataset(
                _LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["targets"]["target_depths"]),
                pad_idx=self.label_dictionary.pad(),
            ),
            "target_padding_mask": RightPadDataset(
                _LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["targets"]["target_padding_mask"]),
                pad_idx=1,
            ),
            "target_parents": RightPadDataset(
                _LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["targets"]["target_parents"]),
                pad_idx=self.label_dictionary.pad(),
            ),
            "target_preterminals": RightPadDataset(
                _LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["targets"]["target_preterminals"]),
                pad_idx=self.label_dictionary.pad(),
            ),
            "target_parent_flags": RightPad2dDataset(
                _LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["targets"]["target_parent_flags"]),
                pad_idx=self.label_dictionary.pad(),
            ),
            "target_preterm_flags": RightPad2dDataset(
                _LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["targets"]["target_preterm_flags"]),
                pad_idx=self.label_dictionary.pad(),
            ),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "nwords": NumWordsDataset(src_tokens, self.dictionary, self.is_word_initial),
        }

        nested_dataset = NestedDictionaryDatasetFix(dataset, sizes=greynirparsing_dataset.sizes)

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

    @property
    def root_label_index(self):
        return self.label_dictionary.index(ROOT_LABEL)
