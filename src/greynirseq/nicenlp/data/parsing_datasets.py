# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

from functools import lru_cache
from typing import Any, List, Optional

import numpy as np
import torch
from fairseq.data import BaseWrapperDataset, Dictionary, data_utils
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

import greynirseq.nicenlp.utils.constituency.greynir_utils as greynir_utils
from greynirseq.nicenlp.utils.constituency.incremental_parsing import (
    EOS_LABEL,
    ParseAction,
    get_depths,
    get_incremental_parse_actions,
    get_preorder_index_of_right_chain,
)


def _remove_nt(node: greynir_utils.Node, nonterm: str, prob: float):
    def should_remove(node, nonterm):
        if node.terminal:
            return False
        if node.nonterminal == nonterm and np.random.rand() < (prob):
            return True
        new_children = []
        for child in node.children:
            if should_remove(child, nonterm):
                continue
            new_children.append(child)
        if not new_children:
            return True
        node._children = new_children
        return False

    should_remove(node, nonterm)


class GreynirTreeJSONLDataset(BaseWrapperDataset):
    def __init__(self, line_list):
        super().__init__(line_list)
        self._sizes = None
        # self.trees = {}

    def __getitem__(self, index: int):
        # if index in self.trees:
        #     return self.trees[index]
        tree = greynir_utils.Node.from_json(self.dataset[index])
        tree = tree.split_multiword_tokens()
        tree.wrap_bare_terminals()
        # self.trees[index] = tree
        # return self.trees[index]
        return tree

    @classmethod
    def from_path(cls, path, limit=None):
        with open(str(path), "r") as in_fh:
            line_list = in_fh.readlines()
        if limit is not None and limit > 0:
            line_list = line_list[:limit]
        return cls(line_list)


    @property
    def sizes(self):
        if self._sizes is not None:
            return self.sizes
        self._sizes = [len(item) for item in self.dataset]
        return self._sizes


class GreynirTreeAugmentationDataset(BaseWrapperDataset):
    def __init__(self, greynirtree_dataset: Dataset, noise_prob, seed: int=1):
        super().__init__(greynirtree_dataset)
        self.noise_prob = noise_prob
        self.epoch = 1
        self.seed = seed

    def __getitem__(self, index: int):
        tree = self.dataset[index].clone()

        with data_utils.numpy_seed(self.seed, self.epoch, index):
            # randomly remove punctuation
            _remove_nt(tree, "POS|GRM", self.noise_prob)

            # randomly lowercase all
            if np.random.rand() < self.noise_prob:
                for leaf in tree.leaves:
                    leaf._text = leaf.text.lower()
        return tree

    def set_epoch(self, epoch):
        self.epoch = epoch


class GreynirParsingDataset(BaseWrapperDataset):
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
        self.epoch = 1

    @classmethod
    def decompose(cls, dataset):
        from greynirseq.nicenlp.data.lambda_dataset import LambdaDataset
        return {
            "src_tokens": LambdaDataset(dataset, lambda_fn=lambda x: x["src_tokens"]),
            "preorder_nts": LambdaDataset(dataset, lambda_fn=lambda x: x["preorder_nts"]),
            "preorder_depths": LambdaDataset(dataset, lambda_fn=lambda x: x["preorder_depths"]),
            "preorder_mask": LambdaDataset(dataset, lambda_fn=lambda x: x["preorder_mask"]),
            "chain_mask": LambdaDataset(dataset, lambda_fn=lambda x: x["chain_mask"]),
            "preorder_spans": LambdaDataset(dataset, lambda_fn=lambda x: x["preorder_spans"]),
            "preorder_flags": LambdaDataset(dataset, lambda_fn=lambda x: x["preorder_flags"]),
            "nwords_per_step": LambdaDataset(dataset, lambda_fn=lambda x: x["nwords_per_step"]),
            "target_depths": LambdaDataset(dataset, lambda_fn=lambda x: x["target_depths"]),
            "target_padding_mask": LambdaDataset(dataset, lambda_fn=lambda x: x["target_padding_mask"]),
            "target_parents": LambdaDataset(dataset, lambda_fn=lambda x: x["target_parents"]),
            "target_preterms": LambdaDataset(dataset, lambda_fn=lambda x: x["target_preterms"]),
            "target_parent_flags": LambdaDataset(dataset, lambda_fn=lambda x: x["target_parent_flags"]),
            "target_preterm_flags": LambdaDataset(dataset, lambda_fn=lambda x: x["target_preterm_flags"]),
        }

    @lru_cache(maxsize=64)
    def __getitem__(self, index: int):
        tree = self.dataset[index]
        action_seq, preorder_list = get_incremental_parse_actions(tree, collapse=False, eos=EOS_LABEL)

        ret = {}
        ret.update(GreynirParsingDataset._encode_inputs(preorder_list, action_seq, self.label_dictionary, self.padding_idx))
        ret.update(GreynirParsingDataset._encode_targets(action_seq, self.label_dictionary, self.padding_idx))

        ret["src_tokens"] = GreynirParsingDataset._encode_text(
            tree.text, self.source_dictionary, self.bpe, self.prepend_tensor
        )
        del tree
        del action_seq
        del preorder_list

        return ret

    @staticmethod
    def _encode_text(text: str, source_dictionary: Dictionary, bpe: Any, prepend_tensor: Optional[Tensor] = None):
        hf_ids_string = bpe.encode(text if text.startswith(" ") else " " + text)
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
        chain_mask = torch.zeros_like(preorder_mask)
        preorder_depths = torch.zeros(len(action_seq), len(preorder_list), dtype=torch.long)
        for step, action in enumerate(action_seq):
            preorder_mask[step, action.preorder_indices] = 1
            chain_mask[step, action.right_chain_indices] = 1
            preorder_depths[step, action.preorder_indices] = torch.tensor(action.preorder_depths, dtype=torch.long)
        nwords_per_step = torch.tensor([node.nwords for node in action_seq], dtype=torch.long)
        return {
            "preorder_nts": preorder_nts,
            "preorder_spans": preorder_spans,
            "preorder_depths": preorder_depths,
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
            "target_preterms": target_preterms,
            "target_parent_flags": target_parent_flags,
            "target_preterm_flags": target_preterm_flags,
        }

    def prepare_tree_for_inference(self, tree: Any, collated=False):
        preorder_list = tree.preorder_list()
        preorder_nts = torch.tensor(
            [self.label_dictionary.index(node.label_without_flags) for node in preorder_list], dtype=torch.long
        )
        preorder_spans = torch.tensor([node.span for node in preorder_list], dtype=torch.long)
        padded_flags = pad_sequence(
            [
                torch.tensor([self.label_dictionary.index(f) for f in node.label_flags], dtype=torch.long)
                if node.label_flags
                else torch.tensor([self.padding_idx], dtype=torch.long)
                for node in preorder_list
            ],
            batch_first=True,
            padding_value=self.padding_idx,
        )
        preorder_mask = torch.ones(len(preorder_list), dtype=torch.bool)
        chain_mask = torch.zeros(len(preorder_list), dtype=torch.bool)
        right_chain_indices = get_preorder_index_of_right_chain(tree, include_terminals=False, preserve_indices=True)
        preorder_depths = torch.tensor(get_depths(tree), dtype=torch.long)
        chain_mask[right_chain_indices] = 1
        nwords = torch.tensor(len(tree.leaves))
        ret = {
            "preorder_nts": preorder_nts,
            "preorder_spans": preorder_spans,
            "preorder_flags": padded_flags,
            "preorder_mask": preorder_mask,
            "preorder_depths": preorder_depths,
            "chain_mask": chain_mask,
            "nwords": nwords,
        }
        if collated:
            for key in ret:
                ret[key] = ret[key].unsqueeze(0)
        return ret

    @property
    def sizes(self):
        if self._sizes is not None:
            return self._sizes
        sizes = torch.zeros(len(self.dataset), dtype=torch.long)
        for idx in range(len(self.dataset)):
            sizes[idx] = self[idx]["nwords_per_step"][-1]
        self._sizes = sizes
        return self._sizes

    def set_epoch(self, epoch):
        self.epoch = epoch
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)


