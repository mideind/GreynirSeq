from functools import lru_cache
from typing import Any, List, Optional

import torch
from fairseq.data import BaseWrapperDataset, Dictionary
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
        if node.nonterminal == nonterm and torch.rand(1).squeeze() < (prob):
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
    @lru_cache(maxsize=64)
    def __getitem__(self, index: int):
        tree = greynir_utils.Node.from_json(self.dataset[index])
        tree = tree.split_multiword_tokens()
        tree.wrap_bare_terminals()
        return tree

    @classmethod
    def from_path(cls, path):
        with open(str(path), "r") as in_fh:
            line_list = in_fh.readlines()
        return cls(line_list)


class GreynirTreeAugmentationDataset(BaseWrapperDataset):
    def __init__(self, greynirtree_dataset: Dataset, noise_prob):
        super().__init__(greynirtree_dataset)
        self.noise_prob = noise_prob

    @lru_cache(maxsize=64)
    def __getitem__(self, index: int):
        tree = self.dataset[index].clone()

        # randomly remove punctuation
        _remove_nt(tree, "POS|GRM", self.noise_prob)

        # randomly lowercase all
        if torch.rand(1).squeeze() < self.noise_prob:
            for leaf in tree.leaves:
                leaf._text = leaf.text.lower()

        return tree


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

    @lru_cache(maxsize=64)
    def __getitem__(self, index: int):
        tree = self.dataset[index]
        action_seq, preorder_list = get_incremental_parse_actions(tree, collapse=False, eos=EOS_LABEL)

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
            sizes[idx] = len(self[idx]["inputs"]["preorder_nts"])
        self._sizes = sizes
        return self._sizes
