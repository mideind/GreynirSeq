from typing import List, Tuple, Optional, Callable
from functools import lru_cache

import numpy as np

import argparse
from torch.utils.data.dataset import Dataset
from fairseq.data import (
    BaseWrapperDataset,
    NestedDictionaryDataset,
    LRUCacheDataset,
    ListDataset,
    Dictionary,
)
from fairseq.data.nested_dictionary_dataset import _unflatten

import torch
from torch import LongTensor, BoolTensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from fairseq.data import data_utils


class ByteNoising(BaseWrapperDataset):
    """Typical character noise applied at byte level:
            - transposition
            - deletion
            - insertion
            - masking
            - replacement
    """

    def __init__(
        self,
        byte_sequence_dataset: Dataset,
        mask_prob: float = 0.02,
        delete_prob: float = 0.02,
        insert_prob: float = 0.02,
        replace_prob: float = 0.02,
        transpose_prob: float = 0.02,
        mask_idx: int = 256,
    ):
        super().__init__(byte_sequence_dataset)
        self.dataset = byte_sequence_dataset

        self.mask_prob = mask_prob
        self.delete_prob = delete_prob
        self.insert_prob = insert_prob
        self.replace_prob = replace_prob
        noise_probs = [
            self.delete_prob,
            self.mask_prob,
            self.insert_prob,
            self.replace_prob,
        ]
        assert all(
            0.0 <= r <= 1.0 for r in noise_probs
        ), "Probability must be between 0 and 1"
        assert 0.0 <= sum(noise_probs) <= 1.0
        self.noop_prob = 1 - sum(noise_probs)
        self.probs = np.array(
            [
                self.noop_prob,
                self.delete_prob,
                self.mask_prob,
                self.insert_prob,
                self.replace_prob,
            ]
        )
        assert 0 <= transpose_prob <= 0.5
        self.transpose_prob = transpose_prob
        self.transpose_weight = np.sqrt(self.transpose_prob / 2)
        self.mask_idx = mask_idx

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        seq = self.dataset[index]
        byte_seq = seq.byte_seq
        bpe_lens = seq.bpe_lens
        word_lens = seq.word_lens
        num_bytes = seq.byte_seq.numel()

        # shuffle within a distance of at most one with a
        vals = torch.tensor([-0.6, 0.0, 0.6])
        w = np.sqrt(self.transpose_prob / 2)
        weights = torch.tensor(
            [self.transpose_weight, 1 - self.transpose_prob, self.transpose_weight]
        )
        # TODO: make reproducible
        idxs = torch.multinomial(weights, num_bytes, replacement=True)
        dists = vals[idxs]
        perm = (torch.arange(num_bytes).float() + dists).sort().indices
        # TODO: should this respect word boundaries? probably not
        # NOTE: it's easy to make this respect word boundaries by changing sign of shifts on boundary if it points outside of word
        byte_seq = byte_seq[perm]

        NUM_ACTIONS = 5
        NOOP, DELETE, MASK, INSERT, REPLACE = range(NUM_ACTIONS)
        actions = np.random.choice(NUM_ACTIONS, size=num_bytes, p=self.probs)
        random_bytes = np.random.choice(256, size=num_bytes)
        new_byte_seq = []

        byte_index_to_bpe_list_index = torch.repeat_interleave(
            torch.arange(bpe_lens.numel()), bpe_lens
        )
        new_bpe_lens = [i for i in bpe_lens]
        byte_index_to_word_list_index = torch.repeat_interleave(
            torch.arange(word_lens.numel()), word_lens
        )
        new_word_lens = [i for i in word_lens]
        for byte_index, (byte, action, random_byte) in enumerate(
            zip(byte_seq, actions, random_bytes)
        ):
            bpe_list_index = byte_index_to_bpe_list_index[byte_index]
            word_list_index = byte_index_to_word_list_index[byte_index]
            if action == NOOP:
                new_byte_seq.append(byte)
            elif action == DELETE:
                new_bpe_lens[bpe_list_index] -= 1
                new_word_lens[word_list_index] -= 1
            elif action == MASK:
                new_byte_seq.append(self.mask_idx)
            elif action == INSERT:
                new_byte_seq.append(random_byte)
                new_byte_seq.append(byte)
                new_bpe_lens[bpe_list_index] += 1
                new_word_lens[word_list_index] += 1
            elif action == REPLACE:
                new_byte_seq.append(random_byte)
            else:
                assert False, "unreachable"

        new_seq = seq.clone()
        new_seq.byte_seq = torch.tensor(new_byte_seq, dtype=torch.long)
        new_seq.word_lens = torch.tensor(new_word_lens, dtype=torch.long)
        new_seq.bpe_lens = torch.tensor(new_bpe_lens, dtype=torch.long)
        # TODO: we might have a word id that should be removed (it has no bytes left)
        assert all(new_seq.bpe_lens.ge(0)), "Unexpected bpe token with length 0"
        return new_seq
