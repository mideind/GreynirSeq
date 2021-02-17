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


from greynirseq.nicenlp.byte_sequence import ByteSequence
from greynirseq.nicenlp.utils.data_utils import lengths_to_begin_mask


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
        from enum import IntEnum
        class Actions(IntEnum):
            NOOP = 0
            DELETE = 1
            MASK = 2
            INSERT = 3
            REPLACE = 4

        actions = np.random.choice(NUM_ACTIONS, size=num_bytes, p=self.probs)
        actions[-1] = Actions.NOOP  # do not disturb eos
        random_bytes = np.random.choice(256, size=num_bytes)
        new_byte_seq = []

        assert bpe_lens.sum() == len(seq.byte_seq)
        assert word_lens.sum() == len(seq.byte_seq)
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
            if action == Actions.NOOP:
                new_byte_seq.append(byte)
            elif action == Actions.DELETE:
                new_bpe_lens[bpe_list_index] -= 1
                new_word_lens[word_list_index] -= 1
            elif action == Actions.MASK:
                new_byte_seq.append(self.mask_idx)
            elif action == Actions.INSERT:
                new_byte_seq.append(random_byte)
                new_byte_seq.append(byte)
                new_bpe_lens[bpe_list_index] += 1
                new_word_lens[word_list_index] += 1
            elif action == Actions.REPLACE:
                new_byte_seq.append(random_byte)
            else:
                assert False, "unreachable"

        new_bpe_ids = seq.bpe_ids.clone()
        if min(new_bpe_lens) <= 0:
            new_bpe_ids, new_bpe_lens = zip(*list(item for item in zip(seq.bpe_ids, new_bpe_lens) if item[1] > 0))
            new_bpe_ids = torch.tensor(new_bpe_ids)

        assert seq.word_ids is None, "Not implemented"
        assert sum(new_bpe_lens) == len(new_byte_seq)
        assert len(new_bpe_lens) == len(new_bpe_ids)
        assert sum(new_bpe_lens) == sum(new_word_lens)
        assert 0 not in new_bpe_lens

        new_word_lens = torch.tensor(new_word_lens, dtype=torch.long)
        new_bpe_lens = torch.tensor(new_bpe_lens, dtype=torch.long)
        new_seq = ByteSequence(
            str_seq=seq.str_seq,
            byte_seq=torch.tensor(new_byte_seq, dtype=torch.long),
            word_lens=new_word_lens,
            bpe_lens=new_bpe_lens,
            bpe_mask=lengths_to_begin_mask(new_bpe_lens),
            word_mask=lengths_to_begin_mask(new_word_lens),
            bpe_ids=new_bpe_ids,
        )

        assert all(new_seq.bpe_lens.gt(0)), "Unexpected bpe token with length 0"
        return new_seq
