from typing import List, Tuple, Optional, Callable
from functools import lru_cache
from enum import IntEnum

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


class Actions(IntEnum):
    NOOP = 0
    MASK = 2
    REPLACE = 4
    DELETE = 1
    INSERT = 3


class ByteNoising(BaseWrapperDataset):
    """Typical character noise applied at byte level:
            - transposition
            - deletion
            - insertion
            - masking
            - replacement
    """

    _CACHE_SIZE = 32

    def __init__(
        self,
        byte_sequence_dataset: Dataset,
        mask_prob: float = 0.02,
        delete_prob: float = 0.02,
        insert_prob: float = 0.02,
        replace_prob: float = 0.02,
        transpose_prob: float = 0.02,
        mask_idx: int = 256,
        seed: int = 0,
    ):
        super().__init__(byte_sequence_dataset)
        self.dataset = byte_sequence_dataset

        self.mask_prob = mask_prob
        self.delete_prob = delete_prob
        self.insert_prob = insert_prob
        self.replace_prob = replace_prob
        noise_probs = [
            self.mask_prob,
            self.replace_prob,
            self.delete_prob,
            self.insert_prob,
        ]
        assert all(
            0.0 <= r <= 1.0 for r in noise_probs
        ), "Probability must be between 0 and 1"
        assert 0.0 <= sum(noise_probs) <= 1.0
        self.noop_prob = 1 - sum(noise_probs)
        self.probs = np.array(
            [
                self.noop_prob,
                self.mask_prob,
                self.replace_prob,
                self.delete_prob,
                self.insert_prob,
            ]
        )
        assert 0 <= transpose_prob <= 0.5
        self.transpose_prob = transpose_prob
        self.transpose_weight = np.sqrt(self.transpose_prob / 2)
        self.mask_idx = mask_idx
        self.seed = seed
        self.epoch = 0

    @lru_cache(maxsize=_CACHE_SIZE)
    def __getitem__(self, index):
        seq = self.dataset[index].clone()
        byte_seq = seq.byte_seq
        bpe_lens = seq.bpe_lens
        word_lens = seq.word_lens
        num_bytes = seq.byte_seq.numel()

        with data_utils.numpy_seed(self.seed, self.epoch, index):
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
            new_word_lens = torch.tensor([i for i in word_lens], dtype=torch.long)

            # XXX: The block-placement algorithm wants sequence lengths upfront
            #      in order to decide block boundaries.
            #      Since delete-rate and insertion-rate are approximately equal
            #      (and also small) we can enforce an upper-bound of length-increase.
            #      Since we mostly care about exceeding block budget or if a given sequence is
            #      a document boundary
            length = len(byte_seq)
            num_deletions = int(self.delete_prob * length + np.random.rand())  # probabalistic rounding
            num_insertions = int(self.insert_prob * length + np.random.rand())  # probabalistic rounding

            replace_mask = torch.rand(num_bytes) < self.replace_prob
            # # replace_values = np.random.choice(256, replace_mask.sum().item())
            replace_values = torch.randint(256, size=(replace_mask.sum().item(),))
            seq_replaced = byte_seq.clone()
            seq_replaced[replace_mask] = replace_values

            action_idxs = torch.from_numpy(np.random.choice(length, num_deletions + num_insertions, replace=False))
            insert_idxs = action_idxs[:num_deletions]
            delete_idxs = action_idxs[num_deletions:num_deletions + num_insertions]
            actions = torch.zeros_like(byte_seq)
            actions[insert_idxs] = Actions.INSERT
            actions[delete_idxs] = Actions.DELETE
            random_bytes = torch.randint(256, size=(num_bytes,))

            for byte_index, (byte, action, random_byte) in enumerate(
                zip(seq_replaced, actions, random_bytes)
            ):
                bpe_list_index = byte_index_to_bpe_list_index[byte_index]
                word_list_index = byte_index_to_word_list_index[byte_index]

                if action == Actions.NOOP:
                    new_byte_seq.append(byte)
                elif action == Actions.INSERT:
                    new_byte_seq.append(random_byte)
                    new_byte_seq.append(byte)
                    new_bpe_lens[bpe_list_index] += 1
                    new_word_lens[word_list_index] += 1
                elif action == Actions.DELETE:
                    new_bpe_lens[bpe_list_index] -= 1
                    new_word_lens[word_list_index] -= 1
                else:
                    assert False, "unreachable"

            mask_mask = torch.rand(len(new_byte_seq)) < self.mask_prob
            new_byte_seq = torch.tensor(new_byte_seq, dtype=torch.long)
            new_byte_seq[mask_mask] = self.mask_idx

        new_bpe_ids = seq.bpe_ids.clone()
        if min(new_bpe_lens) <= 0:
            new_bpe_ids, new_bpe_lens = zip(*list(item for item in zip(seq.bpe_ids, new_bpe_lens) if item[1] > 0))
            new_bpe_ids = torch.tensor(new_bpe_ids)
        new_word_lens = new_word_lens[new_word_lens.gt(0)]

        assert seq.word_ids is None, "Not implemented"
        assert sum(new_bpe_lens) == len(new_byte_seq)
        assert len(new_bpe_lens) == len(new_bpe_ids)
        assert sum(new_bpe_lens) == sum(new_word_lens)
        assert 0 not in new_bpe_lens

        new_bpe_lens = torch.tensor(new_bpe_lens, dtype=torch.long)
        new_seq = ByteSequence(
            str_seq=seq.str_seq,
            byte_seq=new_byte_seq,
            word_lens=new_word_lens,
            bpe_lens=new_bpe_lens,
            bpe_mask=lengths_to_begin_mask(new_bpe_lens),
            word_mask=lengths_to_begin_mask(new_word_lens),
            bpe_ids=new_bpe_ids,
        )

        assert all(new_seq.bpe_lens.gt(0)), "Unexpected bpe token with length 0"
        assert torch.all(new_seq.bpe_lens.gt(0)), "Unexpected bpe token with length 0"
        assert torch.all(new_seq.word_lens.gt(0)), "Unexpected word with length 0"
        return new_seq

    @property
    def sizes(self):
        # This is used to prefilter sentences that exceed max-length and also to
        # assemble multiple sequences (sentences) into blocks. This is also used
        # to determine document boundaries (when seq size is 1)
        #
        # Since we use dynamic word segmentation, this size is not known before runtime.
        # However, we dont really need exact estimates of size (except when it is 0 or 1)
        # so we can provide an upper bound on sentence length (sequence size)
        sizes = self.dataset.sizes

        num_deletions = (sizes * self.delete_prob).long() + 1
        num_insertions = (sizes * self.insert_prob).long()
        return sizes + num_deletions + num_insertions

    @property
    def supports_prefetch(self):
        return False

    def prefetch(self, indices):
        if self.dataset.supports_prefetch:
            self.dataset.prefetch(indices)
            for index in indices:
                self[index]

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False
