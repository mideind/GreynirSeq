from typing import List, Tuple, Optional, Callable
from functools import lru_cache

import numpy as np

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


def space_separated_offsets(text):
    last = None
    word_offsets = []
    for idx, char in enumerate(text):
        if char != " " and (last == " " or last is None):
            word_offsets.append(idx)
        last = char
    return word_offsets


def mock_bpe_offsets(word):
    """Split words into segments of 5 characters"""
    num_parts = 1 + ((len(word) - 1) // 5)
    return [i * 5 for i in range(num_parts)]


class SpaceSeparatedOffsetsDataset(BaseWrapperDataset):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        str_seq: str = self.dataset[index]
        return LongTensor(space_separated_offsets(str_seq))


class InsertionMaskedByteSequenceDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: Dataset,
        word_byte_offsets_dataset: Dataset,
        dictionary: Dictionary,
        return_targets: bool = False,
        seed: int = 1,
        bpe_offsets_fn: Callable[[str], List[int]] = None,
    ):
        super().__init__(dataset)
        self.word_byte_offsets = word_byte_offsets_dataset
        self.return_targets = return_targets
        self.dictionary = dictionary
        self.epoch = 0
        self.seed = seed
        self.bpe_offsets_fn = bpe_offsets_fn
        self.pad_idx = [256]
        self.byte_mask_idx = 257
        self.bpe_marker_seq = [258]
        self.word_marker_seq = [259]
        self.bos_seq = [260]
        self.eos_seq = [261]

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        """WIP: Masking strategy is not implemented yet"""
        str_seq: str = self.dataset[index]
        str_bytes = str_seq.encode()
        bpe_byte_counts = [len(self.bos_seq)]
        word_byte_offsets = self.word_byte_offsets[index]

        cursor, word_start = 0, 0
        last_bpe_start, last_word_start = None, None
        bpe_offsets = []
        byte_seq = list(self.bos_seq)
        for offset, byte in enumerate(str_bytes):
            if (
                cursor < len(word_byte_offsets)
                and offset == word_byte_offsets[cursor]
            ):
                last_word_start = len(byte_seq)
                byte_seq.extend(self.word_marker_seq)
                word_start = word_byte_offsets[cursor]
                word_end = len(str_bytes)
                if cursor + 1 < len(word_byte_offsets):
                    word_end = word_byte_offsets[cursor + 1]
                word = str_seq[word_start:word_end]
                bpe_offsets = self.bpe_offsets_fn(word)
                cursor += 1
            if offset - word_start in bpe_offsets:
                bpe_start = len(byte_seq)
                if offset == last_word_start:
                    bpe_start = last_word_start
                byte_seq.extend(self.bpe_marker_seq)
                if last_bpe_start is not None:
                    length = bpe_start - last_bpe_start
                    bpe_byte_counts.append(length)
                last_bpe_start = bpe_start
            byte_seq.append(byte)

        bpe_byte_counts.append(
            len(str_bytes)
            - bpe_offsets[-1]
            - word_byte_offsets[-1]
            + len(self.bpe_marker_seq)
        )
        bpe_byte_counts.append(len(self.eos_seq))

        word_mask = [0] * len(byte_seq)
        bpe_mask = [0] * len(byte_seq)
        for idx, val in enumerate(byte_seq):
            if val in self.word_marker_seq:
                word_mask[idx] = 1
            if val in self.bpe_marker_seq:
                bpe_mask[idx] = 1

        assert len(byte_seq) == sum(bpe_byte_counts)
        return (
            LongTensor(byte_seq),
            LongTensor(bpe_byte_counts),
            BoolTensor(word_mask),
            BoolTensor(bpe_mask),
        )


class SeparateMaskedByteSequenceDataset(InsertionMaskedByteSequenceDataset):
    def __init__(
        self,
        dataset: Dataset,
        word_byte_offsets_dataset: Dataset,
        dictionary: Dictionary,
        return_targets: bool = False,
        seed: int = 1,
        bpe_offsets_fn: Callable[[str], List[int]] = None,
        char_mask_rate: float = 0.1,
        shift_char_rate: float = 0.1,
        drop_char_rate: float = 0.1,
        leave_unmasked_prob: float = 0.1,
        bpe_mask_rate: float = 0.1,
    ):
        super().__init__(
            dataset,
            word_byte_offsets_dataset,
            dictionary,
            return_targets=return_targets,
            seed=seed,
            bpe_offsets_fn=bpe_offsets_fn,
        )

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        """WIP: Masking strategy is not implemented yet"""
        str_seq: str = self.dataset[index]
        word_byte_offsets = self.word_byte_offsets[index]
        bpe_byte_counts = [len(self.bos_seq)]
        str_as_bytes = str_seq.encode()

        byte_seq = []
        byte_seq.extend(self.bos_seq)
        byte_seq.extend(str_as_bytes)
        byte_seq.extend(self.eos_seq)
        bos_len = len(self.bos_seq)

        word_mask = [0] * len(byte_seq)
        bpe_mask = [0] * len(byte_seq)
        last_bpe_start = None
        for cursor, word_start in enumerate(word_byte_offsets):
            word_mask[word_start + bos_len] = 1
            word_end = len(str_as_bytes)
            if cursor + 1 < len(word_byte_offsets):
                word_end = word_byte_offsets[cursor + 1]
            word = str_as_bytes[word_byte_offsets[cursor] : word_end].decode()
            bpe_offsets = self.bpe_offsets_fn(word)
            for bpe_offset in bpe_offsets:
                bpe_start = word_byte_offsets[cursor] + bpe_offset
                bpe_mask[bpe_start + bos_len] = 1
                if last_bpe_start is not None:
                    length = bpe_start - last_bpe_start
                    bpe_byte_counts.append(length)
                last_bpe_start = bpe_start
        bpe_byte_counts.append(
            len(str_as_bytes) - bpe_offsets[-1] - word_byte_offsets[-1]
        )

        bpe_byte_counts.append(len(self.eos_seq))
        assert len(byte_seq) == sum(bpe_byte_counts)
        return (
            LongTensor(byte_seq),
            LongTensor(bpe_byte_counts),
            BoolTensor(word_mask),
            BoolTensor(bpe_mask),
        )
