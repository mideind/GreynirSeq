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


from greynirseq.nicenlp.data.byte_dictionary import ByteDictionary

def space_separated_offsets(text):
    last = None
    word_offsets = []
    for idx, char in enumerate(text):
        if char == " " or last is None:
            word_offsets.append(idx)
        last = char
    return word_offsets


def mock_bpe_offsets(word):
    """Split words into segments of 5 characters"""
    num_parts = 1 + ((len(word) - 1) // 5)
    return [i * 5 for i in range(num_parts)]


def lengths_to_offsets(lengths: torch.Tensor):
    offsets = lengths.cumsum(dim=0).roll(1, dims=[0])  # shift right
    offsets[0] = 0  # first item is always at 0 offset
    return offsets


class SpaceSeparatedOffsetsDataset(BaseWrapperDataset):
    def __init__(self, dataset: Dataset):
        super().__init__(dataset)

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        str_seq: str = self.dataset[index]
        return torch.tensor(space_separated_offsets(str_seq), dtype=torch.long)


class SeparateMaskedByteSequenceDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: Dataset,
        bpe_dictionary: Optional[Dictionary],
        bpe_mask_prob: float=0.15,
        word_mask_prob float=0.15,
        word_dictionary: Optional[Dictionary]=None,
        seed: int = 1,
    ):
        super().__init__(dataset)
        self.return_targets = return_targets
        self.dictionary = dictionary
        self.epoch = 0
        self.seed = seed
        self.bpe_mask_prob = bpe_mask_prob
        self.word_mask_prob = word_mask_prob
        if (bpe_mask_prob is None or bpe_mask_prob <= 0) and (word_mask_prob is None or word_mask_prob <= 0):
            raise ValueError("bpe_mask_prob and word_mask_prob cannot both be zero or None")

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        """WIP: Masking strategy is not implemented yet"""
        seq = self.dataset[index]
        word_byte_offsets = lengths_to_offsets(seq.word_lens)
        bpe_byte_offsets = lengths_to_offsets(seq.bpe_lens)

        """
        string
        get word boundaries
        icelandic character noise
        get byteseq
        bytenoise

        get masking strategy
            (length-based, bpe-based, whole-word-based)
        make targets
            if bpe, then bpe encode input and sample targets
            if whole-word, then sample targets (that are in the frequent-word-vocabulary)
            if length based, then sample starting points and lengths
            (need bpe encoding sampling bpe targets, need word vocab for word ids)
        make target mask
        make contraction offsets, respecting mask and bpe boundaries (or whole-word boundaries)

        """

        new_byte_seq = seq.byte_seq.tolist()
        new_byte_seq.append(self.byte_dictionary.eos_idx)

        new_bpe_lens = seq.bpe_lens.tolist()
        new_bpe_lens.append(1)

        new_seq = seq.clone()
        new_seq.byte_seq = torch.tensor(new_byte_seq).long()
        new_seq.bpe_lens = torch.tensor(new_bpe_lens).long()
        assert (
            new_seq.byte_seq.numel() == new_seq.bpe_lens.sum()
        ), "Expected byte sequence length to match bpe lengths"

        word_begin_mask = torch.zeros_like(new_seq.byte_seq)
        word_begin_mask[word_byte_offsets] = 1
        bpe_begin_mask = torch.zeros_like(word_begin_mask)
        bpe_begin_mask[bpe_byte_offsets] = 1

        new_seq.word_mask = word_begin_mask
        new_seq.bpe_mask = bpe_begin_mask
        return new_seq


class MaskedByteSequenceDataset(SeparateMaskedByteSequenceDataset):
    pass
