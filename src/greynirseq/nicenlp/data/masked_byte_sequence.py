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


from greynirseq.nicenlp.byte_sequence import ByteSequence
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


class MaskedByteSequenceDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: Dataset,
        bpe_dictionary: Optional[Dictionary],
        mask_prob: float = 0.15,
        word_dictionary: Optional[Dictionary] = None,
        seed: int = 1,
        prohibit_first=True,
    ):
        super().__init__(dataset)
        self.epoch = 0
        self.seed = seed
        self.mask_prob = mask_prob
        self.bpe_dictionary = bpe_dictionary
        self.word_dictionary = word_dictionary
        # TODO: length-based masking
        self.masking_strategy = self.uniform_strategy
        self.prohibit_first = prohibit_first

    def uniform_strategy(self, seq: ByteSequence) -> torch.Tensor:
        assert seq.targets is None
        assert seq.target_mask is None
        assert self.word_dictionary is None  # defensiveness
        length = len(seq.bpe_ids)
        mask = torch.rand(length) <= self.mask_prob
        mask[length - 1] = False
        if self.prohibit_first:
            mask[0] = False
        targets = torch.randint(
            low=self.bpe_dictionary.nspecial,
            high=len(self.bpe_dictionary),
            size=(length,),
        )
        if not torch.any(targets.sum() > 0):
            raise ValueError("Unexpected negative length")

        return targets[mask], mask

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        """WIP: Masking strategy is not implemented yet"""
        seq = self.dataset[index]

        with data_utils.numpy_seed(self.seed, self.epoch, index):
            targets, mask = self.masking_strategy(seq)

        seq.targets = targets
        seq.target_mask = mask
        return seq
