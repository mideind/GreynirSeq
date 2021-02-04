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


# def get_char_lens(text):
#     return [len(char.encode()) for char in text]


class ByteDictionary(Dictionary):
    def __init__(
        self,
        bos_idx=256,
        pad_idx=257,
        eos_idx=258,
        byte_mask_idx=259,
        word_mask_idx=260,
        bpe_marker_idx=261,
    ):
        self.bos_idx = bos_idx
        self.pad_idx = pad_idx
        self.eos_idx = eos_idx
        self.byte_mask_idx = byte_mask_idx
        self.word_mask_idx = word_mask_idx
        self.bpe_marker_idx = bpe_marker_idx

    def bos(self):
        return self.bos_idx

    def eos(self):
        return self.eos_idx

    def pad(self):
        return self.pad_idx

    def byte_mask(self):
        return self.bos_idx

    def bpe_marker(self):
        return self.bos_idx

    def __len__(self):
        return 262
