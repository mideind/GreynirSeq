from typing import List, Tuple, Optional, Callable
from functools import lru_cache
import unittest

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
from fairseq.data import data_utils, ListDataset

from greynirseq.nicenlp.data.masked_byte_sequence import SeparateMaskedByteSequenceDataset, SpaceSeparatedOffsetsDataset, mock_bpe_offsets, space_separated_offsets

class TestMaskedByteSequenceDataset(unittest.TestCase):
    def test_forward(self):
        text = "Is sequence"
        dikt = Dictionary()
        for sym in text.split(" "):
            dikt.add_symbol(sym)

        words = ["Is", "sequence"]
        word_offsets = space_separated_offsets(text)
        bpe_strings = [mock_bpe_offsets(w) for w in words]
        self.assertEquals(word_offsets, [0, 3])
        self.assertEquals(bpe_strings, [[0], [0, 5]])

        dataset = SeparateMaskedByteSequenceDataset(
            ListDataset([text]),
            ListDataset([word_offsets]),
            dikt,
            bpe_offsets_fn=mock_bpe_offsets,
        )
        seq, byte_counts_per_bpe, word_mask, bpe_mask = dataset[0]
        # dots are bos/eos                                   .  I  s  _  S  e  q  u  e  n  c  e  .
        self.assertEquals(list(word_mask), [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEquals(list(bpe_mask), [0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0])
        # skip bos/eos in sequences
        self.assertEquals(list(seq[1:-1]), list(text.encode()))
        self.assertEquals(list(byte_counts_per_bpe[1:-1]), [3, 5, 3])
