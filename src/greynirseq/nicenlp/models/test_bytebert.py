import unittest

import torch

from fairseq.data import ListDataset, Dictionary, NumelDataset

from greynirseq.nicenlp.data.masked_byte_sequence import (
    SeparateMaskedByteSequenceDataset,
    SpaceSeparatedOffsetsDataset,
    mock_bpe_offsets,
)
from greynirseq.nicenlp.models.bytebert import ByteBertSentenceEncoder


def mock_bpe_offsets(word):
    num_parts = 1 + ((len(word) - 1) // 5)
    return [i * 5 for i in range(num_parts)]


class TestByteBert(unittest.TestCase):
    def test_forward(self):
        bsz = 1
        num_words = 2
        embed_dim = 8
        pad_idx = 256
        vocab_size = 265
        seq = torch.LongTensor([[2, 3, 4]])
        lens = torch.LongTensor([[1, 2]])
        num_bpe = torch.LongTensor([2])
        wmask = torch.BoolTensor([[True, False, True]])
        bmask = torch.BoolTensor([[True, True, False]])

        model = ByteBertSentenceEncoder(pad_idx, vocab_size, embedding_dim=embed_dim, character_embed_dim=5)
        inner_states, _extra = model(seq, num_bpe, lens, word_mask=wmask, bpe_mask=bmask)
        self.assertEquals(list(inner_states[-1].shape), [num_words, bsz, embed_dim])
