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


class BPEEncoderDataset(BaseWrapperDataset):
    def __init__(
        self,
        args: argparse.Namespace,
        dataset: Dataset,
        word_byte_offsets_dataset: Dataset,
        dictionary: Dictionary,
        seed: int = 1,
        dropout: float = 0,
    ):
        super().__init__(dataset)
        assert isinstance(
            dataset[0], str
        ), f"Expected dataset to contain str but it contains: {type(dataset[0])}"
        self.word_byte_offsets = word_byte_offsets_dataset
        self.dictionary = dictionary
        self.seed = seed
        self.dropout = None if dropout is None or dropout == 0 else dropout
        self.epoch = 0

        # TODO: receive bpeencoder as argument
        import tokenizers

        self.hf_tokenizer = tokenizers.ByteLevelBPETokenizer(
            args.gpt2_encoder_json,
            args.gpt2_vocab_bpe,
            add_prefix_space=True,
            dropout=self.dropout,
        )
        self._sizes = self.dataset._sizes

    @lru_cache(maxsize=8)
    def __getitem__(self, index: int) -> ByteSequence:
        str_seq: str = self.dataset[index]
        word_byte_offsets = self.word_byte_offsets[index]
        byte_seq = str_seq.encode()
        seq_hf_bpe_ids = []
        word_lens = []
        seq_bpe_offsets = []
        bpe_lens = []
        for (word_start, word_end) in zip(
            word_byte_offsets, word_byte_offsets[1:].tolist() + [len(str_seq)]
        ):
            # NOTE: word_byte_offsets have whitespace on the left side of words
            #       while Huggingface expects it on the right side.
            #       Due to pretokenization in huggingface, a single leading space
            #       is a de facto indicator of word-initial BPE tokens
            word_lens.append(word_end - word_start)
            word = byte_seq[word_start:word_end].decode()
            hf_encoding = self.hf_tokenizer.encode(word.rstrip())
            seq_bpe_offsets.extend(
                [
                    word_start + bpe_start
                    for (bpe_start, _bpe_end) in hf_encoding.offsets
                ]
            )
            bpe_lens.extend(
                [bpe_end - bpe_start for (bpe_start, bpe_end) in hf_encoding.offsets]
            )
            seq_hf_bpe_ids.extend([str(hf_bpe_id) for hf_bpe_id in hf_encoding.ids])
        fairseq_bpe_ids = self.dictionary.encode_line(" ".join(seq_hf_bpe_ids))

        # starting offsets, these are for temporal contraction, so they include the whitespace
        seq_bpe_offsets = torch.tensor(seq_bpe_offsets, dtype=torch.long)
        bpe_lens = torch.tensor(bpe_lens, dtype=torch.long)

        seq = ByteSequence(
            str_seq=str_seq,
            byte_seq=torch.tensor(list(byte_seq), dtype=torch.long),
            bpe_ids=fairseq_bpe_ids,
            bpe_lens=bpe_lens,
            word_lens=torch.tensor(word_lens, dtype=torch.long),
        )

        return seq

    @property
    def sizes(self):
        return self._sizes
