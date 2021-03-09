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
from greynirseq.nicenlp.utils.data_utils import lengths_to_offsets, lengths_to_begin_mask


class BPEEncoderDataset(BaseWrapperDataset):

    _CACHE_SIZE = 1028  # due to non-reproducibility nondeterminism in tokenizers

    def __init__(
        self,
        args: argparse.Namespace,
        dataset: Dataset,
        word_byte_offsets_dataset: Dataset,
        dictionary: Dictionary,
        byte_dictionary: Dictionary,
        seed: int = 1,
        dropout: float = 0,
        append_eos: bool = True,
    ):
        super().__init__(dataset)
        assert isinstance(
            dataset[0], str
        ), f"Expected dataset to contain str but it contains: {type(dataset[0])}"
        self.word_byte_offsets = word_byte_offsets_dataset
        self.dictionary = dictionary
        self.byte_dictionary = byte_dictionary
        self.seed = seed
        self.dropout = None if dropout is None or dropout == 0 else dropout
        self.epoch = 0
        self.append_eos = append_eos

        # TODO: receive bpeencoder as argument
        import tokenizers

        self.hf_tokenizer = tokenizers.ByteLevelBPETokenizer(
            args.gpt2_encoder_json,
            args.gpt2_vocab_bpe,
            add_prefix_space=True,
            dropout=self.dropout,
        )
        self._sizes = self.dataset._sizes

    @lru_cache(maxsize=_CACHE_SIZE)
    def __getitem__(self, index: int) -> ByteSequence:
        str_seq: str = self.dataset[index]
        if not str_seq:
        if not str_seq and self.append_eos:
            byte_eos_t = torch.tensor([self.byte_dictionary.eos()]).long()
            bpe_eos_t = torch.tensor([self.dictionary.eos()]).long()
            one_t = torch.tensor([1]).long()
            return ByteSequence("", byte_eos_t, bpe_lens=one_t, bpe_mask=one_t, bpe_ids=bpe_eos_t, word_lens=one_t, word_mask=one_t)
        elif not str_seq:
            empty = torch.tensor([]).long()
            return ByteSequence("", empty, bpe_lens=empty, bpe_mask=empty, bpe_ids=empty, word_lens=empty, word_mask=empty)

        word_byte_offsets = self.word_byte_offsets[index]
        byte_seq = str_seq.encode()
        assert len(byte_seq) >= 1, "empty lines are currently not supported"
        seq_hf_bpe_ids = []
        word_lens = []
        seq_bpe_offsets = []
        bpe_lens = []
        hf_space_id = self.hf_tokenizer.encode(" ").ids[0]
        for (word_start, word_end) in zip(
            word_byte_offsets, word_byte_offsets[1:].tolist() + [len(byte_seq)]
        ):
            # NOTE: word_byte_offsets have whitespace on the left side of words
            #       while Huggingface expects it on the right side.
            #       Due to pretokenization in huggingface, a single leading space
            #       is a de facto indicator of word-initial BPE tokens
            word_lens.append(word_end - word_start.item())
            word = byte_seq[word_start:word_end].decode()
            hf_encoding = self.hf_tokenizer.encode(word.rstrip())

            # NOTE: due to BPE-dropout, the merger of the prefixed can be dropped,
            #      if the original token did not contain a space, but since it was added by hf_tokenizer,
            #      the space will be a separate bpe segment, it has the same span as following segment (part of the actual token)
            #      e.g.  "6" becomes " 6" due to space prefixing, which will be split into " " at span (0, 1) and "6" at span (0,1)
            #
            #      A similar situation also occurs when multi-byte unicode symbols are split,
            #      (each part has the same origin span)
            #
            #      otherwise this would not be an explicit for-loop
            new_start_offsets = []
            accum = 0
            delta = 0
            new_ids = []
            for i, (segment_str, hf_segment_id) in enumerate(zip(hf_encoding.tokens, hf_encoding.ids)):
                segment_start_offset = accum + word_start.item() - delta  # the start token is always at relative 0
                accum += len(segment_str)
                if  i == 0 and word[0] != " " and hf_segment_id != hf_space_id:
                    # example: word = ":)" and in hf sees it as " :)" and segments as [" :)"]
                    # discard the space from offsets calculation (it is not in byte_seq)
                    delta = 1
                    # do not discard bpe_segment, bpe segment is its word-initial variant
                elif i == 0 and word[0] != " ":
                    # example: word = ":)" and in hf sees it as " :)" and segments as [" ", ":)"]
                    # discard the space from offsets calculation and from ids
                    delta = 1
                    # discard bpe_segment, following bpe segment will be a word-internal variant
                    continue
                else:
                    # example: word = " :)" and in hf sees it as " :)" and segments as [" ", ":)"]
                    # we want to keep the space in offsets calculation and in ids (it is in byte_seq)
                    pass
                new_start_offsets.append(segment_start_offset)
                new_ids.append(str(hf_segment_id))

            seq_bpe_offsets.extend(new_start_offsets)
            seq_hf_bpe_ids.extend(new_ids)

        fairseq_bpe_ids = self.dictionary.encode_line(" ".join(seq_hf_bpe_ids), append_eos=False, add_if_not_exist=False).long()

        byte_seq = torch.tensor(list(byte_seq), dtype=torch.long)
        # starting offsets, these are for temporal pooling (contraction), so they include the whitespace
        seq_bpe_offsets.append(len(byte_seq))
        seq_bpe_offsets = torch.tensor(seq_bpe_offsets, dtype=torch.long)
        bpe_lens = seq_bpe_offsets[1:] - seq_bpe_offsets[:-1]
        if 0 in bpe_lens:
            ic(str_seq, byte_seq, fairseq_bpe_ids, bpe_lens, len(byte_seq), sum(bpe_lens), sum(word_lens), len(fairseq_bpe_ids), word_lens)
            import pdb; pdb.set_trace()
            print(flush=True)
        assert 0 not in bpe_lens

        if self.append_eos:
            fairseq_bpe_ids = torch.cat([fairseq_bpe_ids, torch.tensor([self.dictionary.eos()])])
            byte_seq = torch.cat([byte_seq, torch.tensor([self.byte_dictionary.eos()])])
            bpe_lens = torch.cat([bpe_lens, torch.tensor([1])])
            word_lens.extend([1])

        bpe_mask = torch.zeros_like(byte_seq, dtype=torch.bool)
        bpe_mask[lengths_to_offsets(bpe_lens)] = 1
        word_lens = torch.tensor(word_lens, dtype=torch.long)
        word_mask = torch.zeros_like(byte_seq, dtype=torch.bool)
        word_mask[lengths_to_offsets(word_lens)] = 1

        assert 0 not in bpe_lens
        assert bpe_lens.sum() == len(byte_seq)
        assert bpe_lens.shape == fairseq_bpe_ids.shape


        assert len(bpe_lens) == bpe_mask.sum()
        assert bpe_lens.sum() == sum(word_lens)

        seq = ByteSequence(
            str_seq=str_seq,
            byte_seq=byte_seq,
            bpe_ids=fairseq_bpe_ids,
            bpe_lens=bpe_lens,
            word_lens=word_lens,
            word_mask=word_mask,
            bpe_mask=bpe_mask,
        )

        return seq

    @property
    def sizes(self):
        return self._sizes + (1 if self.append_eos else 0)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False

    @property
    def supports_prefetch(self):
        return True

    def prefetch(self, indices):
        assert len(indices) <= self._CACHE_SIZE
        for index in indices:
            self[index]
