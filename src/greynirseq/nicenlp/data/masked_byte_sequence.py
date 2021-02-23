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
from greynirseq.nicenlp.utils.data_utils import lengths_to_begin_mask


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
        mask_prob: float = 0.15,  # probability that "masking" occurs (as defined in original BERT paper)
        word_dictionary: Optional[Dictionary] = None,
        seed: int = 1,
        prohibit_first=True,
        mask_weight=80,
        noop_weight=10,
        byte_mask_index=None,
    ):
        super().__init__(dataset)
        if byte_mask_index is None:
            raise ValueError("byte_mask_index must be provided")
        self.epoch = 0
        self.seed = seed
        self.mask_prob = mask_prob
        self.byte_mask_index = byte_mask_index
        # XXX: BERT has replace but that doesnt work so cleanly as with BPE,
        #      since we need to swap out multiple characters
        total_weight = mask_weight + noop_weight
        assert total_weight > 0
        self.mask_rate = mask_weight / total_weight  # rate at which masking proceeds normally
        self.noop_rate = noop_weight / total_weight
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
        mask = torch.zeros_like(seq.bpe_ids).bool()

        # NOTE: why should this not be sampled from wider distribution (e.g. a gaussian)?
        #       the no-operation action does this! Albeit uniformly
        num_actions = int(self.mask_prob * length + np.random.rand())  # probabalistic rounding

        mask[np.random.choice(length, num_actions, replace=False)] = True

        noop = torch.rand(length) <= self.noop_rate
        assert seq.bpe_lens.sum() == len(seq.byte_seq)

        new_seq = seq.clone()
        byte_mask_tensor = torch.tensor([self.byte_mask_index])
        use_mask = mask * noop.logical_not()
        new_seq.bpe_ids[use_mask] = self.bpe_dictionary.index("<mask>")
        new_seq.bpe_lens[use_mask] = 1
        byte_subseqs = [
            byte_mask_tensor if use_mask_ else subseq
            for (subseq, use_mask_) in zip(
                    seq.byte_seq.split(seq.bpe_lens.tolist(), dim=0),
                    use_mask.tolist(),
            )
        ]

        new_seq.byte_seq = torch.cat(byte_subseqs)
        new_seq.bpe_mask = lengths_to_begin_mask(new_seq.bpe_lens)
        new_seq.targets = seq.bpe_ids.clone()
        new_seq.target_mask = mask

        return new_seq

    @lru_cache(maxsize=32)
    def __getitem__(self, index):
        seq = self.dataset[index]

        with data_utils.numpy_seed(self.seed, self.epoch, index):
            new_seq = self.masking_strategy(seq)

        assert torch.all(new_seq.bpe_lens.gt(0)), "Unexpected bpe token with length 0"
        return new_seq

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False
