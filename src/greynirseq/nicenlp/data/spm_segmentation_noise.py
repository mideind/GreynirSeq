# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.


import torch
from fairseq.data import BaseWrapperDataset, Dictionary
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE

from .noised_sequence import NoisedSequence
from .noiser import Noiser


class SpmNoiser(Noiser):
    """Encode input encoded with SentencepieceBPE using sampling and possibly alpha."""

    def __init__(self, dictionary: Dictionary, noisy_bpe: SentencepieceBPE):
        self.dictionary = dictionary
        self.noisy_spm = noisy_bpe

    def apply(self, sequence: str) -> NoisedSequence:
        return spm_reencode_w_segmentation_noise(sequence, self.dictionary, self.noisy_spm)


def spm_reencode_w_segmentation_noise(sequence: str, dictionary: Dictionary, noisy_spm: SentencepieceBPE):
    """Encode input encoded with SentencepieceBPE using sampling and possibly alpha."""
    noise_allowed_mask = torch.tensor([], dtype=torch.bool)

    spm = noisy_spm.encode(sequence)
    encoded = dictionary.encode_line(spm, append_eos=False, add_if_not_exist=False).long()
    noise_allowed_mask = torch.tensor([True] * len(encoded))
    return NoisedSequence(encoded, noise_allowed_mask)


class SentencepieceSegmentationNoiseDataset(BaseWrapperDataset):
    """
    Re-encode input encoded with SentencepieceBPE, using BPE dropout.

    Input can be of type str or list[Union[str, int]]
    If the input is str it is passed directly to spm
    If the input is a list, each string is individually encoded with spm and then
        concatted in order with the ints in the input.
    """

    def __init__(self, args, dataset, src_dict):
        super().__init__(dataset)
        self.spm = SentencepieceBPE(args)
        self.src_dict = src_dict

    def __getitem__(self, index: int) -> NoisedSequence:
        item = self.dataset[index]
        if isinstance(item, str):
            item = [item]
        return spm_reencode_w_segmentation_noise(item, self.src_dict, self.spm)
