from typing import Optional, List, Union

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset

import torch
from torch import Tensor
from fairseq import utils
from fairseq.data import LanguagePairDataset, BaseWrapperDataset, FairseqDataset
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE

from .noiser import Noiser
from .noised_sequence import NoisedSequence


class SpmNoiser(Noiser):
    def __init__(self, dictionary, spm):
        """docstring"""
        self.dictionary = dictionary
        self.spm = spm

    def apply(self, sequence: List[Union[str, int, Tensor]]) -> NoisedSequence:
        return spm_reencode_w_segmentation_noise(sequence, self.dictionary, self.spm)


def spm_reencode_w_segmentation_noise(
    sequence: List[Union[str, int, Tensor]], dictionary, spm
):
    # TODO: convert to tensor here?
    encoded_sample = []
    noise_allowed_mask = torch.tensor([], dtype=bool)

    for i in sequence:
        if isinstance(i, str):
            original_str = i  # self.spm.decode(self.src_dict.string(i))
            re_spm = spm.encode(original_str)
            re_embed = dictionary.encode_line(re_spm)
            if re_embed[-1] == dictionary.eos():
                # SPM will insert eos at the end of each substring but we only want it at the end of the whole sequence
                re_embed = re_embed[:-1]
            encoded_sample.extend(re_embed)
            noise_allowed_mask = torch.cat(
                (noise_allowed_mask, torch.tensor([True] * len(re_embed)))
            )
        elif isinstance(i, int):
            encoded_sample.append(i)
            noise_allowed_mask = torch.cat((noise_allowed_mask, torch.tensor([False])))
        else:
            assert False, f"Unknown type {type(i)}"

    encoded_sample.append(dictionary.eos())
    noise_allowed_mask = torch.cat((noise_allowed_mask, torch.tensor([False])))

    # TODO: <unk> handling
    # return torch.tensor(encoded_sample), noise_allowed_mask
    return NoisedSequence(torch.tensor(encoded_sample), noise_allowed_mask)


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
