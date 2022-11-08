from typing import List, Union

import torch
from fairseq.data import encoders

from greynirseq.nicenlp.data.fragment_noise import FragmentNoiser
from greynirseq.nicenlp.data.spm_segmentation_noise import SpmNoiser
from greynirseq.nicenlp.data.word_noise import WordNoiser


class Encoder:
    def __init__(self, args, dictionary, min_val: int, max_val: int):
        """docstring"""
        self.args = args
        self.bpe = encoders.build_bpe(self.args)
        self.dictionary = dictionary
        self.min_val = min_val
        self.max_val = max_val
        import copy

        _args_w_bpe_sampling = copy.deepcopy(self.args)
        _args_w_bpe_sampling.bpe = "sentencepiece_sampled"
        self.bpe_noisy = encoders.build_bpe(_args_w_bpe_sampling)

        word_noise_prob = self.args.word_noise_prob
        max_shuffle_dist = self.args.max_shuffle_dist
        fragment_noise_prob = self.args.fragment_noise_prob
        self.word_noiser = WordNoiser(word_noise_prob, max_shuffle_dist)
        self.noisy_subword_enc = SpmNoiser(self.dictionary, self.bpe_noisy)
        self.fragment_noiser = FragmentNoiser(fragment_noise_prob, min_val=self.min_val, max_val=self.max_val)

    def encode(self, parts: List[Union[int, str]]):
        if isinstance(parts, (str, int)):
            parts = [parts]
        encoded_parts = []
        encoded_parts = [
            self.dictionary.encode_line(self.bpe.encode(part), append_eos=False, add_if_not_exist=False).long()
            if isinstance(part, str)
            else torch.tensor([part]).long()
            for part in parts
        ]
        return torch.cat(encoded_parts).long()

    def encode_noisy(self, sequence: List[Union[str, int, torch.Tensor]]):
        if not isinstance(sequence, list):
            sequence = [sequence]
        res = self.word_noiser.apply(sequence)
        res = self.noisy_subword_enc.apply(res)
        seq_tensor = self.fragment_noiser.apply(res.sequence, res.noise_allowed_mask)
        return seq_tensor
