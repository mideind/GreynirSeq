# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.


import numpy as np
import torch
from fairseq.data import Dictionary
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE

from greynirseq.nicenlp.data.char_noise import CharacterNoiser, CharacterNoiserConfig
from greynirseq.nicenlp.data.fragment_noise import FragmentNoiser
from greynirseq.nicenlp.data.spm_segmentation_noise import SpmNoiser
from greynirseq.nicenlp.data.word_noise import WordNoiser, WordNoiserConfig


class Encoder:
    """Encodes a sequence of tokens into a sequence of integers using BPE and then the fairseq dictionary.

    The encoded sequence can also be noised using various noise models.
    Special care needs to be taken in which order the noising is applied since some models expect the input to be
    strings, and others expect integers.
    """

    def __init__(
        self,
        bpe: SentencepieceBPE,
        noisy_bpe: SentencepieceBPE,
        dictionary: Dictionary,
        allowed_dictionary_min: int,
        allowed_dictionary_max: int,
        fragment_noise_prob: float,
        global_skip_noise_prob: float,
        word_noise_config: WordNoiserConfig,
        char_noise_config: CharacterNoiserConfig,
        max_sequence_length: int,
    ):
        self.bpe = bpe
        self.dictionary = dictionary

        self.word_noiser = WordNoiser(word_noise_config)
        self.noisy_subword_enc = SpmNoiser(self.dictionary, noisy_bpe=noisy_bpe)
        self.global_skip_noise_prob = global_skip_noise_prob
        self.fragment_noiser = FragmentNoiser(
            fragment_noise_prob, min_val=allowed_dictionary_min, max_val=allowed_dictionary_max
        )
        self.char_noiser = CharacterNoiser(
            char_noiser_config=char_noise_config,
        )
        self.max_sequence_length = max_sequence_length

    def encode(self, sequence: str) -> torch.Tensor:
        """Encode a sequence of tokens into a sequence of integers using BPE and then the fairseq dictionary."""
        return self.dictionary.encode_line(self.bpe.encode(sequence), append_eos=False, add_if_not_exist=False).long()

    def encode_noisy(self, sequence: str) -> torch.Tensor:
        """Encode a sequence of tokens into a sequence of integers using BPE and then the fairseq dictionary and
        apply noise to the sequence."""
        if np.random.rand() < self.global_skip_noise_prob:
            return self.encode(sequence)
        sequence = self.char_noiser.apply(sequence)
        res = self.word_noiser.apply(sequence)
        res = self.noisy_subword_enc.apply(res)
        seq_tensor = self.fragment_noiser.apply(res.sequence, res.noise_allowed_mask)
        # If the noisy sequence is too long, we encode it again without noise
        if len(seq_tensor) > self.max_sequence_length:
            return self.encode(sequence)
        return seq_tensor
