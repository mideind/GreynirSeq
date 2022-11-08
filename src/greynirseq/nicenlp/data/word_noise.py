# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

from typing import List, Union

import torch
from fairseq.data import BaseWrapperDataset
from torch import Tensor

from .noiser import Noiser

_POS_PROB = 0.0025  # results in 0.5-1.0% words being shifted using categorial distribution


class WordNoiser(Noiser):
    def __init__(self, drop_prob: float, max_shuffle_distance: int, pos_prob: float = _POS_PROB):
        self.drop_prob = drop_prob
        self.pos_prob = pos_prob
        self.max_shuffle_distance = max_shuffle_distance
        self.pos_noise_probs = torch.tensor(
            [1 - max_shuffle_distance * self.pos_prob] + [self.pos_prob] * max_shuffle_distance, dtype=torch.float
        )
        self.pos_noise_dist = torch.distributions.categorical.Categorical(probs=self.pos_noise_probs)

    def apply(self, sequence: List[Union[str, int, Tensor]]):
        return word_noise_v2(
            sequence,
            self.drop_prob,
            pos_noise_dist=self.pos_noise_dist,
            max_shuffle_distance=self.max_shuffle_distance,
        )


def calculate_dropout_keep_matrix(length, drop_prob):
    """
    Calculate a keep-iff-true matrix for a sequence of the given length
    """
    keep_matrix = torch.full((length,), True)
    # Don't drop anything for sequences that are too short
    MIN_LENGTH = 3
    if length <= MIN_LENGTH:
        return keep_matrix

    # Otherwise, use drop_prob as the target fraction of elements to drop
    # Use stochastic rounding
    drop_amount = int(length * drop_prob + torch.rand(1).item())
    # Still force a minimum amount remaining
    drop_amount = min(drop_amount, length - MIN_LENGTH)

    # Low drop probability will sometimes try to drop nothing in short sequences
    if drop_amount <= 0:
        return keep_matrix

    drop_weights = torch.ones((length,))
    drop_indexes = torch.multinomial(drop_weights, drop_amount)
    keep_matrix[drop_indexes] = False

    return keep_matrix


def word_noise_v2(sequence: List[Union[str, int, Tensor]], drop_prob, *, pos_noise_dist, max_shuffle_distance: int):
    noised_sample = []
    for part in sequence:
        if isinstance(part, str):
            # Only add noise to strings

            # Consider words to be space separated.
            # Consecutive spaces will cause empty-string words, but that's fine.
            words = part.split(" ")

            # Shuffle words to within-k distance
            unnoised_pos = torch.arange(len(words))
            if max_shuffle_distance > 0:
                # pos_noise = torch.randint(0, max_shuffle_distance, (len(words),))
                pos_noise = pos_noise_dist.sample(sample_shape=(len(words),))
                noised_pos = unnoised_pos + pos_noise
                perm = noised_pos.argsort()
            else:
                perm = unnoised_pos

            # Drop words with probability p,
            # keep_matrix = self.calculate_dropout_keep_matrix(len(words))
            keep_matrix = calculate_dropout_keep_matrix(len(words), drop_prob)
            reordered_words = [words[i] for i in perm if keep_matrix[i]]
            reordered_part = " ".join(reordered_words)

            noised_sample.append(reordered_part)
        else:
            noised_sample.append(part)

    return noised_sample


def word_noise(sequence: List[Union[str, int, Tensor]], prob, *, max_shuffle_distance: int):
    noised_sample = []
    for part in sequence:
        if isinstance(part, str):
            # Only add noise to strings

            # Consider words to be space separated.
            # Consecutive spaces will cause empty-string words, but that's fine.
            words = part.split(" ")

            # Shuffle words to within-k distance
            unnoised_pos = torch.arange(len(words))
            if max_shuffle_distance > 0:
                pos_noise = torch.randint(0, max_shuffle_distance, (len(words),))
                noised_pos = unnoised_pos + pos_noise
                perm = noised_pos.argsort()
            else:
                perm = unnoised_pos

            # Drop words with probability p,
            # keep_matrix = self.calculate_dropout_keep_matrix(len(words))
            keep_matrix = calculate_dropout_keep_matrix(len(words), prob)
            reordered_words = [words[i] for i in perm if keep_matrix[i]]
            reordered_part = " ".join(reordered_words)

            noised_sample.append(reordered_part)
        else:
            noised_sample.append(part)

    return noised_sample


class WordNoiseDataset(BaseWrapperDataset):
    """
    Apply word noise to the string segments in a sequence.

    The types of noise are:
        within-k word shuffling
        word dropout

    Input should be of type List[Union[str, int]]
        strings are literal strings in the sample
        ints are embedding indexes from the source dict
    Output type is the same List[Union[str, int]]
    """

    """
    splitta hverjum streng á bilum og rugla innan strengs

    -- word shuffling
    búa til 0..n-1 vector (arange) af lengd #fjöldi-orða-í-splitti
    leggja við hvert stak random tölu á bilinu [0-k[ (þar sem k er max-dist í umröðuninni)
    sortera fylkið  og taka út .indexes (það er í boði í numpy - sjá líka bart kóðann) sem segir umröðunina sem sorteringin gerði
    nota umröðunina til að endurraða orðum indexa upp á nýtt

    -- word dropout
    henda orðum úr listanum með líkum

    splæsa saman með bilum
    """

    def __init__(self, args, dataset, noise_prob, max_shuffle_distance):
        super().__init__(dataset)
        # TODO: should noise_prob and max_shuffle_distance be part of args?
        self.p = noise_prob
        self.k = max_shuffle_distance

    def __getitem__(self, index):
        item = self.dataset[index]
        return word_noise(item, self.p, max_shuffle_distance=self.k)
