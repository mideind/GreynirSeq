from typing import List, Optional, Union

import torch
from fairseq import utils
from fairseq.data import BaseWrapperDataset, FairseqDataset, LanguagePairDataset
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset
from torch import Tensor

from .noiser import Noiser


class WordNoiser(Noiser):
    def __init__(self, prob, max_shuffle_distance):
        """docstring"""
        self.prob = prob
        self.max_shuffle_distance = max_shuffle_distance

    def apply(self, sequence: List[Union[str, int, Tensor]]):
        return word_noise(
            sequence,
            self.prob,
            max_shuffle_distance=self.max_shuffle_distance,
        )


def calculate_dropout_keep_matrix(length, prob):
    """
    Calculate a keep-iff-true matrix for a sequence of the given length
    """
    output = torch.full((length,), True)
    # Don't drop anything for sequences that are too short
    MIN_LENGTH = 3
    if length <= MIN_LENGTH:
        return output

    # Otherwise, use prob as the target fraction of elements to drop
    # Use stochastic rounding
    drop_amount = int(length * prob + torch.rand(1).item())
    # Still force a minimum amount remaining
    drop_amount = min(drop_amount, length - MIN_LENGTH)

    # Low drop probability will sometimes try to drop nothing in short sequences
    if drop_amount == 0:
        return output

    drop_weights = torch.ones((length,))
    drop_indexes = torch.multinomial(drop_weights, drop_amount)
    output[drop_indexes] = False

    return output


def word_noise(
    sequence: List[Union[str, int, Tensor]], prob, *, max_shuffle_distance: int
):
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
