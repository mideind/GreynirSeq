# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

from dataclasses import dataclass, field

import torch
from fairseq.dataclass import FairseqDataclass
from torch import Tensor

from .noiser import Noiser

# The probability of a word being shifted to a different position
_POS_PROB = 0.0025  # results in 0.5-1.0% words being shifted using categorial distribution
# The minimum length of a sequence to apply word noise to
MIN_LENGTH = 3


@dataclass
class WordNoiserConfig(FairseqDataclass):
    drop_word_prob: float = field(
        default=0.1,
        metadata={"help": "Probability of dropping a word"},
    )
    max_shift_distance: int = field(
        default=3,
        metadata={"help": "Maximum distance to shift a word. 0 means no shifting."},
    )
    shift_prob: float = field(
        default=_POS_PROB,
        metadata={"help": "Probability of each word being shifted to a different position"},
    )


class WordNoiser(Noiser):
    """Noiser that applies word-level noise to a sequence of strings.

    The noise consists of:
    - Dropping words (with a given probability)
    - Shifting words to a different position (with a probability that decreases with distance)

    A word is a sequence of characters separated by spaces.
    """

    def __init__(self, config: WordNoiserConfig):
        self.config = config
        # Create a categorical distribution for the word position noise
        # Example: for max_shuffle_distance=3 and pos_prob=0.0025, we get
        # [0.9875, 0.0025, 0.0025, 0.0025]
        # which means that 98.75% of words are not shifted, 0.25% are shifted by 1,
        # 0.25% by 2, and 0.25% by 3 positions.
        self.pos_noise_dist = get_shift_distribution(
            max_shuffle_distance=config.max_shift_distance, shift_prob=config.shift_prob
        )

    def apply(self, sequence: str):
        return word_noise(
            sequence,
            self.config.drop_word_prob,
            pos_noise_dist=self.pos_noise_dist,
            max_shuffle_distance=self.config.max_shift_distance,
        )


def get_shift_distribution(max_shuffle_distance: int, shift_prob) -> torch.distributions.Distribution:
    pos_noise_probs = torch.tensor(
        [1 - max_shuffle_distance * shift_prob] + [shift_prob] * max_shuffle_distance,
        dtype=torch.float,
    )
    return torch.distributions.categorical.Categorical(probs=pos_noise_probs)


def calculate_dropout_keep_matrix(length: int, drop_prob: float) -> Tensor:
    """Calculate a boolean matrix of length `length` where each element is True with probability 1-drop_prob."""
    keep_matrix = torch.full((length,), True)
    if length <= MIN_LENGTH:
        return keep_matrix

    # Calculate the number of words to drop using stochastic rounding
    drop_amount = int(length * drop_prob + torch.rand(1).item())
    # Still force a minimum amount remaining
    drop_amount = min(drop_amount, length - MIN_LENGTH)

    # If we're not dropping any words, return the keep matrix as-is
    if drop_amount <= 0:
        return keep_matrix

    # Calculate a multinomial distribution for dropping words
    drop_weights = torch.ones((length,))
    drop_indexes = torch.multinomial(drop_weights, num_samples=drop_amount, replacement=False)
    keep_matrix[drop_indexes] = False

    return keep_matrix


def word_noise(
    sequence: str,
    drop_prob: float,
    *,
    pos_noise_dist: torch.distributions.Distribution,
    max_shuffle_distance: int,
) -> str:
    # Consider words to be space separated.
    # Consecutive spaces will cause empty-string words, but that's fine.
    words = sequence.split(" ")
    num_words = len(words)

    unnoised_pos = torch.arange(num_words)
    if max_shuffle_distance > 0:
        # For each word, sample a distance to shift it by
        pos_noise = pos_noise_dist.sample(sample_shape=torch.Size((num_words,)))
        # Example for max_shuffle_distance=3 and num_words=4:
        #   pos_noise           = [1, 2, 1, 0]
        #   unnoised_pos        = [0, 1, 2, 3]
        #   shifted_pos_noise   = [1, 3, 3, 3]
        # TODO: This seems to be a bug - i.e. a shift of 1 seems to imply that the word would get shifted
        # but the example above shows that the first word is not shifted - since it has the same index as another word.
        noised_pos = unnoised_pos + pos_noise
        # We then do an argsort to get the indexes of the words in the new order
        perm = noised_pos.argsort()
    else:
        perm = unnoised_pos

    # Drop words with probability p,
    # keep_matrix = self.calculate_dropout_keep_matrix(len(words))
    keep_matrix = calculate_dropout_keep_matrix(num_words, drop_prob)
    reordered_words = [words[i] for i in perm if keep_matrix[i]]
    reordered_part = " ".join(reordered_words)

    return reordered_part
