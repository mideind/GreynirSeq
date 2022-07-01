from typing import Optional

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset

import torch
from torch import Tensor
from fairseq import utils
from fairseq.data import LanguagePairDataset, BaseWrapperDataset, FairseqDataset
from fairseq.data.encoders.sentencepiece_bpe import SentencepieceBPE

from icecream import ic
from .noiser import Noiser


class FragmentNoiser(Noiser):
    def __init__(self, prob, *, min_val, max_val):
        ic(min_val, max_val)
        """docstring"""
        self.prob = prob
        self.min_val = min_val
        self.max_val = max_val

    def apply(
        self, encoded_sequence: Tensor, noise_allowed_mask: Optional[Tensor] = None
    ):
        return fragment_noise(
            encoded_sequence,
            self.prob,
            min_val=self.min_val,
            max_val=self.max_val,
            noise_allowed_mask=noise_allowed_mask,
        )


def fragment_noise(
    encoded_sequence: Tensor,
    prob: float,
    min_val,
    max_val,
    noise_allowed_mask: Optional[Tensor] = None,
):
    if not prob or prob <= 0:
        return encoded_sequence

    if noise_allowed_mask is None:
        assert False
        noise_allowed_mask = torch.ones_like(encoded_sequence)
    noise_allowed_mask = noise_allowed_mask.bool()

    # Replace fragments with random ones
    substitution_mask = torch.rand(len(encoded_sequence)).lt(prob)
    random_fragments = torch.randint(
        low=min_val, high=max_val, size=(len(encoded_sequence),)
    )
    # b /data/scratch/haukur/document_translation/fairseq_user_dir/fragment_noise.py:55
    # print(substitution_mask, noise_allowed_mask)
    # breakpoint()
    encoded_sequence[substitution_mask * noise_allowed_mask] = random_fragments[
        substitution_mask * noise_allowed_mask
    ]

    # Remove some fragments entirely.
    keep_mask = (
        torch.rand(len(encoded_sequence)) < 1 - prob
    )  # Keep fragments at these indexes, remove others
    encoded_sequence = encoded_sequence[keep_mask + noise_allowed_mask.logical_not()]
    noise_allowed_mask = noise_allowed_mask[
        keep_mask + noise_allowed_mask.logical_not()
    ]

    # Insert some fragments.
    # We will select some tokens from the embedding and place a random token in front of those.
    # The strategy is:
    #   - Pick locations to insert
    #   - Calculate the size of the new embedded sequence
    #   - Calculate the positions of the old data in the new sequence
    #   - Copy the old data
    #   - Calculate the positions of the new data in the new sequence (i.e. where the old data did not go)
    #   - Generate and copy new data

    # Pick locations to insert
    # insert_mask is True iff we're going to insert in front of the matching token
    insert_mask = torch.rand(len(encoded_sequence)) < prob
    # We're not allowed to insert in front of the no-noise tokens.
    # We make this restriction to prevent breaking up of no-noise spans.
    insert_mask *= noise_allowed_mask

    # Calculate the size of the new embedded sequence
    num_inserts = torch.sum(insert_mask).item()
    new_size = len(encoded_sequence) + num_inserts

    # Calculate the positions of the old data in the new sequence
    # The positions are shifted by the amount of new elements inserted before them, i.e. the cumsum of the insertion mask
    new_indexes_for_old_data = torch.arange(len(encoded_sequence)) + torch.cumsum(
        insert_mask, 0
    )

    # Preallocate a correctly sized tensor to contain the new sequence
    new_enc_seq = encoded_sequence.new_zeros(new_size)
    # Copy the old data into the correct positions
    new_enc_seq[new_indexes_for_old_data] = encoded_sequence

    # Calculate positions for new data
    # The positions are the indexes of True values in the insertion mask, shifted by the number of elements inserted before
    new_indexes_for_inserted_data = torch.arange(len(encoded_sequence))[
        insert_mask
    ] + torch.arange(num_inserts)

    # Generate new data
    # We assume that anything above nspecial in the dict is a legal target. Is this true for all models?
    random_fragments = torch.randint(
        low=min_val, high=max_val, size=(num_inserts,)
    )
    # random_fragments = torch.randint(low=-100, high=-10, size=(num_inserts,))
    # Copy new data
    new_enc_seq[new_indexes_for_inserted_data] = random_fragments

    # TODO: Update noise_allowed_mask OR just ignore it since this is the last dataset in our use case
    # return (new_enc_seq, noise_allowed_mask)
    return new_enc_seq


class FragmentNoiseDataset(BaseWrapperDataset):
    """
    Apply noise to word fragments in a sequence.
    The types of noise applied are:
        random replacement
        fragment dropout
        insert random fragments
    Noise will be applied to individual fragments with probability p.
    Noise will only be applied to fragments where the allow_noise_mask is true.
        This is useful for excluding meta tokens like <bos>, <eos>.
    """

    def __init__(self, args, dataset, src_dict, noise_prob):
        super().__init__(dataset)
        self.src_dict = src_dict
        # Noise prob ætti sennilega að vera í args?
        self.p = noise_prob

    def __getitem__(self, index):
        sample = self.dataset[index]
        encoded_sequence, noise_allowed_mask = sample
        return fragment_noise(
            encoded_sequence,
            self.p,
            min_val=self.src_dict.nspecial,
            max_val=len(self.src_dict) - 1,
            noise_allowed_mask=noise_allowed_mask,
        )
