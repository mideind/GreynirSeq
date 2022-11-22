# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import string
from typing import List

import torch
from fairseq.data import BaseWrapperDataset

from greynirseq.nicenlp.data.noiser import Noiser

KEEP_CHARS = "123456789()[]"
# TODO: add other languages
_INSERTABLE_LOWER_ORDINALS = [ord(i) for i in (string.ascii_lowercase + "þæöð´'°.,'")]
_INSERTABLE_UPPER_ORDINALS = [ord(i.upper()) for i in (string.ascii_lowercase + "þæöð´'°.,'")]
_ALL_INSERTABLE_ORDINALS = _INSERTABLE_LOWER_ORDINALS + _INSERTABLE_UPPER_ORDINALS


class CharacterNoiser(Noiser):
    def __init__(
        self,
        swap_prob: float = 0.01,
        delete_prob: float = 0.01,
        insert_prob: float = 0.01,
        duplicate_prob: float = 0.01,
        case_prob: float = 0.01,
        substitution_prob: float = 0.01,
        seq_lower_prob: float = 0.01,
        seq_upper_prob: float = 0.00,
    ):
        self.swap_prob = swap_prob
        self.delete_prob = delete_prob
        self.insert_prob = insert_prob
        self.duplicate_prob = duplicate_prob
        self.case_prob = case_prob
        self.substitution_prob = substitution_prob
        self.seq_lower_prob = seq_lower_prob
        self.seq_upper_prob = seq_upper_prob

    def apply(self, string: str):
        return apply_character_noise(
            string,
            swap_prob=self.swap_prob,
            delete_prob=self.delete_prob,
            insert_prob=self.insert_prob,
            duplicate_prob=self.duplicate_prob,
            case_prob=self.case_prob,
            substitution_prob=self.substitution_prob,
            seq_lower_prob=self.seq_lower_prob,
            seq_upper_prob=self.seq_upper_prob,
        )


def apply_character_noise(
    string: str,
    *,
    swap_prob: float = 0.01,
    delete_prob: float = 0.01,
    insert_prob: float = 0.01,
    duplicate_prob: float = 0.01,
    case_prob: float = 0.01,
    substitution_prob: float = 0.01,
    seq_lower_prob: float = 0.01,
    seq_upper_prob: float = 0.00,
    insertion_ordinals: List[int] = _ALL_INSERTABLE_ORDINALS,
):
    assert all(p >= 0 for p in [swap_prob, delete_prob, insert_prob, duplicate_prob, case_prob, substitution_prob])
    if not any(p > 0 for p in [swap_prob, delete_prob, insert_prob, duplicate_prob, case_prob, substitution_prob]):
        return string
    assert (
        len(_ALL_INSERTABLE_ORDINALS) % 2 == 0
    ), "We expect the first half of insertion_ordinals to lower case and the latter part to be the same but upper case"

    # XXX TODO: do not destroy: numbers
    # XXX TODO: implement preservation behavior

    seq_case_was_shifted = False
    if seq_lower_prob > 0:
        if torch.rand(1) < seq_lower_prob:
            string = string.lower()
            seq_case_was_shifted = True

    if seq_upper_prob > 0 and not seq_case_was_shifted:
        if torch.rand(1) < seq_upper_prob:
            string = string.upper()
            seq_case_was_shifted = True

    # Shift case for some characters
    if case_prob > 0:
        # TODO XXX: add noise_allowed
        flip_case_mask = torch.rand(len(string)).lt(case_prob)
        string = "".join(
            c.upper() if c.islower() and should_flip else (c.lower() if c.isupper() and should_flip else c)
            for c, should_flip in zip(string, flip_case_mask)
        )

    # TODO: we should probably not noise numbers
    noise_allowed_mask = torch.tensor([c in KEEP_CHARS for c in string], dtype=torch.bool)

    sequence = torch.tensor([ord(c) for c in string])
    insertion_pool_size = len(insertion_ordinals) // 2

    # Duplicate some fragments
    if duplicate_prob > 0:
        old_sequence = sequence
        will_be_duplicated = torch.rand(len(old_sequence)) < duplicate_prob  # only duplicate some fragments
        will_be_duplicated *= noise_allowed_mask
        num_dupes_per_item = (
            torch.empty_like(old_sequence).geometric_(0.5).clamp(1, 10)
        )  # roll number of duplicates seperately
        num_dupes_per_item[will_be_duplicated.logical_not()] = 1
        sequence = torch.repeat_interleave(old_sequence, num_dupes_per_item)
        noise_allowed_mask = torch.repeat_interleave(
            noise_allowed_mask, num_dupes_per_item
        )  # since we modified lengths
        string = "".join(chr(int(o)) for o in sequence)

    # Replace fragments with random ones
    if substitution_prob > 0:
        substitution_mask = torch.rand(len(sequence)).lt(substitution_prob)
        # sample indices into ordinals first
        shift_upper = torch.tensor([c.isupper() for c in string]).long() * insertion_pool_size
        random_fragments = shift_upper + torch.randint(low=0, high=insertion_pool_size, size=(len(sequence),))
        random_fragments = torch.tensor([insertion_ordinals[i] for i in random_fragments], dtype=torch.long)
        sequence[substitution_mask * noise_allowed_mask] = random_fragments[substitution_mask * noise_allowed_mask]

    # Delete some fragments
    if delete_prob > 0:
        keep_mask = torch.rand(len(sequence)) < 1 - delete_prob  # Keep fragments at these indexes, remove others
        sequence = sequence[keep_mask + noise_allowed_mask.logical_not()]
        noise_allowed_mask = noise_allowed_mask[keep_mask + noise_allowed_mask.logical_not()]

    # Swap some fragments (with max shuffle dist 1)
    if swap_prob:
        # TODO XXX: how should noise_allowed_mask be handled here?
        EPSILON = 1e-1
        prenoise_positions = torch.arange(len(sequence)).float()
        # we use rand since max shuffle_dist is 1
        positional_noise = (torch.rand(len(sequence)) < swap_prob).float() * (1 + EPSILON)
        permutation = (prenoise_positions + positional_noise).argsort()
        # retrieve elements after position noising
        sequence = sequence[permutation]

    #######################
    # Insert some fragments.
    # We will select some tokens from the embedding and place a random token in front of those.
    # The strategy is:
    #   - Pick locations to insert
    #   - Calculate the size of the new embedded sequence
    #   - Calculate the positions of the old data in the new sequence
    #   - Copy the old data
    #   - Calculate the positions of the new data in the new sequence (i.e. where the old data did not go)
    #   - Generate and copy new data
    if insert_prob:
        # Pick locations to insert
        # insert_mask is True iff we're going to insert in front of the matching token
        old_sequence = sequence
        insert_mask = torch.rand(len(old_sequence)) < insert_prob
        # We're not allowed to insert in front of the no-noise tokens.
        # We make this restriction to prevent breaking up of no-noise spans.
        insert_mask *= noise_allowed_mask

        shift_upper = (
            torch.tensor([c.isupper() for c in "".join(chr(int(o)) for o in old_sequence)]).long() * insertion_pool_size
        )
        shift_upper = shift_upper[insert_mask]

        # Calculate the size of the new embedded sequence
        num_inserts = int(torch.sum(insert_mask).item())
        new_size = len(old_sequence) + num_inserts

        # Calculate the positions of the old data in the new sequence
        # The positions are shifted by the amount of new elements inserted before them, i.e. the cumsum of the insertion mask
        new_indexes_for_old_data = torch.arange(len(old_sequence)) + torch.cumsum(insert_mask, 0)

        # Preallocate a correctly sized tensor to contain the new sequence
        sequence = sequence.new_zeros(new_size)
        # Copy the old data into the correct positions
        sequence[new_indexes_for_old_data] = old_sequence

        # Calculate positions for new data
        # The positions are the indexes of True values in the insertion mask, shifted by the number of elements inserted before
        new_indexes_for_inserted_data = torch.arange(len(old_sequence))[insert_mask] + torch.arange(num_inserts)

        # Generate new data
        # We assume that anything above nspecial in the dict is a legal target. Is this true for all models?
        # sample indices into ordinals first
        random_fragments = shift_upper + torch.randint(low=0, high=insertion_pool_size, size=[num_inserts])
        random_fragments = torch.tensor([insertion_ordinals[i] for i in random_fragments], dtype=torch.long)
        # Copy new data
        sequence[new_indexes_for_inserted_data] = random_fragments  # type: ignore

    output_string = "".join(chr(int(o)) for o in sequence)
    return output_string


class CharacterNoiseDataset(BaseWrapperDataset):
    def __init__(self, args, dataset):
        super().__init__(dataset)
        self.swap_prob = args.char_swap_prob
        self.delete_prob = args.char_delete_prob
        self.insert_prob = args.char_insert_prob
        self.duplicate_prob = args.char_duplicate_prob
        self.case_prob = args.char_case_prob
        self.substitution_prob = args.char_substitution_prob
        self.seq_lower_prob = args.seq_lower_prob
        self.seq_upper_prob = args.seq_upper_prob

    def __getitem__(self, index):
        string = self.dataset[index]
        return apply_character_noise(
            string,
            swap_prob=self.swap_prob,
            delete_prob=self.delete_prob,
            insert_prob=self.insert_prob,
            duplicate_prob=self.duplicate_prob,
            case_prob=self.case_prob,
            substitution_prob=self.substitution_prob,
            seq_lower_prob=self.seq_lower_prob,
            seq_upper_prob=self.seq_upper_prob,
        )
