import torch
from fairseq.utils import make_positions

from greynirseq.nicenlp.tasks.translation_with_glossary import (
    make_positions_with_constraints,
    whole_word_lengths,
    whole_word_sampling,
)


def test_original_make_position():
    pad_idx = 0
    test = torch.tensor(
        [
            [1, 1, 2, 3, 99, 4, 0],
            [5, 6, 99, 7, 8, 0, 0],
        ]
    )
    expected_positions = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 0],
            [1, 2, 3, 4, 5, 0, 0],
        ]
    )
    calculated_positions = make_positions(test, padding_idx=pad_idx)
    assert torch.all(expected_positions.eq(calculated_positions))


def test_original_make_position_with_padding():
    pad_idx = 100
    test = torch.tensor(
        [
            [1, 1, 2, 3, 99, 4, 100],
            [5, 6, 99, 7, 8, 100, 100],
        ]
    )
    expected_positions = torch.tensor(
        [
            [101, 102, 103, 104, 105, 106, 100],
            [101, 102, 103, 104, 105, 100, 100],
        ]
    )
    calculated_positions = make_positions(test, padding_idx=pad_idx)
    assert torch.all(expected_positions.eq(calculated_positions))


def test_make_positions_with_constraints_no_constraint_params():
    pad_idx = 0
    test = torch.tensor(
        [
            [1, 1, 2, 3, 99, 4, 0],
            [5, 6, 99, 7, 8, 0, 0],
        ]
    )
    expected_positions = torch.tensor(
        [
            [1, 2, 3, 4, 5, 6, 0],
            [1, 2, 3, 4, 5, 0, 0],
        ]
    )
    calculated_positions = make_positions_with_constraints(test, padding_idx=pad_idx)
    assert torch.all(expected_positions.eq(calculated_positions))


def test_make_positions_with_constraints_no_constraint_params_with_padding():
    pad_idx = 100
    test = torch.tensor(
        [
            [1, 1, 2, 3, 99, 4, 100],
            [5, 6, 99, 7, 8, 100, 100],
        ]
    )
    expected_positions = torch.tensor(
        [
            [101, 102, 103, 104, 105, 106, 100],
            [101, 102, 103, 104, 105, 100, 100],
        ]
    )
    calculated_positions = make_positions_with_constraints(test, padding_idx=pad_idx)
    assert torch.all(expected_positions.eq(calculated_positions))


def test_make_positions_with_constraints_apply_offset():
    pad_idx = 0
    positional_marker_symbol_idx = 99
    positional_idx_restart_offset = 10
    test = torch.tensor(
        [
            [1, 1, 2, 3, 99, 4, 0],
            [5, 6, 99, 7, 8, 0, 0],
        ]
    )
    expected_positions = torch.tensor(
        [
            [1, 2, 3, 4, 11, 12, 0],
            [1, 2, 11, 12, 13, 0, 0],
        ]
    )
    calculated_positions = make_positions_with_constraints(
        test,
        padding_idx=pad_idx,
        positional_marker_symbol_idx=positional_marker_symbol_idx,
        positional_idx_restart_offset=positional_idx_restart_offset,
    )
    assert torch.all(expected_positions.eq(calculated_positions))


def test_make_positions_with_constraints_apply_shift_and_padding():
    pad_idx = 100
    positional_marker_symbol_idx = 99
    positional_idx_restart_offset = 10
    test = torch.tensor(
        [
            [1, 1, 2, 3, 99, 4, 100],
            [5, 6, 99, 7, 8, 100, 100],
        ]
    )
    expected_positions = torch.tensor(
        [
            [101, 102, 103, 104, 111, 112, 100],
            [101, 102, 111, 112, 113, 100, 100],
        ]
    )
    calculated_positions = make_positions_with_constraints(
        test,
        padding_idx=pad_idx,
        positional_marker_symbol_idx=positional_marker_symbol_idx,
        positional_idx_restart_offset=positional_idx_restart_offset,
    )
    assert torch.all(expected_positions.eq(calculated_positions))


def test_make_positions_with_constraints_apply_shift_and_padding_additional_sent_with_no_constraints():
    pad_idx = 100
    positional_marker_symbol_idx = 99
    positional_idx_restart_offset = 10
    test = torch.tensor(
        [
            [1, 1, 2, 3, 99, 4, 100],
            [5, 6, 99, 7, 8, 100, 100],
            [5, 6, 9, 7, 8, 100, 100],
        ]
    )
    expected_positions = torch.tensor(
        [
            [101, 102, 103, 104, 111, 112, 100],
            [101, 102, 111, 112, 113, 100, 100],
            [101, 102, 103, 104, 105, 100, 100],
        ]
    )
    calculated_positions = make_positions_with_constraints(
        test,
        padding_idx=pad_idx,
        positional_marker_symbol_idx=positional_marker_symbol_idx,
        positional_idx_restart_offset=positional_idx_restart_offset,
    )
    assert torch.all(expected_positions.eq(calculated_positions))


def test_make_positions_with_constraints_apply_shift_and_padding_all_sent_with_no_constraints():
    pad_idx = 100
    positional_marker_symbol_idx = 99
    positional_idx_restart_offset = 10
    test = torch.tensor(
        [
            [1, 1, 2, 3, 5, 4, 100],
            [5, 6, 6, 7, 8, 100, 100],
            [5, 6, 9, 7, 8, 100, 100],
        ]
    )
    expected_positions = torch.tensor(
        [
            [101, 102, 103, 104, 105, 106, 100],
            [101, 102, 103, 104, 105, 100, 100],
            [101, 102, 103, 104, 105, 100, 100],
        ]
    )
    calculated_positions = make_positions_with_constraints(
        test,
        padding_idx=pad_idx,
        positional_marker_symbol_idx=positional_marker_symbol_idx,
        positional_idx_restart_offset=positional_idx_restart_offset,
    )
    assert torch.all(expected_positions.eq(calculated_positions))


def test_masks_lengths():
    # The test sequence does not start with a whole word, so it is not included in the lengths.
    test = [0, 1, 0, 0, 1, 1, 1, 0, 0]
    expected_lengths = [3, 1, 1, 3]
    calculated_lengths = whole_word_lengths(test)
    assert expected_lengths == calculated_lengths


def test_whole_word_target_sampling():
    test = torch.arange(0, 10).long()
    whole_word_masker = {idx: 1 for idx in range(10)}
    # Set some words as partial tokens
    whole_word_masker[0] = 0
    whole_word_masker[5] = 0
    whole_word_masker[9] = 0
    result = whole_word_sampling(
        test, whole_word_masker, seq_sample_ratio=1.0, word_count_to_sample=2, contains_eos=False
    )
    assert len(result) == 2, "There should be two constraints"


def test_whole_word_target_sampling_with_eos():
    test = torch.arange(0, 10).long()  # eos is the last element and is considered a whole word
    whole_word_masker = {idx: 1 for idx in range(10)}
    # Set some words as partial tokens
    whole_word_masker[5] = 0
    whole_word_masker[9] = 0
    result = whole_word_sampling(
        test, whole_word_masker, seq_sample_ratio=1.0, word_count_to_sample=2, contains_eos=True
    )
    assert len(result) == 2, "There should be two constraints"


def test_whole_word_target_sampling_all_partial():
    test = torch.arange(0, 10).long()
    whole_word_masker = {idx: 1 for idx in range(10)}
    # make sure that every other word is a partial token
    whole_word_masker[1] = 0
    whole_word_masker[3] = 0
    whole_word_masker[5] = 0
    whole_word_masker[7] = 0
    whole_word_masker[9] = 0
    result = whole_word_sampling(
        test, whole_word_masker, seq_sample_ratio=1.0, word_count_to_sample=2, contains_eos=False
    )
    assert len(result) == 2, "There should be two constraints"
    assert all(list(len(constraint) == 2 for constraint in result)), "All constraints should have two elements"


def test_whole_word_target_sampling_count_larger_than_length():
    test = torch.arange(0, 10).long()
    whole_word_masker = {idx: 1 for idx in range(10)}
    # Set some words as partial tokens
    whole_word_masker[0] = 0
    whole_word_masker[5] = 0
    whole_word_masker[9] = 0
    result = whole_word_sampling(
        test, whole_word_masker, seq_sample_ratio=1.0, word_count_to_sample=10, contains_eos=False
    )
    assert len(result) == 7, "There should be seven constraints"


def test_whole_word_target_sampling_count_negative():
    test = torch.arange(0, 10).long()
    whole_word_masker = {idx: 1 for idx in range(10)}
    # Set some words as partial tokens
    whole_word_masker[0] = 0
    whole_word_masker[5] = 0
    whole_word_masker[9] = 0
    result = whole_word_sampling(
        test, whole_word_masker, seq_sample_ratio=1.0, word_count_to_sample=-10, contains_eos=False
    )
    assert len(result) == 0, "There should be no constraints"
