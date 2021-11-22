import torch
from fairseq.utils import make_positions

from greynirseq.nicenlp.tasks.translation_with_glossary import make_positions_with_constraints


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

def test_make_positions_with_constraints_apply_shift():
    pad_idx = 0
    shift_idx = 99
    shift_amount = 10
    test = torch.tensor(
        [
            [1, 1, 2, 3, 99, 4, 0],
            [5, 6, 99, 7, 8, 0, 0],
        ]
    )
    expected_positions = torch.tensor(
        [
            [1, 2, 3, 4, 15, 16, 0],
            [1, 2, 13, 14, 15, 0, 0],
        ]
    )
    calculated_positions = make_positions_with_constraints(test, padding_idx=pad_idx, shift_from_symbol=shift_idx, shift_amount=shift_amount)
    assert torch.all(expected_positions.eq(calculated_positions))

def test_make_positions_with_constraints_apply_shift_and_padding():
    pad_idx = 100
    shift_idx = 99
    shift_amount = 10
    test = torch.tensor(
        [
            [1, 1, 2, 3, 99, 4, 100],
            [5, 6, 99, 7, 8, 100, 100],
        ]
    )
    expected_positions = torch.tensor(
        [
            [101, 102, 103, 104, 115, 116, 100],
            [101, 102, 113, 114, 115, 100, 100],
        ]
    )
    calculated_positions = make_positions_with_constraints(test, padding_idx=pad_idx, shift_from_symbol=shift_idx, shift_amount=shift_amount)
    assert torch.all(expected_positions.eq(calculated_positions))

def test_make_positions_with_constraints_apply_shift_and_padding_additional_sent_with_no_constraints():
    pad_idx = 100
    shift_idx = 99
    shift_amount = 10
    test = torch.tensor(
        [
            [1, 1, 2, 3, 99, 4, 100],
            [5, 6, 99, 7, 8, 100, 100],
            [5, 6, 9, 7, 8, 100, 100],
        ]
    )
    expected_positions = torch.tensor(
        [
            [101, 102, 103, 104, 115, 116, 100],
            [101, 102, 113, 114, 115, 100, 100],
            [101, 102, 103, 104, 105, 100, 100],
        ]
    )
    calculated_positions = make_positions_with_constraints(test, padding_idx=pad_idx, shift_from_symbol=shift_idx, shift_amount=shift_amount)
    assert torch.all(expected_positions.eq(calculated_positions))
