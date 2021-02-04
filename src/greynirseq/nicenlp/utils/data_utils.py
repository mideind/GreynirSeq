import torch


def lengths_to_offsets(lengths: torch.Tensor):
    offsets = lengths.cumsum(dim=0).roll(1, dims=[0])  # shift right
    offsets[0] = 0  # first item is always at 0 offset
    return offsets


def lengths_to_begin_mask(lengths: torch.Tensor, length=None):
    if length is None:
        length = lengths.sum()
    offsets = lengths.cumsum(dim=0).roll(1, dims=[0])  # shift right
    offsets[0] = 0  # first item is always at 0 offset
    mask = torch.zeros(length, dtype=torch.bool)
    mask[offsets] = 1
    return mask
