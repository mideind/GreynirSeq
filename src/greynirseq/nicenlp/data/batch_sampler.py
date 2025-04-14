# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import numpy as np


def batch_by_size(
    indices,
    lengths,
    max_tokens=None,
    max_sequences=None,
    drop_last_batch: bool = False,
    shuffle=True,
):
    assert max_tokens is not None or max_sequences is not None
    assert max_tokens is not None
    batch = []
    batches = []
    olengths = []
    batch_width = 0
    for index, length in zip(indices, lengths):
        assert length <= max_tokens
        batch_within_seqs = max_sequences is not None and len(batch) <= max_sequences
        new_batch_width = max(batch_width, length)
        batch_within_tokens = ((len(batch) + 1) * new_batch_width) <= max_tokens

        if batch_within_seqs and batch_within_tokens:
            batch.append(index)
            olengths.append(length)
            batch_width = new_batch_width
        elif batch:
            # yield batch
            batches.append(batch)
            batch = [index]
            olengths = [length]
            batch_width = length
        else:
            raise NotImplementedError
    if batch and not drop_last_batch:
        # yield batch
        batches.append(batch)
    if shuffle:
        np.random.shuffle(batches)
    return batches
