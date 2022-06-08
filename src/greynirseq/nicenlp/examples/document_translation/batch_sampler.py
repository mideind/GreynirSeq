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
    batch = []
    ntokens = 0
    batches = []
    for index, length in zip(indices, lengths):
        assert length < max_tokens
        batch_exceeds_seqs = max_sequences is not None and len(batch) >= max_sequences
        batch_exceeds_tokens = max_tokens is not None and ntokens + length >= max_tokens
        if not (batch_exceeds_seqs or batch_exceeds_tokens):
            batch.append(index)
            ntokens += length
        elif batch:
            batches.append(batch)
            batch = [index]
            ntokens = length
        else:
            raise NotImplementedError
    if batch and not drop_last_batch:
        batches.append(batch)
    if shuffle:
        np.random.shuffle(batches)
    return batches
