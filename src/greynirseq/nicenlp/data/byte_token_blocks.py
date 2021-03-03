from typing import Optional
from dataclasses import asdict

import numpy as np
import torch
from fairseq.data import FairseqDataset, plasma_utils

from torch.utils.data import Dataset

from greynirseq.nicenlp.byte_sequence import ByteSequence


class ByteTokenBlockDataset(FairseqDataset):
    """Break a Dataset of tokens into blocks.

    Args:
        dataset (~torch.utils.data.Dataset): dataset to break into blocks
        sizes (List[int]): sentence lengths (required for 'complete' and 'eos')
        block_size (int): maximum block size (ignored in 'eos' break mode)
        break_mode (str, optional): Mode used for breaking tokens. Values can
            be one of:
            - 'none': break tokens into equally sized blocks (up to block_size)
            - 'complete': break tokens into blocks (up to block_size) such that
                blocks contains complete sentences, although block_size may be
                exceeded if some sentences exceed block_size
            - 'complete_doc': similar to 'complete' mode, but do not
                cross document boundaries
            - 'eos': each block contains one sentence (block_size is ignored)
        include_targets (bool, optional): return next tokens as targets
            (default: False).
        document_sep_len (int, optional): document separator size (required for
            'complete_doc' break mode). Typically 1 if the sentences have eos
            and 0 otherwise.
    """

    def __init__(
        self,
        byteseq_dataset: Dataset,
        sizes,
        block_size: int,
        pad: int,
        eos: int,
        break_mode: Optional[str]=None,
        document_sep_len: int=1,
    ):
        try:
            from fairseq.data.token_block_utils_fast import (
                _get_slice_indices_fast,
                _get_block_to_dataset_index_fast,
            )
        except ImportError:
            raise ImportError(
                "Please build Cython components with: `pip install --editable .` "
                "or `python setup.py build_ext --inplace`"
            )

        super().__init__()
        self.dataset = byteseq_dataset
        self.pad = pad
        self.eos = eos

        assert len(byteseq_dataset) == len(sizes)
        assert len(byteseq_dataset) > 0

        if isinstance(sizes, list):
            sizes = np.array(sizes, dtype=np.int64)
        else:
            if torch.is_tensor(sizes):
                sizes = sizes.numpy()
            sizes = sizes.astype(np.int64)

        break_mode = break_mode if break_mode is not None else "none"

        # For "eos" break-mode, block_size is not required parameters.
        if break_mode == "eos" and block_size is None:
            block_size = 0

        slice_indices = _get_slice_indices_fast(
            sizes, str(break_mode), block_size, document_sep_len
        )
        self._sizes = slice_indices[:, 1] - slice_indices[:, 0]

        # build index mapping block indices to the underlying dataset indices
        if break_mode == "eos":
            # much faster version for eos break mode
            block_to_dataset_index = np.stack(
                [
                    np.arange(len(sizes)),  # starting index in dataset
                    np.zeros(
                        len(sizes), dtype=np.long
                    ),  # starting offset within starting index
                    np.arange(len(sizes)),  # ending index in dataset
                ],
                1,
            )
        else:
            block_to_dataset_index = _get_block_to_dataset_index_fast(
                sizes,
                slice_indices,
            )
        # for info on why plasmarray:
        #   https://github.com/pytorch/fairseq/commit/439ead5a7738bc5080d1d4643ae4bf6dfc78b8ca
        self._slice_indices = plasma_utils.PlasmaArray(slice_indices)
        self._sizes = plasma_utils.PlasmaArray(self._sizes)
        self._block_to_dataset_index = plasma_utils.PlasmaArray(block_to_dataset_index)

    @property
    def slice_indices(self):
        return self._slice_indices.array

    @property
    def sizes(self):
        return self._sizes.array

    @property
    def block_to_dataset_index(self):
        return self._block_to_dataset_index.array

    def attr(self, attr: str, index: int):
        start_ds_idx, _, _ = self.block_to_dataset_index[index]
        return self.dataset.attr(attr, start_ds_idx)

    def __getitem__(self, index):
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]
        # we wouldnt know the corresponding offsets for other attributes than the main one
        assert start_offset == 0, "start_offset != 0 not supported for ByteSequence blocks"

        items = [self.dataset[idx] for idx in range(start_ds_idx, end_ds_idx + 1)]
        return ByteSequence.cat(items)

    def getitem_attr(self, index, attr):
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]
        assert start_offset == 0, "start_offset != 0 not supported for ByteSequence blocks"
        ic(start_ds_idx, start_offset, end_ds_idx)

        buffer = torch.cat(
            [getattr(self.dataset[idx], attr) for idx in range(start_ds_idx, end_ds_idx + 1)]
        )

        return buffer

    def get_mapped(self, index, transform):
        start_ds_idx, start_offset, end_ds_idx = self.block_to_dataset_index[index]
        # we wouldnt know the corresponding offsets for other attributes than the main one
        assert start_offset == 0, "start_offset != 0 not supported for ByteSequence blocks"
        ic(start_ds_idx, start_offset, end_ds_idx)

        buffer = torch.cat(
            # [getattr(attr, self.dataset[idx]) for idx in range(start_ds_idx, end_ds_idx + 1)]
            [transform(self.dataset[idx]) for idx in range(start_ds_idx, end_ds_idx + 1)]
            # [self.dataset[idx].bpe_ids for idx in range(start_ds_idx, end_ds_idx + 1)]
        )

        return buffer

    def __len__(self):
        return len(self.slice_indices)

    @property
    def supports_prefetch(self):
        return self.dataset.supports_prefetch

    def prefetch(self, indices):
        if self.dataset.supports_prefetch:
            self.dataset.prefetch(
                {
                    ds_idx
                    for index in indices
                    for start_ds_idx, _, end_ds_idx in [self.block_to_dataset_index[index]]
                    for ds_idx in range(start_ds_idx, end_ds_idx + 1)
                }
            )
