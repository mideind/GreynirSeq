import os
from functools import lru_cache

import numpy as np
import torch

from fairseq.data import FairseqDataset


class MmappedTextDataset(FairseqDataset):
    """Takes a text file as input, indexes it at instantiation"""

    def __init__(self, path):
        super().__init__()
        self.path = path
        assert self.exists(path), "Could not find dataset: {}".format(str(path))
        self.offsets = None
        self._sizes = None
        self.memmap_buffer = None
        self.encoded_lines = []
        self._len = None

    def read_data(self, path):
        self.memmap_buffer = np.memmap(path, mode='r', order='C')
        self.offsets = [0]
        self._sizes = []
        fp = self.memmap_buffer._mmap
        fp.seek(0)
        while True:
            line = fp.readline()
            offset = fp.tell()
            if not line:
                break
            self.offsets.append(offset)
            self._sizes.append(len(line))
        fp.seek(0)
        self._len = len(self._sizes)
        self._sizes = torch.tensor(self.sizes)

    def check_index(self, i):
        if i < 0 or i >= self._len:
            raise IndexError('index out of range')

    def __del__(self):
        if self.memmap_buffer is not None:
            del self.memmap_buffer
            self.memmap_buffer = None

    @lru_cache(maxsize=8)
    def __getitem__(self, idx):
        if self.memmap_buffer is None:
            self.read_data(self.path)
        self.check_index(idx)

        byte_range = self.memmap_buffer[self.offsets[idx] : self.offsets[idx + 1]]
        string = byte_range.tostring().decode("utf8").strip("\n")
        return string

    def __len__(self):
        if self.memmap_buffer is None:
            self.read_data(self.path)
        return self._len

    def size(self, index):
        return self.sizes[index]

    # def ordered_indices(self):
    #     return self.sizes.sort()[1]

    @property
    def sizes(self):
        assert self._sizes is not None, "Must load data first"
        return self._sizes

    @staticmethod
    def exists(path):
        return os.path.exists(path)

    @property
    def supports_prefetch(self):
        return False


