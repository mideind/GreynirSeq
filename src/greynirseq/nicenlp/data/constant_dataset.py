# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

from typing import Any

from fairseq.data import FairseqDataset

import torch


class ConstantDataset(FairseqDataset):
    def __init__(self, constant: Any, collapse_collate=True):
        super().__init__()
        self.constant = constant
        self.collapse = collapse_collate

    def __getitem__(self, index):
        return self.constant

    def __len__(self):
        return 0

    def collater(self, samples):
        if self.collapse:
            return torch.tensor(samples[0])
        return torch.tensor(samples)
