# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import torch
from fairseq.data import BaseWrapperDataset


class LookupDataset(BaseWrapperDataset):
    def __init__(self, dataset, lookup_dictionary, default=1, label_first=True):
        super().__init__(dataset)
        self.lookup_dictionary = lookup_dictionary
        self.default = default
        self.label_first = label_first

    def __getitem__(self, index):
        indexes = [self.lookup_dictionary.get(int(v), self.default) for v in self.dataset[index]]  # [:-1]
        if self.label_first:
            indexes[0] = self.default
        return torch.tensor(indexes).int()
