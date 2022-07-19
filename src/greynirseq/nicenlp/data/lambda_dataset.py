# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

from fairseq.data import BaseWrapperDataset
from torch.utils.data import Dataset


class LambdaDataset(BaseWrapperDataset):
    def __init__(self, dataset: Dataset, lambda_fn):
        super().__init__(dataset)
        self.lambda_fn = lambda_fn

    def __getitem__(self, index: int):
        return self.lambda_fn(self.dataset[index])

    def set_epoch(self, epoch):
        if hasattr(self.dataset, "set_epoch"):
            self.dataset.set_epoch(epoch)

