# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

from collections import OrderedDict

import torch
from fairseq.data import BaseWrapperDataset, NestedDictionaryDataset
from fairseq.data.nested_dictionary_dataset import _unflatten
from torch.utils.data.dataloader import default_collate


class MutexBinaryDataset(BaseWrapperDataset):
    def __init__(self, dataset, default=1, num_mutex_classes=34, skip_n=5, separator=-1):
        super().__init__(dataset)
        self.default = default
        self.num_mutex_classes = num_mutex_classes
        self.skip_n = skip_n
        self.separator = -1

    def __getitem__(self, index):
        seq_labels = self.dataset[index]
        a = []
        b = []
        for v in seq_labels:
            if v < self.skip_n:
                continue
            if b and v < self.skip_n + self.num_mutex_classes:
                a.append(b)
                b = []
            b.append(v)
        a.append(b)

        res = []
        for w in a:
            for t in w:
                res.append(t)
            res.append(self.separator)

        return torch.tensor(res).int()


class NestedDictionaryDatasetFix(NestedDictionaryDataset):
    def collater(self, samples):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate

        Returns:
            dict: a mini-batch suitable for forwarding with a Model
        """
        if len(samples) == 0:
            return {}
        sample = OrderedDict()
        for k, ds in self.defn.items():
            try:
                sample[k] = ds.collater([s[k] for s in samples])
            except (NotImplementedError, AttributeError) as e:  # noqa
                sample[k] = default_collate([s[k] for s in samples])
        return _unflatten(sample)

    def set_epoch(self, epoch):
        for ds in self.defn.values():
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)
