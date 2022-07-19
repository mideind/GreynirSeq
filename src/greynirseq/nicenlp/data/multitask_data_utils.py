# This file is a slightly modified file from fairseq which has the following liccense: 
#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict

import numpy as np

from fairseq.data import BaseWrapperDataset, FairseqDataset, iterators, data_utils


class MultiItr(object):
    def __init__(self, itr, seed=1, epoch=1):
        self.itr = itr
        self.seed = seed
        self.epoch = epoch
        self._counts = [0 for x in itr]

    def __len__(self):
        return sum(len(itr) for itr in self.itr)

    def __iter__(self):
        return self

    def __next__(self):
        sampling_weights = [(len(itr) - count) for count, itr in zip(self._counts, self.itr)]
        if any(w < 0 for w in sampling_weights):
            breakpoint()
        total = sum(sampling_weights)
        banned_idxs = [i for i, v in enumerate(sampling_weights) if v <= 0]
        if total == 0:
            raise StopIteration
            # return next(self.itr[0])  # cause stopiteration
            # sampling_weights = [1 for weight in sampling_weights]
            # total = sum(sampling_weights)
        sampling_weights = [weight/total for weight in sampling_weights]
        with data_utils.numpy_seed(self.seed, self.epoch, sum(self._counts)):
            idx = np.random.choice(np.arange(len(sampling_weights)), p=sampling_weights)
            assert idx not in banned_idxs

        self._counts[idx] += 1
        return next(self.itr[idx])


class MultidatasetEpochBatchIterator(iterators.EpochBatchIterating):
    """A wrapper around multiple epoch batch iterators."""

    def __init__(
        self,
        dataset,
        batch_sampler,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
    ):

        assert isinstance(dataset, OrderedDict)
        assert len(dataset)
        assert isinstance(dataset[next(iter(dataset))], FairseqDataset)
        self.seed = seed

        self.iterators = []

        self.epoch = epoch
        for key, dt in dataset.items():
            epoch_iter = iterators.EpochBatchIterator(
                dataset=dt,
                collate_fn=dt.collater,
                batch_sampler=batch_sampler[key],
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=0,
                epoch=epoch,
            )
            self.iterators.append(epoch_iter)

    def __len__(self):
        return sum(len(itr) for itr in self.iterators)

    def next_epoch_itr(self, shuffle=True, fix_batches_to_gpus=False):
        # `self.epoch += 1` should be handled by underlying `EpochBatchIterator`s.
        return MultiItr(
            [
                itr.next_epoch_itr(
                    shuffle=shuffle, fix_batches_to_gpus=fix_batches_to_gpus
                )
                for itr in self.iterators
            ],
            seed=self.seed,
            epoch=self.epoch,
        )

    def end_of_epoch(self):
        return all(itr.end_of_epoch() for itr in self.iterators)

    @property
    def next_epoch_idx(self):
        """Return the epoch index after *next_epoch_itr* is called."""

        epochs = [itr.next_epoch_idx for itr in self.iterators]
        self.epoch = epochs[0]
        breakpoint()
        assert all(epoch == self.epoch for epoch in epochs)

        return self.epoch

    @property
    def iterations_in_epoch(self):
        return sum(itr.iterations_in_epoch for itr in self.iterators)

    def state_dict(self):
        return {
            "iterators": [it.state_dict() for it in self.iterators],
            "epoch": self.epoch,
        }

    def load_state_dict(self, state_dict):
        self.epoch = state_dict["epoch"]
        for it, d in zip(self.iterators, state_dict["iterators"]):
            it.load_state_dict(d)


class MultitaskDatasetWrapper(BaseWrapperDataset):
    """A wrapper for a multitask dataset."""

    def __init__(self, dataset, sampling_weight=1.0, name=""):
        super().__init__(dataset)
        self.sampling_weight = sampling_weight
        self.name = name

    def collater(self, *args, **kwargs):
        ans = self.dataset.collater(*args, **kwargs)
        if "net_input" in ans:
            ans["net_input"]["dataset_name"] = self.name
        return ans

    def num_tokens(self, *args, **kwargs):
        return self.dataset.num_tokens(*args, **kwargs)

    def ordered_indices(self, *args, **kwargs):
        indices = self.dataset.ordered_indices(*args, **kwargs)
        # Hacky solution for sampling
        size = int(self.sampling_weight * indices.shape[0])

        return indices.take(np.sort(np.random.permutation(indices.shape[0])[:size]))

    def size(self, index: int):
        return self.dataset.size(index)

    @property
    def supports_prefetch(self):
        """Whether this dataset supports prefetching."""
        return getattr(self.dataset, "supports_prefetch", False)

    def prefetch(self, indices):
        return self.dataset.prefetch(indices)

    # def __getitem__(self, index):
    #     return self.dataset[index % len(self.dataset)]
