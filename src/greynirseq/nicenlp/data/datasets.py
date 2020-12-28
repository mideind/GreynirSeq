from collections import OrderedDict
from functools import lru_cache

from fairseq.data import BaseWrapperDataset, NestedDictionaryDataset, LRUCacheDataset
from fairseq.data.nested_dictionary_dataset import _unflatten

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import default_collate
from fairseq.data import data_utils


class LabelledSpanDataset(BaseWrapperDataset):
    """
    Read in a dataset of [word_start_i, word_end_i, word_label_i, ...]
    """

    def __init__(self, dataset, return_spans=False):
        super().__init__(dataset)
        self.return_spans = return_spans

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        item = self.dataset[index]
        assert len(item) % 3 == 0, "Illegal number of labelled_span elements"
        numel = len(item) // 3
        if self.return_spans:
            return item.reshape(numel, 3)[:, :2].reshape(-1)
        return item.reshape(numel, 3)[:, 2]


class DynamicLabelledSpanDataset(BaseWrapperDataset):
    """
    Same as LabelledSpanDataset, except the binarization of each is sampled random at each epoch
    """

    @classmethod
    def make_both(cls, dataset, label_dictionary, rebinarize_fn=None, seed=1):
        dataset = LRUCacheDataset(dataset)
        return (
            DynamicLabelledSpanDataset(
                dataset,
                label_dictionary,
                return_spans=True,
                rebinarize_fn=rebinarize_fn,
                seed=seed,
            ),
            DynamicLabelledSpanDataset(
                dataset,
                label_dictionary,
                return_spans=False,
                rebinarize_fn=rebinarize_fn,
                seed=seed,
            ),
        )

    def __init__(
        self, dataset, label_dictionary, rebinarize_fn=None, return_spans=None, seed=1
    ):
        assert rebinarize_fn is not None, "Rebinarization function must be provided"
        assert isinstance(return_spans, bool), "Must provide boolean for return_spans"
        super().__init__(dataset)
        self.rebinarize_fn = rebinarize_fn
        self.label_dictionary = label_dictionary
        self.return_spans = return_spans
        self.epoch = 0
        self.seed = seed

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch, index):
            item = self.dataset[index]
            assert len(item) % 3 == 0, "Illegal number of labelled_span elements"
            numel = len(item) // 3
            seq_spans = item.reshape(numel, 3)[:, :2].tolist()
            seq_labels = item.reshape(numel, 3)[:, 2].tolist()
            seq_labels = [self.label_dictionary.symbols[l_idx] for l_idx in seq_labels]

            new_seq_spans, seq_labels = self.rebinarize_fn(seq_spans, seq_labels)
            new_seq_labels = [
                self.label_dictionary.index(label) for label in seq_labels
            ]
            if self.return_spans:
                return torch.tensor(new_seq_spans).view(-1)
            return torch.tensor(new_seq_labels)


class WordSpanDataset(BaseWrapperDataset):
    def __init__(self, dataset, dictionary, is_word_initial):
        super().__init__(dataset)
        self.is_word_initial = is_word_initial
        self.dictionary = dictionary

    @lru_cache(maxsize=8)
    def __getitem__(self, index):
        item = self.dataset[index]
        offset = 1 if item[0] == self.dictionary.bos() else 0
        seq_end = len(item) - 1 if item[-1] == self.dictionary.eos() else len(item)
        idxs = [
            self.is_word_initial.get(int(v), 0)
            for v in item[offset:seq_end]  # ignore bos and eos
        ]
        idxs[0] = 1
        starts = [
            idx + offset
            for (idx, is_word_initial) in enumerate(idxs)
            if is_word_initial
        ]
        ends = starts[1:]
        ends.append(len(idxs) + offset)
        spans = list(sum(zip(starts, ends), ()))
        return torch.Tensor(spans).long()


class SparseProductSpanDataset(BaseWrapperDataset):
    def __init__(self, span_dataset, end_is_fence_post=True):
        """Product span, a subset (starts x ends) where span direction is forward."""
        super().__init__(span_dataset)
        # whether we need to subtract 1 for position index
        self.end_is_fence_post = end_is_fence_post

    def __getitem__(self, index):
        seq_spans = self.dataset[index].reshape(-1, 2)
        span_start = seq_spans[:, 0]
        offset = 0
        if self.end_is_fence_post:
            offset = 1
        span_end = seq_spans[:, 1]
        nwords = len(span_start)
        out_spans = []
        for ii in range(nwords):
            for jj in range(nwords):
                if ii > jj:
                    continue
                out_spans.extend((span_start[ii], span_end[jj]))
        return torch.tensor(out_spans).long()


class ProductSpanDataset(BaseWrapperDataset):
    """Take in span starts and span ends as contiguous 1d tensor.
    Return product spans, (starts x ends), as contiguous 1d tensor."""

    def __init__(self, span_dataset, end_is_fence_post=True):
        super().__init__(span_dataset)
        # whether we need to subtract 1 for position index
        self.end_is_fence_post = end_is_fence_post

    def __getitem__(self, index):
        seq_spans = self.dataset[index].reshape(-1, 2)
        span_start = seq_spans[:, 0]
        offset = 0
        if self.end_is_fence_post:
            offset = 1
        span_end = seq_spans[:, 1]
        nwords = len(span_start)
        tiled_starts = span_start.unsqueeze(0).repeat(nwords, 1).permute(1, 0)
        tiled_ends = span_end.unsqueeze(0).repeat(nwords, 1)
        all_contig = span_start.new_zeros(nwords, nwords, 2)
        all_contig[:, :, 0] = tiled_starts
        all_contig[:, :, 1] = tiled_ends
        # TODO: maybe make lower triangle (and middle) all 0?
        return all_contig.reshape(-1)


def split_tensor_on(tensor, sep_value):
    assert len(tensor.shape) == 1
    start = 0
    numel = tensor.shape[0]
    items = []
    for idx, val in enumerate(tensor):
        val_ = val.item()
        if val == sep_value:
            end = idx
            items.append(tensor[start:end])
            start = end + 1
            continue
        end = idx
    if start < numel:
        items.append(tensor[start:])
    return items


def collate_2d(values, pad_idx, left_pad=False):
    """Copy list of 2d tensors into padded 3d tensor"""
    if left_pad:
        raise NotImplementedError("Left pad 2d missing")
    # each item is 2d
    bsz = len(values)
    max_rows = max(item.shape[0] for item in values)
    max_width = max(item.shape[1] for item in values)

    new_tensor = values[0].new(bsz, max_rows, max_width).fill_(pad_idx)
    for (idx, item) in enumerate(values):
        rows, cols = item.shape
        new_tensor[idx, :rows, :cols].copy_(item)
    return new_tensor


class RightPad2dDataset(BaseWrapperDataset):
    def __init__(self, dataset, pad_idx):
        super().__init__(dataset)
        self.pad_idx = pad_idx

    def collater(self, samples):
        return collate_2d(samples, self.pad_idx, left_pad=False)


class POSDataset(BaseWrapperDataset):
    @classmethod
    def make_both(cls, dataset, src_dictionary, label_dictionary):
        dataset = LRUCacheDataset(dataset)
        return (
            POSDataset(
                dataset, src_dictionary, label_dictionary, return_categories=True),
            POSDataset(
                dataset, src_dictionary, label_dictionary, return_categories=False
            ),
        )

    def __init__(
            self, dataset, src_dictionary, label_dictionary, return_categories=True
    ):
        super().__init__(dataset)
        self.label_dict = label_dictionary
        self.src_dict = src_dictionary
        self.return_categories = return_categories
        self.start_offset = 1 if dataset[0][0] == self.src_dict.bos() else 0

    def __getitem__(self, index):
        item = self.dataset[index]
        word_items = split_tensor_on(
            item[self.start_offset:-1],
            self.label_dict.sep()
        )
        assert all(subseq.numel() > 0 for subseq in word_items)
        assert len(word_items) == (item.eq(self.label_dict.sep()).sum() + 1)
        num_words = len(word_items)
        label_shift = self.label_dict.nspecial
        num_labels = len(self.label_dict) - label_shift

        cats = item.new_zeros(num_words)
        attrs = item.new_zeros(num_words, num_labels)

        for word_idx, pos_items in enumerate(word_items):
            cat, *word_attrs = pos_items
            cats[word_idx] = cat
            for attr_lbl_idx in word_attrs:
                attr_vec_index = attr_lbl_idx - label_shift
                attrs[word_idx, attr_vec_index] = 1

        if self.return_categories:
            return cats
        return attrs


class WordEndMaskDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        dictionary,
        is_word_initial,
        bos_value=1,
        eos_value=0,
    ):
        super().__init__(dataset)
        self.is_word_initial = is_word_initial
        self.dictionary = dictionary
        self.start_offset = 1 if dataset[0][0] == self.dictionary.bos() else 0
        self.bos_value = bos_value
        self.eos_value = eos_value

    def __getitem__(self, index):
        item = self.dataset[index]
        mask = torch.tensor([self.is_word_initial.get(int(v), 1) for v in item])
        mask[self.start_offset] = 1  # temporary hack due to incorrect preprocessing (missing prepend_space)
        mask[:self.start_offset] = self.bos_value
        mask[-1] = self.eos_value
        return mask


class GroupMaskDataset(BaseWrapperDataset):
    def __init__(self, dataset, group_masks):
        super().__init__(dataset)
        self.group_masks = group_masks

    def __getitem__(self, index):
        item = self.dataset[index]
        assert all(lbl_idx in self.group_masks for lbl_idx in item.tolist())
        masks = torch.stack([self.group_masks[lbl_idx] for lbl_idx in item.tolist()])
        return masks


class NumWordsDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset,
        dictionary,
        is_word_initial,
    ):
        super().__init__(dataset)
        self.is_word_initial = is_word_initial
        self.dictionary = dictionary
        self.start_offset = 1 if dataset[0][0] == self.dictionary.bos() else 0

    def __getitem__(self, index):
        word_starts = [
            self.is_word_initial.get(int(v), 1)
            for v in self.dataset[index][self.start_offset:-1]  # ignore bos and eos
        ]
        word_starts[self.start_offset] = 1  # temporary hack due to incorrect preprocessing (missing prepend_space)
        return torch.tensor(sum(word_starts)).long()


class NumSpanDataset(BaseWrapperDataset):
    def __init__(self, dataset):
        super().__init__(dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        assert len(item) % 2 == 0, "Illegal number of span elements"
        numel = len(item) // 2
        return torch.tensor(numel).long()


class LossMaskDataset(BaseWrapperDataset):
    """Dataset for masking out specific label groups (mutually exclusive classes)"""

    def __init__(self, dataset, label_idx_to_loss_mask):
        super().__init__(dataset)
        self.label_idx_to_loss_mask = label_idx_to_loss_mask

    def __getitem__(self, index):
        item = self.dataset[index]
        return torch.stack([self.label_idx_to_loss_mask[idx] for idx in item]).reshape(
            -1
        )


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
            except (NotImplementedError, AttributeError) as e:
                sample[k] = default_collate([s[k] for s in samples])
        return _unflatten(sample)

    def set_epoch(self, epoch):
        for ds in self.defn.values():
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)


class NestedDictionaryDatasetFix2(NestedDictionaryDatasetFix):
    def collater(self, samples):
        """Temp fix for inference"""
        if len(samples) == 0:
            return {}
        sample = OrderedDict()
        for k, ds in samples[0].items():
            try:
                sample[k] = ds.collater([s[k] for s in samples])
            except (NotImplementedError, AttributeError) as e:
                sample[k] = default_collate([s[k] for s in samples])
        return _unflatten(sample)

    def set_epoch(self, epoch):
        for ds in self.defn.values():
            if hasattr(ds, "set_epoch"):
                ds.set_epoch(epoch)
