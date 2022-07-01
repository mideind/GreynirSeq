from typing import List, Any, Optional, Callable, Union

import numpy as np
import torch
from fairseq.data import BaseWrapperDataset, Dictionary, data_utils, LanguagePairDataset
import datasets as hf_datasets
from datasets import Dataset as HFDataset

from .encoders import Encoder
from .indexed_parallel_documents_dataset import IndexedParallelDocumentsDataset, KEYS


class IndexedParallelBTDocumentsDataset(LanguagePairDataset):
    def __init__(
        self,
        parallel_datasets: Union[IndexedParallelDocumentsDataset, List[IndexedParallelDocumentsDataset]],
        bt_dataset: IndexedParallelDocumentsDataset,
        dictionary,
        encoder: Encoder,
        append_source_id=None,
        append_target_id=None,
        parallel_prob=None,
        seed: int = 1,
    ):
        super().__init__(None, 0, dictionary)
        self.dictionary = dictionary
        self.parallel_prob = parallel_prob
        self.encoder = encoder
        self.seed = seed

        if not isinstance(parallel_datasets, list):
            parallel_datasets = [parallel_datasets]
        self.document_dataset = hf_datasets.concatenate_datasets(
            [dset.document_dataset for dset in parallel_datasets] + [bt_dataset.document_dataset], axis=0
        ) #.flatten_indices().flatten_indices(keep_in_memory=True)
        doc_idx_offset = 0
        index_datasets = []
        for i, dset in enumerate(parallel_datasets):
            if i > 0:
                shifted_index_dataset = dset.index_dataset.remove_columns(
                    KEYS.DOCUMENT_INDEX
                ).add_column(
                    KEYS.DOCUMENT_INDEX, np.array(dset.index_dataset[KEYS.DOCUMENT_INDEX]) + doc_idx_offset
                )
                index_datasets.append(shifted_index_dataset)
            else:
                index_datasets.append(dset.index_dataset)
            doc_idx_offset += len(dset.document_dataset)
        index_datasets = hf_datasets.concatenate_datasets(index_datasets) if len(index_datasets) > 1 else index_datasets[0]
        self.parallel_index_dataset = self._add_bt_field(
            index_datasets, value="False"
        )

        bt_index = bt_dataset.index_dataset
        shifted_bt_index_dataset = bt_index.remove_columns(
            KEYS.DOCUMENT_INDEX
        ).add_column(
            KEYS.DOCUMENT_INDEX, np.array(bt_index[KEYS.DOCUMENT_INDEX]) + doc_idx_offset
        )
        self.bt_index_dataset = self._add_bt_field(
            shifted_bt_index_dataset, value="True"
        ) #.flatten_indices().flatten_indices(keep_in_memory=True)

        uniform_prob = len(self.parallel_index_dataset) / (
            len(self.parallel_index_dataset) + len(self.bt_index_dataset)
        )
        self.mixture_ratios = (
            None
            if uniform_prob < self.parallel_prob
            else [self.parallel_prob, 1 - self.parallel_prob]
        )

        # this gets set after set_epoch or interleave_indices is called
        self.index_dataset = None
        self.epoch = None
        self._interleave_seed = None
        # definitions for langpairdataset functionality
        # ConcatDataset expects a numpy array or list
        self._sizes = None
        self.append_source_id = append_source_id
        self.append_target_id = append_target_id
        self.tgt_eos = (
            self.dictionary.eos()
            if self.append_target_id is None
            else self.append_target_id
        )
        # this is compatibility with LanguagePairDataset collater and its teacher forcing adjustments
        self.src_dict = self.dictionary
        self.left_pad_source = False  # expected by bart model
        self.left_pad_target = False  # expected by bart model
        self.src_lang_id = (
            None
        )  # fairseq 0.10.2 accesses these in LanguagePairDataset.collater (so this attribute must exist)
        self.tgt_lang_id = None
        self.src_sizes = self.sizes
        self.tgt_sizes = None
        self._dataset_ntokens = None
        self._sorted_indices = None
        self._sorted_lengths = None

    def __getitem__(self, index):
        assert (
            self.index_dataset is not None
        ), "You must call the interleave_indices() on this dataset before accessing items"
        item = self.index_dataset[int(index)]
        assert "is_bt" in item
        maybe_noised_encode_fn = (
            self.encoder.encode_noisy if item["is_bt"] else self.encoder.encode
        )
        doc_idx = item[KEYS.DOCUMENT_INDEX]
        src_doc = self.document_dataset[doc_idx][f"{KEYS.DOCUMENT}.{KEYS.LANG1}"]
        tgt_doc = self.document_dataset[doc_idx][f"{KEYS.DOCUMENT}.{KEYS.LANG2}"]
        src_segments = [src_doc[item[KEYS.PARAGRAPH_INDEX]][idx] for idx in item[f"{KEYS.INDICES}.{KEYS.LANG1}"]]
        tgt_segments = [tgt_doc[item[KEYS.PARAGRAPH_INDEX]][idx] for idx in item[f"{KEYS.INDICES}.{KEYS.LANG2}"]]

        with data_utils.numpy_seed(self.seed, self.epoch, index):
            insert_sep = np.random.randint(2, dtype=np.bool)
        if insert_sep and len(src_segments) > 1:
            bos = torch.tensor([self.dictionary.bos()])
            src_out = [bos] * (len(src_segments) * 2 - 1)
            src_out[0::2] = [maybe_noised_encode_fn(seg) for seg in src_segments]
            tgt_out = [bos] * (len(tgt_segments) * 2 - 1)
            tgt_out[0::2] = [self.encoder.encode(seg) for seg in tgt_segments]
        else:
            src_out = [maybe_noised_encode_fn(seg) for seg in src_segments]
            tgt_out = [self.encoder.encode(seg) for seg in tgt_segments]

        src_affix = [self.dictionary.eos()] if self.append_source_id is None else [self.dictionary.eos(), self.append_source_id]
        tgt_affix = [self.dictionary.eos()] if self.append_target_id is None else [self.dictionary.eos(), self.append_target_id]
        src_out = torch.cat(src_out + [torch.tensor(src_affix)])
        tgt_out = torch.cat(tgt_out + [torch.tensor(tgt_affix)])

        if len(src_out) > 1020 or len(tgt_out) > 1020:
            print(len(src_out), len(tgt_out))
            breakpoint()
        return {
            "id": index,
            "source": src_out,
            "target": tgt_out,
        }

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.interleave_indices()

    def interleave_indices(self):
        if self.epoch != self._interleave_seed:
            self.index_dataset = hf_datasets.interleave_datasets(
                [self.parallel_index_dataset, self.bt_index_dataset],
                seed=self.epoch,
                probabilities=self.mixture_ratios,
            ) #.flatten_indices().flatten_indices(keep_in_memory=True)
            self._interleave_seed = self.epoch
            lengths = np.array(self.index_dataset[KEYS.LENGTH])
            self._sorted_indices = lengths.argsort()
            self._sorted_lengths = lengths[self._sorted_indices]
            self._sizes = self._sorted_lengths

    @classmethod
    def _add_bt_field(cls, dataset, value: bool):
        return dataset.add_column(
            "is_bt", column=np.tile(np.array([value]), len(dataset))
        )

    def ordered_sizes(self):
        return self._sorted_lengths

    def ordered_indices(self):
        return self._sorted_indices

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return False
