# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

# Based on fairseq/tasks/translation.py and fairseq/tasks/translation_from_pretrained_bart.py
# that has the following license
#
#    Copyright (c) Facebook, Inc. and its affiliates.
#
#    This source code is licensed under the MIT license found in the
#    LICENSE file in the root directory of this source tree.

import json
import logging
from typing import Any, List, Optional, Union

import numpy
import torch
from fairseq.data import encoders
from fairseq import utils
from fairseq.data import BaseWrapperDataset, Dictionary, data_utils, FairseqDataset, iterators
from fairseq.data.language_pair_dataset import collate
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset
from fairseq.tasks.translation_from_pretrained_bart import (
    TranslationFromPretrainedBARTTask
)
from fairseq.data import ConcatDataset

from torch.utils.data import Dataset
import datasets as hf_datasets

from icecream import ic

from .document_dataset import DocumentJONSLDataset
from .fragment_noise import FragmentNoiser
from .word_noise import WordNoiser
from .spm_segmentation_noise import SpmNoiser


logger = logging.getLogger(__name__)


# class LambdaDataset(BaseWrapperDataset):
#     def __init__(self, dataset: Dataset, lambda_fn):
#         super().__init__(dataset)
#         self.lambda_fn = lambda_fn

#     def __getitem__(self, index: int):
#         return self.lambda_fn(self.dataset[index])


@register_task("document_translation_from_pretrained_bart")
class DocumentTranslationFromPretrainedBART(TranslationFromPretrainedBARTTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)
        parser.add_argument('--langs',  type=str, metavar='LANG',
                            help='comma-separated list of monolingual language, '
                                 'for example, "en,de,fr". These should match the '
                                 'langs from pretraining (and be in the same order). '
                                 'You should always add all pretraining language idx '
                                 'during finetuning.')
        parser.add_argument('--prepend-bos', action='store_true',
                            help='prepend bos token to each sentence, which matches '
                                 'mBART pretraining')
        parser.add_argument('--bt-subset', type=str, default="")
        parser.add_argument('--sentencepiece-alpha', type=float, default=0.01)
        parser.add_argument('--parallel-prob', type=float, help="Probability of sampling parallel data if bt data is included (Note: NOT sample weight)", default=0.33)
        parser.add_argument('--word-noise-prob', type=float, default=0.01)
        parser.add_argument('--fragment-noise-prob', type=float, default=0.01)
        parser.add_argument('--max-shuffle-dist', type=int, default=3)
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.langs = args.langs.split(",")
        for dict_ in [src_dict, tgt_dict]:
            for lang in self.langs:
                dict_.add_symbol("[{}]".format(lang))
            dict_.add_symbol("<mask>")

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        # this is for sharding
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_dir_path = paths[(epoch - 1) % len(paths)]
        # this is for different datasets that comprise the training set
        dataset_names = split.split(",")
        bt_dataset_names = self.args.bt_subset.split(",")
        parallel_dataset_names = [
            name for name in dataset_names if name not in bt_dataset_names
        ]
        assert parallel_dataset_names, "Expected at least one parallel dataset"

        # infer langcode and translation direction
        src, tgt = self.args.source_lang, self.args.target_lang
        direction = f"{src}-{tgt}"

        from .sentencepiece_bpe_sampling import SentencepieceBPESampled
        import copy

        _args_w_bpe_sampling = copy.deepcopy(self.args)
        _args_w_bpe_sampling.bpe = "sentencepiece_sampled"
        spm_w_noise = encoders.build_bpe(_args_w_bpe_sampling)
        # self.args.word_noise_prob = 0.02
        # max_shuffle_dist = 3
        # fragment_noise_prob = 0.02
        word_noise_prob = self.args.max_shuffle_dist
        max_shuffle_dist = self.args.max_shuffle_dist
        fragment_noise_prob = self.args.fragment_noise_prob
        word_noiser = WordNoiser(word_noise_prob, max_shuffle_dist)
        noisy_subword_enc = SpmNoiser(self.src_dict, spm_w_noise)
        fragment_noiser = FragmentNoiser(self.src_dict, fragment_noise_prob, min_val=self.src_dict.nspecial, max_val=len(self.src_dict) - 1 - len(self.langs))
        # XXX: since this is at document level, we probably dont want to apply this too aggressively
        # XXX: e.g. only enable a noiser with some probability

        bpe = encoders.build_bpe(self.args)
        from .indexed_parallel_documents_dataset import IndexedParallelDocumentsDataset

        def encode_parallel_fn(parts: List[Union[int, str]]):
            encoded_parts = []
            encoded_parts = [
                self.src_dict.encode_line(bpe.encode(part), append_eos=False)
                if isinstance(part, str)
                else torch.tensor([part])
                for part in parts
            ]
            encoded_parts.append(torch.tensor([self.src_dict.eos()]))
            return torch.cat(encoded_parts)

        def encode_bt_fn(string: List[Union[int, str]]):
            res = word_noiser.apply(string)
            res = noisy_subword_enc.apply(res)
            seq_tensor = fragment_noiser.apply(res.sequence, res.noise_allowed_mask)
            return seq_tensor

        src_paths = [
            f"{data_dir_path}/{name}.{direction}.{src}.jsonl"
            for name in parallel_dataset_names
        ]
        tgt_paths = [
            f"{data_dir_path}/{name}.{direction}.{tgt}.jsonl"
            for name in parallel_dataset_names
        ]
        max_seq_len = int(self.args.max_source_positions * 0.8)  # to account for segmentation noise
        ic(split)
        parallel_dataset = IndexedParallelDocumentsDataset.from_parallel_jsonl_many(
            src_paths,
            tgt_paths,
            bpe,
            self.src_dict,
            encode_fn=encode_parallel_fn,
            max_seq_len=max_seq_len,
            append_source_id=self.src_dict.index("[{}]".format(self.args.source_lang)),
            append_target_id=self.tgt_dict.index("[{}]".format(self.args.target_lang)),
        )
        # print(parallel_dataset.index_dataset["length"].max(), parallel_dataset.index_dataset["length"].argmax())
        print(split, parallel_dataset[0])

        # noised_src_01 = word_noiser.apply(parallel_dataset[0]["source"])
        # noised_seq_02 = noisy_subword_enc.apply(noised_src_01)
        # noised_seq_03 = fragment_noiser.apply(noised_seq_02.sequence, noised_seq_02.noise_allowed_mask)

        # noised_src = noisy_subword_enc.apply(parallel_dataset[0]["source"])
        # fragment_noiser.apply(parallel_dataset[0]["source"])
        # parallel_dataset[0]

        # from pprint import pprint
        # pprint(set([bpe_w_segm_noise.encode("sample_ratios is sampling weight of each dataset, not an actual ratio") for _ in range(100)]))

        if "valid" in split or "test" in split:
            self.datasets[split] = parallel_dataset
            return parallel_dataset

        src_paths = [
            f"{data_dir_path}/{name}.{direction}.{src}.jsonl"
            for name in bt_dataset_names
        ]
        tgt_paths = [
            f"{data_dir_path}/{name}.{direction}.{tgt}.jsonl"
            for name in bt_dataset_names
        ]
        bt_dataset = IndexedParallelDocumentsDataset.from_parallel_jsonl_many(
            src_paths,
            tgt_paths,
            bpe,
            self.src_dict,
            encode_fn=encode_bt_fn,
            max_seq_len=max_seq_len,
            append_source_id=self.src_dict.index("[{}]".format(self.args.source_lang)),
            append_target_id=self.tgt_dict.index("[{}]".format(self.args.target_lang)),
        )


        # glossary_start = 131
        # glossary_end = 132
        # domain_start_id = 133
        # domain_end_id = 134
        # string_noise_prob = 0.1
        # foo = SentencepieceDecodingDataset(self.cfg, foo, self.src_dict)
        # foo = FakeGlossaryDataset(self.cfg, foo, glossary_start, glossary_end)
        # foo = FakeStringDomainDataset(foo, domain_start_id, domain_end_id)
        # foo = FakeStringNoiseDataset(self.cfg, foo, string_noise_prob)
        # foo = WordNoiseDataset(self.cfg, foo, word_noise_prob, max_shuffle_dist)
        # foo = SentencepieceSegmentationNoiseDataset(self.cfg, foo, self.src_dict)
        # foo = FragmentNoiseDataset(self.cfg, foo, self.src_dict, fragment_noise_prob)

        desired_prob = self.args.parallel_prob
        bt_ntokens = bt_dataset.dataset_ntokens
        parallel_ntokens = parallel_dataset.dataset_ntokens
        # sample_ratios is sampling weight of each dataset, not an actual ratio
        total = parallel_ntokens + bt_ntokens
        dataset_weights = [parallel_ntokens, bt_ntokens]
        # we don't care if parallel rate is greater than desired we don't do anything
        # since we don't want to upsample backtranslations
        if (parallel_ntokens / total) < desired_prob:
            # parallel is less frequent than we desire so:
            # assume a is parallel, b is bt, w is weight, p is desired prob
            # then the current rate is
            #     a / (a + b) = p_current < p
            # but we want this
            #     wa / (wa + b) = p
            # rearrange to get
            #     w = (b/a) * (p / (1 - p))
            upsample_primary_weight = (bt_ntokens / parallel_ntokens) * (
                self.args.parallel_prob / (1 - self.args.parallel_prob)
            )
            dataset_weights = [upsample_primary_weight * parallel_ntokens, bt_ntokens]
        # rescale so that smallest upsampling rate is exactly 1 (so that dataset remains unchanged)
        min_weight = min(dataset_weights)
        dataset_weights = [int(w / min_weight) for w in dataset_weights]
        # dataset_weights = [w / min_weight for w in dataset_weights]
        ic(dataset_weights)

        dataset = parallel_dataset
        if bt_dataset_names:
            dataset = ConcatDataset(
                [parallel_dataset, bt_dataset], sample_ratios=dataset_weights,
            )
        ic("Finished concatenating")

        self.datasets[split] = dataset
        return dataset

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=1,
    ):
        # For default fairseq task, return same iterator across epochs
        # as datasets are not dynamic, can be overridden in task specific
        # setting.
        if dataset in self.dataset_to_epoch_iter:
            return self.dataset_to_epoch_iter[dataset]

        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # get indices ordered by example size
        with data_utils.numpy_seed(seed):
            indices = dataset.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            indices = data_utils.filter_by_size(
                indices,
                dataset,
                max_positions,
                raise_exception=(not ignore_invalid_inputs),
            )

        # create mini-batches with given size constraints
        batch_sampler = data_utils.batch_by_size(
            indices,
            dataset.num_tokens,
            max_tokens=max_tokens,
            max_sentences=max_sentences,
            required_batch_size_multiple=required_batch_size_multiple,
        )

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )
        self.dataset_to_epoch_iter[dataset] = epoch_iter
        return epoch_iter
