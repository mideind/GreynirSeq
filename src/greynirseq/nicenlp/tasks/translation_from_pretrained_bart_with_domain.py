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

from typing import List

import numpy
import torch
from fairseq import utils
from fairseq.data import BaseWrapperDataset, Dictionary, data_utils
from fairseq.data.language_pair_dataset import collate
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset
from fairseq.tasks.translation_from_pretrained_bart import TranslationFromPretrainedBARTTask
from torch.utils.data import Dataset

_EES_DEFAULT_DOMAIN = "ees_ótiltekið"


class DomainPrefixingDataset(BaseWrapperDataset):
    """Prefix each source segment of a LangPairDataset with a domain specifer token"""

    def __init__(
        self,
        langpair_dataset: Dataset,
        domain_per_example: List[str],
        src_dict: Dictionary,
        domain_dict: Dictionary,
        seed: int = 1,
    ):
        self.domain_per_example = domain_per_example
        self.domain_dict = domain_dict
        self.src_dict = src_dict
        self.eos = self.dataset.eos
        self.seed = seed
        self.epoch = 0

    def __getitem__(self, index: int):
        pair_dict = self.dataset[index]
        domains = self.domain_per_example[index].split(" ")
        sampled_index = 0
        if len(domains) > 1:
            normal_domain_weight = 1.0
            default_domain_weight = 0.1
            weights = numpy.array(
                [default_domain_weight if d == _EES_DEFAULT_DOMAIN else normal_domain_weight for d in domains]
            )
            with data_utils.numpy_seed(self.seed, self.epoch, index):
                sampled_index = numpy.random.choice(range(len(domains)), p=weights / sum(weights))
        domain = domains[sampled_index]
        domain_index = self.src_dict.index(f"<{domain}>")
        vec = pair_dict["source"].new_ones(1) * domain_index
        pair_dict["source"] = torch.cat([vec, pair_dict["source"]])
        return pair_dict

    def __getattr__(self, attr):
        return getattr(self.dataset, attr)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )
        if self.src_lang_id is not None or self.tgt_lang_id is not None:
            src_tokens = res["net_input"]["src_tokens"]
            bsz = src_tokens.size(0)
            if self.src_lang_id is not None:
                res["net_input"]["src_lang_id"] = torch.LongTensor([[self.src_lang_id]]).expand(bsz, 1).to(src_tokens)
            if self.tgt_lang_id is not None:
                res["tgt_lang_id"] = torch.LongTensor([[self.tgt_lang_id]]).expand(bsz, 1).to(src_tokens)
        return res


@register_task("translation_from_pretrained_bart_domain")
class TranslationFromPretrainedBARTTaskWithDomain(TranslationFromPretrainedBARTTask):
    """An extension of TranslationFromPretrainedBARTTask from fairseq that includes domain specifer tokens"""

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
        parser.add_argument('--domain-dict', type=str, required=True,
                            help='Path a file that contains a list of all domains (same format as dict.txt)')
        parser.add_argument('--train-domains', type=str, required=True,
                            help='File of same line count as training split where each '
                            'line has some domain from the domain_dict.txt')
        parser.add_argument('--valid-domains', type=str, required=True,
                            help='File of same line count as validation split where each '
                            'line has some domain from the domain_dict.txt')
        # fmt: on

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)
        self.langs = args.langs.split(",")
        self.load_domains(args)
        for dict_ in [src_dict, tgt_dict]:
            for lang in self.langs:
                dict_.add_symbol("[{}]".format(lang))
            dict_.add_symbol("<mask>")
            for idx in range(len(self.domain_dict)):
                if idx < self.domain_dict.nspecial:
                    continue
                symbol = self.domain_dict.symbols[idx]
                dict_.add_symbol(f"<{symbol}>")

    def load_domains(self, args):
        self.domain_dict = self.load_dictionary(args.domain_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.args.source_lang, self.args.target_lang

        if "train" in split:
            domain_path = self.args.train_domains
        elif "valid" in split:
            domain_path = self.args.valid_domains
        else:
            assert False, "We currently do not handle test splits"
        with open(domain_path) as in_fh:
            domain_per_example = [domain.strip() for domain in in_fh]

        self.datasets[split] = load_langpair_dataset(
            data_path,
            split,
            src,
            self.src_dict,
            tgt,
            self.tgt_dict,
            combine=combine,
            dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=getattr(self.args, "max_source_positions", 512),
            max_target_positions=getattr(self.args, "max_target_positions", 512),
            load_alignments=self.args.load_alignments,
            prepend_bos=getattr(self.args, "prepend_bos", False),
            append_source_id=True,
        )
        self.datasets[split] = DomainPrefixingDataset(
            langpair_dataset=self.datasets[split],
            domain_per_example=domain_per_example,
            src_dict=self.src_dict,
            domain_dict=self.domain_dict,
            seed=self.args.seed,
        )
