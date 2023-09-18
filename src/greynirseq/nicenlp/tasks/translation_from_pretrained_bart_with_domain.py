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
from fairseq.data import BaseWrapperDataset, Dictionary, LanguagePairDataset, data_utils
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask, load_langpair_dataset

from .translation_from_pretrained_bart import TranslationFromPretrainedBARTTask

_EES_DEFAULT_DOMAIN = "ees_ótiltekið"


class DomainPrefixingDataset(BaseWrapperDataset):
    """Prefix each source segment of a LangPairDataset with a domain specifer token"""

    def __init__(
        self,
        langpair_dataset: LanguagePairDataset,
        domain_per_example: List[str],
        src_dict: Dictionary,
        domain_dict: Dictionary,
        seed: int = 1,
    ):
        super().__init__(langpair_dataset)
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

    def collater(self, samples, pad_to_length=None):
        return self.dataset.collater(samples, pad_to_length)


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
        parser.add_argument('--train-domains', type=str, required=False,
                            help='File of same line count as training split where each '
                            'line has some domain from the domain_dict.txt')
        parser.add_argument('--valid-domains', type=str, required=False,
                            help='File of same line count as validation split where each '
                            'line has some domain from the domain_dict.txt')
        parser.add_argument('--test-domains', type=str, required=False,
                            help='File of same line count as test split where each '
                            'line has some domain from the domain_dict.txt')
        # fmt: on

    def __init__(self, cfg, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.args = cfg
        self.langs = cfg.langs.split(",")
        self.load_domains(cfg)
        for dict_ in [src_dict, tgt_dict]:
            for lang in self.langs:
                dict_.add_symbol("[{}]".format(lang))
            dict_.add_symbol("<mask>")
            for idx in range(len(self.domain_dict)):
                if idx < self.domain_dict.nspecial:
                    continue
                symbol = self.domain_dict.symbols[idx]
                dict_.add_symbol(f"<{symbol}>")

    def load_domains(self, cfg):
        self.domain_dict = self.load_dictionary(cfg.domain_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

        if "train" in split:
            domain_path = self.cfg.train_domains
        elif "valid" in split:
            domain_path = self.cfg.valid_domains
        elif "test" in split:
            domain_path = self.cfg.test_domains

        else:
            raise ValueError(f"Unknown split {split}")
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
            dataset_impl=self.cfg.dataset_impl,
            upsample_primary=self.cfg.upsample_primary,
            left_pad_source=self.cfg.left_pad_source,
            left_pad_target=self.cfg.left_pad_target,
            max_source_positions=getattr(self.cfg, "max_source_positions", 512),
            max_target_positions=getattr(self.cfg, "max_target_positions", 512),
            load_alignments=self.cfg.load_alignments,
            prepend_bos=getattr(self.cfg, "prepend_bos", False),
            append_source_id=True,
        )
        self.datasets[split] = DomainPrefixingDataset(
            langpair_dataset=self.datasets[split],
            domain_per_example=domain_per_example,
            src_dict=self.src_dict,
            domain_dict=self.domain_dict,
            seed=self.cfg.seed,
        )
