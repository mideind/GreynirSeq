# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Some modifications by Miðeind ehf. below to support the hydra/omegaconf config system.
# See open issues/PR:
# https://github.com/facebookresearch/fairseq/issues/3169
# https://github.com/facebookresearch/fairseq/issues/4479
# https://github.com/facebookresearch/fairseq/pull/4487
from dataclasses import dataclass, field
from typing import cast

import torch
from fairseq import utils
from fairseq.data import Dictionary, LanguagePairDataset
from fairseq.tasks import TASK_CLASS_NAMES, TASK_REGISTRY, register_task
from fairseq.tasks.translation import TranslationConfig, TranslationTask, load_langpair_dataset
from omegaconf import MISSING

# ========================WARNING========================
# we manually edit the TASK_REGISTRY to add our (fixed) version of the task

if "translation_from_pretrained_bart" in TASK_REGISTRY:
    del TASK_REGISTRY["translation_from_pretrained_bart"]
    TASK_CLASS_NAMES.remove("TranslationFromPretrainedBARTTask")


@dataclass
class TranslationFromPretrainedBARTConfig(TranslationConfig):
    langs: str = field(
        default=MISSING,
        metadata={
            "help": "comma-separated list of monolingual language, "
            'for example, "en,de,fr". These should match the '
            "langs from pretraining (and be in the same order). "
            "You should always add all pretraining language idx "
            "during finetuning."
        },
    )

    prepend_bos: bool = field(
        default=False,
        metadata={"help": "prepend bos token to each sentence, which matches " "mBART pretraining"},
    )


@register_task("translation_from_pretrained_bart", dataclass=TranslationFromPretrainedBARTConfig)
class TranslationFromPretrainedBARTTask(TranslationTask):
    """
    Translate from source language to target language with a model initialized with a multilingual pretrain.

    Args:
        src_dict (~fairseq.data.Dictionary): dictionary for the source language
        tgt_dict (~fairseq.data.Dictionary): dictionary for the target language

    .. note::

        The translation task is compatible with :mod:`fairseq-train`,
        :mod:`fairseq-generate` and :mod:`fairseq-interactive`.

    The translation task provides the following additional command-line
    arguments:

    .. argparse::
        :ref: fairseq.tasks.translation_parser
        :prog:
    """

    def __init__(self, cfg: TranslationFromPretrainedBARTConfig, src_dict: Dictionary, tgt_dict: Dictionary):
        super().__init__(cfg, src_dict=src_dict, tgt_dict=tgt_dict)
        self.langs = cfg.langs.split(",")
        for dictionary in [src_dict, tgt_dict]:
            for lang in self.langs:
                dictionary.add_symbol("[{}]".format(lang))
            dictionary.add_symbol("<mask>")

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        self.cfg = cast(TranslationFromPretrainedBARTConfig, self.cfg)
        # WARNING
        # This function has not been tested with the new fairseq version
        # The assumption here is that self.cfg is the 'task' config
        paths = utils.split_paths(self.cfg.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        src, tgt = self.cfg.source_lang, self.cfg.target_lang

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
            max_source_positions=getattr(self.cfg, "max_source_positions", 1024),
            max_target_positions=getattr(self.cfg, "max_target_positions", 1024),
            load_alignments=self.cfg.load_alignments,
            prepend_bos=getattr(self.cfg, "prepend_bos", False),
            append_source_id=True,
        )

    def build_generator(self, models, cfg, **unused):
        if getattr(cfg, "score_reference", False):
            from fairseq.sequence_scorer import SequenceScorer

            return SequenceScorer(
                self.target_dictionary,
                eos=self.tgt_dict.index("[{}]".format(self.cfg.target_lang)),
            )
        else:
            from fairseq.sequence_generator import SequenceGenerator

            return SequenceGenerator(
                models,
                self.target_dictionary,
                beam_size=cfg.get("beam", 5),
                max_len_a=cfg.get("max_len_a", 0),
                max_len_b=cfg.get("max_len_b", 200),
                max_len=cfg.get("max_len", 0),
                min_len=cfg.get("min_len", 1),
                normalize_scores=not cfg.get("unnormalized", False),
                len_penalty=cfg.get("lenpen", 1),
                unk_penalty=cfg.get("unkpen", 0),
                temperature=cfg.get("temperature", 1.0),
                match_source_len=cfg.get("match_source_len", False),
                no_repeat_ngram_size=cfg.get("no_repeat_ngram_size", 0),
                eos=self.tgt_dict.index("[{}]".format(self.cfg.target_lang)),
            )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        src_lang_id = self.source_dictionary.index("[{}]".format(self.cfg.source_lang))
        source_tokens = []
        for s_t in src_tokens:
            s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
            source_tokens.append(s_t)
        dataset = LanguagePairDataset(
            source_tokens,
            src_lengths,
            self.source_dictionary,
            tgt_dict=self.target_dictionary,
            constraints=constraints,
        )
        return dataset
