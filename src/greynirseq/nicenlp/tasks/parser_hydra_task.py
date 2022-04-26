# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from fairseq.data import (
    Dictionary,
    IdDataset,
    ListDataset,
    NumelDataset,
    NumSamplesDataset,
    RightPadDataset,
    SortDataset,
    data_utils,
    encoders,
)
from fairseq.dataclass import FairseqDataclass
from fairseq.tasks import FairseqTask, register_task
from icecream import ic
from omegaconf import II

import greynirseq.nicenlp.utils.constituency.greynir_utils as greynir_utils
from greynirseq.nicenlp.data.datasets import (
    NestedDictionaryDatasetFix,
    NumWordsDataset,
    RightPad2dDataset,
    WordEndMaskDataset,
)
from greynirseq.nicenlp.data.lambda_dataset import LambdaDataset
from greynirseq.nicenlp.data.parsing_datsets import (
    GreynirParsingDataset,
    GreynirTreeAugmentationDataset,
    GreynirTreeJSONLDataset,
)
from greynirseq.nicenlp.data.text_encoder_dataset import TextEncodingDataset
from greynirseq.nicenlp.utils.constituency.incremental_parsing import (
    EOS_LABEL,
    NULL_LABEL,
    ROOT_LABEL,
    ParseAction,
    get_banned_attachments,
)
from greynirseq.nicenlp.utils.label_schema.label_schema import label_schema_as_dictionary, parse_label_schema

logger = logging.getLogger(__name__)


@dataclass
class ParserHydraConfig(FairseqDataclass):
    data: Optional[Any] = field(default=None, metadata={"help": "Data directory, it should also contain dict.txt file"})
    nonterm_schema: Optional[str] = field(default=None, metadata={"help": "Hierarchical label-schema for nonterminals"})
    term_schema: Optional[str] = field(default=None, metadata={"help": "Hierarchical label-schema for terminals"})
    case_punct_noise: Optional[float] = field(
        default=0.0, metadata={"help": "Stochastically remove punctuation and casing in training data"}
    )
    label_file: Optional[str] = field(
        default=None, metadata={"help": "Label dictionary file, analogous to fairseqs dict.txt"}
    )
    # borrow from top level
    _bpe: Any = II("bpe")
    _dataset: Optional[Any] = II("dataset")
    _seed: int = II("common.seed")


@register_task("parser_hydra", dataclass=ParserHydraConfig)
class ParserHydraTask(FairseqTask):
    cfg: ParserHydraConfig

    def __init__(
        self, cfg: ParserHydraConfig, data_dictionary: Dictionary, label_dictionary: Dictionary, is_word_initial
    ):
        super().__init__(cfg)
        self.dictionary = data_dictionary

        if not hasattr(self, "args") and hasattr(self, "cfg"):
            self.args = self.cfg

        self.label_dictionary = label_dictionary

        self.is_word_initial = is_word_initial
        self.num_labels = len(self.label_dictionary)

    @classmethod
    def setup_task(cls, cfg: ParserHydraConfig, **_kwargs):
        data_dict = cls.load_dictionary(cls, os.path.join(cfg.data, "dict.txt"))
        logger.info("[input] dictionary: {} types".format(len(data_dict)))
        label_dict = cls.load_dictionary(cls, cfg.label_file)
        label_dict.add_symbol(EOS_LABEL)
        logger.info("[label] dictionary: {} types".format(len(label_dict)))

        assert cfg._bpe._name == cfg._parent.bpe._name

        is_word_initial = cls.get_word_beginnings(cfg._parent.bpe, data_dict)

        return ParserHydraTask(cfg, data_dict, label_dictionary=label_dict, is_word_initial=is_word_initial)

    @classmethod
    def load_label_dictionary(cls, cfg: ParserHydraConfig, filename: str, **_kwargs):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
        """
        assert Path(filename).exists(), f"Expected label_schema file at {filename}"
        label_schema = parse_label_schema(filename)

        return label_schema_as_dictionary(label_schema), label_schema

    @classmethod
    def load_dictionary(cls, cfg: ParserHydraConfig, filename: str, add_mask: bool = True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        if add_mask:
            dictionary.add_symbol("<mask>")
        return dictionary

    @classmethod
    def get_word_beginnings(cls, cfg: FairseqDataclass, dictionary: Dictionary):
        # Note that cfg here is some 'BPEConfig' and not ParserHydraConfig itself
        bpe = encoders.build_bpe(cfg)
        if bpe is not None:

            def is_beginning_of_word(i):
                if i < dictionary.nspecial:
                    return True
                tok = dictionary[i]
                if tok.startswith("madeupword"):
                    return True
                try:
                    return bpe.is_beginning_of_word(tok)
                except ValueError:
                    return True

            is_word_initial = {}
            for i in range(len(dictionary)):
                is_word_initial[i] = int(is_beginning_of_word(i))
            return is_word_initial
        return None

    def load_dataset(self, split: str, combine: bool = False, **_kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""
        greynirtrees_dataset = GreynirTreeJSONLDataset.from_path(Path(self.cfg.data) / f"{split}.jsonl")

        bpe = encoders.build_bpe(self.cfg._bpe)
        if "train" in split and self.cfg.case_punct_noise > 0.0:
            greynirtrees_dataset = GreynirTreeAugmentationDataset(greynirtrees_dataset, self.cfg.case_punct_noise)
        greynirparsing_dataset = GreynirParsingDataset(
            greynirtrees_dataset,
            label_dictionary=self.label_dictionary,
            source_dictionary=self.source_dictionary,
            bpe=bpe,
            prepend_token=self.source_dictionary.bos(),
        )

        src_tokens = TextEncodingDataset(
            LambdaDataset(greynirtrees_dataset, lambda_fn=lambda x: x.text),
            self.source_dictionary,
            bpe=bpe,
            prepend_token=self.source_dictionary.bos(),
        )
        word_mask = WordEndMaskDataset(
            src_tokens, self.source_dictionary, self.is_word_initial, bos_value=0, eos_value=1
        )

        with data_utils.numpy_seed(self.cfg._seed):
            shuffle = np.random.permutation(len(src_tokens))

        dataset = {
            "id": IdDataset(),
            "net_input": {
                "src_tokens": RightPadDataset(src_tokens, pad_idx=self.source_dictionary.pad()),
                "word_mask": RightPadDataset(word_mask, pad_idx=0),
                "preorder_nts": RightPadDataset(
                    LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["inputs"]["preorder_nts"]),
                    pad_idx=self.label_dictionary.pad(),
                ),
                "preorder_depths": RightPad2dDataset(
                    LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["inputs"]["preorder_depths"]),
                    pad_idx=0,
                ),
                "preorder_mask": RightPad2dDataset(
                    LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["inputs"]["preorder_mask"]), pad_idx=0
                ),
                "chain_mask": RightPad2dDataset(
                    LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["inputs"]["chain_mask"]), pad_idx=0
                ),
                "preorder_spans": RightPad2dDataset(
                    LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["inputs"]["preorder_spans"]), pad_idx=0
                ),
                "preorder_flags": RightPad2dDataset(
                    LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["inputs"]["preorder_flags"]),
                    pad_idx=self.label_dictionary.pad(),
                ),
                "nwords_per_step": RightPadDataset(
                    LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["inputs"]["nwords_per_step"]),
                    pad_idx=0,
                ),
            },
            "target_depths": RightPadDataset(
                LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["targets"]["target_depths"]),
                pad_idx=self.label_dictionary.pad(),
            ),
            "target_padding_mask": RightPadDataset(
                LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["targets"]["target_padding_mask"]),
                pad_idx=1,
            ),
            "target_parents": RightPadDataset(
                LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["targets"]["target_parents"]),
                pad_idx=self.label_dictionary.pad(),
            ),
            "target_preterms": RightPadDataset(
                LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["targets"]["target_preterms"]),
                pad_idx=self.label_dictionary.pad(),
            ),
            "target_parent_flags": RightPad2dDataset(
                LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["targets"]["target_parent_flags"]),
                pad_idx=self.label_dictionary.pad(),
            ),
            "target_preterm_flags": RightPad2dDataset(
                LambdaDataset(greynirparsing_dataset, lambda_fn=lambda x: x["targets"]["target_preterm_flags"]),
                pad_idx=self.label_dictionary.pad(),
            ),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "nwords": NumWordsDataset(src_tokens, self.dictionary, self.is_word_initial),
        }

        nested_dataset = NestedDictionaryDatasetFix(dataset, sizes=greynirparsing_dataset.sizes)

        dataset = SortDataset(nested_dataset, sort_order=[shuffle])

        logger.info("Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def prepare_sentences(self, sentences: List[str]):
        bpe = encoders.build_bpe(self.cfg._bpe)
        src_tokens = TextEncodingDataset(
            sentences, self.source_dictionary, bpe=bpe, prepend_token=self.source_dictionary.bos()
        )
        return self.prepare_tokens(src_tokens)

    def prepare_tokens(self, tokens: torch.Tensor):
        sizes = [len(seq) for seq in tokens]
        src_tokens = ListDataset(tokens, sizes=sizes)
        src_tokens = RightPadDataset(src_tokens, pad_idx=self.source_dictionary.pad())

        word_mask = WordEndMaskDataset(src_tokens, self.dictionary, self.is_word_initial, bos_value=0, eos_value=1)

        dataset = {
            "id": IdDataset(),
            "net_input": {"src_tokens": src_tokens, "word_mask": RightPadDataset(word_mask, pad_idx=0)},
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "nwords": NumWordsDataset(src_tokens, self.dictionary, self.is_word_initial),
            "nsentences": NumSamplesDataset(),
        }
        dataset = NestedDictionaryDatasetFix(dataset, sizes=[src_tokens.sizes])
        return dataset

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.dictionary

    @property
    def root_label_index(self):
        return self.label_dictionary.index(ROOT_LABEL)

    def logits_to_actions(
        self, state: Any, trees: Optional[List[greynir_utils.Node]] = None, finalizable: Optional[List[bool]] = None
    ) -> ParseAction:
        bsz = len(state.preterm_flag_logits[-1])
        if trees:
            assert len(trees) == bsz
            banned_attachments = [get_banned_attachments(tree) for tree in trees]
        illegal = [self.label_dictionary.bos(), self.label_dictionary.pad(), self.label_dictionary.unk()]
        state.parent_flag_logits[-1][:, illegal] = float("-inf")
        state.preterm_flag_logits[-1][:, illegal] = float("-inf")
        state.parent_logits[-1][:, illegal] = float("-inf")
        state.preterm_logits[-1][:, illegal] = float("-inf")

        threshold = 0
        depths = state.attention[-1].argmax(-1)
        parents = [self.label_dictionary.symbols[i] for i in state.parent_logits[-1].argmax(-1)]
        preterms = [self.label_dictionary.symbols[i] for i in state.preterm_logits[-1].argmax(-1)]

        for seq_idx in range(bsz):
            preterm, parent, depth = preterms[seq_idx], parents[seq_idx], depths[seq_idx].item()
            is_finalizable = finalizable[seq_idx]
            if parent == EOS_LABEL:
                preterms[seq_idx] = NULL_LABEL
                continue

            if trees:
                banned_att, depth_to_labels = banned_attachments[seq_idx]
            if is_finalizable and parent == preterm == NULL_LABEL:
                # we can end tree here
                parent = EOS_LABEL
                parents[seq_idx] = parent
                continue
            elif parent == preterm == NULL_LABEL:
                # force preterminal to be non-null since we cannot end tree here
                preterm = self.label_dictionary.symbols[
                    state.preterm_logits[-1][seq_idx].sort(descending=True).indices[1]
                ]
                preterms[seq_idx] = preterm
                ic(f"Parent and preterm are NULL, depth={depth}, forcing preterminal to become {preterms[seq_idx]}")

            if parent != NULL_LABEL and state.parent_flag_logits[-1][seq_idx].gt(threshold).any():
                idxs = state.parent_flag_logits[-1][seq_idx].gt(threshold).nonzero().squeeze(-1)
                parent = parent + "-" + "-".join(self.label_dictionary.string(idxs).split(" "))
                parents[seq_idx] = parent
            if preterm != NULL_LABEL and state.preterm_flag_logits[-1][seq_idx].gt(threshold).any():
                idxs = state.preterm_flag_logits[-1][seq_idx].gt(threshold).nonzero().squeeze(-1)
                preterm = preterm + "-" + "-".join(self.label_dictionary.string(idxs).split(" "))
                preterms[seq_idx] = preterm

            if is_finalizable and trees and (parent in depth_to_labels[depth - 1] or depth in banned_att):
                # parser is trying to extend-or-attach a node with that is either
                #     - the same as current node
                #     - shares label in common with a node that is in the same unary chain
                #     - is part of a unary chain that is already too long
                # so we can end tree here
                parent = EOS_LABEL
                parents[seq_idx] = parent
                continue
            elif preterm == NULL_LABEL and trees and (parent in depth_to_labels[depth - 1] or depth in banned_att):
                # parser is trying to only-extend a node with that is either
                #     - the same as current node
                #     - shares label in common with a node that is in the same unary chain
                #     - is part of a unary chain that is already too long
                # and we cannot end the tree here so we are forced to choose some preterminal
                # to minimize error accumulation we set parent to null
                trees[seq_idx].pretty_print()
                preterms[seq_idx] = self.label_dictionary.symbols[
                    state.preterm_logits[-1][seq_idx].sort(descending=True).indices[1]
                ]
                msg = (
                    f"parent in {depth_to_labels[depth - 1]}"
                    if parent in depth_to_labels[depth - 1]
                    else f"depth in {banned_att[depth]}"
                )
                ic(
                    f"Parser is only-extending: {parent} {preterm} {depth}, {msg}, "
                    f"setting parent to NULL and preterm to {preterms[seq_idx]}"
                )
                parent = NULL_LABEL
                parents[seq_idx] = parent

        assert not any(NULL_LABEL in parents[idx] and NULL_LABEL in preterms[idx] for idx in range(len(parents)))
        parse_actions = [
            ParseAction(parent=par, preterminal=pret, depth=dep)
            for (par, pret, dep) in zip(parents, preterms, depths.tolist())
        ]
        return parse_actions
