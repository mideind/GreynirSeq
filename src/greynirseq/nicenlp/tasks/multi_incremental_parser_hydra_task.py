# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import logging
import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from fairseq.data import (
    Dictionary,
    IdDataset,
    FairseqDataset,
    ListDataset,
    NumelDataset,
    NumSamplesDataset,
    RightPadDataset,
    SortDataset,
    ConcatDataset,
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
from greynirseq.nicenlp.data.constant_dataset import ConstantDataset
from greynirseq.nicenlp.data.parsing_datasets import (
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
from greynirseq.nicenlp.data.multitask_data_utils import MultidatasetEpochBatchIterator, MultitaskDatasetWrapper

logger = logging.getLogger(__name__)
_GREYNIR_TASK_SYMBOL = "__greynir__"
_ICEPAHC_TASK_SYMBOL = "__icepahc__"


@dataclass
class MultiIncrementalParserHydraConfig(FairseqDataclass):
    data: Optional[Any] = field(default=None, metadata={"help": "Data directory, it should also contain dict.txt file"})
    nonterm_schema: Optional[str] = field(default=None, metadata={"help": "Hierarchical label-schema for nonterminals"})
    # term_schema: Optional[str] = field(default=None, metadata={"help": "Hierarchical label-schema for terminals"})
    case_punct_noise: Optional[float] = field(
        default=0.0, metadata={"help": "Stochastically remove punctuation and casing in training data"}
    )
    limit_greynir: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Number of training examples to include from GreynirCorpus"
        },
    )
    limit_icepahc: Optional[int] = field(
        default=-1,
        metadata={
            "help": "Number of training examples to include from IcePaHC"
        },
    )
    sampling_weight_icepahc: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Weight of sampling IcePaHC data (between 0 and 1), note that the the probability of sampling also depends on number of samples"
        },
    )
    sampling_weight_greynir: Optional[float] = field(
        default=1.0,
        metadata={
            "help": "Weight of sampling Greynir data (between 0 and 1), note that the the probability of sampling also depends on number of samples"
        },
    )
    label_file: Optional[str] = field(
        default=None, metadata={"help": "Label dictionary file, analogous to fairseqs dict.txt"}
    )
    max_seq_len: Optional[int] = field(
        default=1024, metadata={"help": "Maximum number of words in parse sequence"}
    )
    # greynir_prefix: Optional[str] = field(default=None)
    # icepahc_prefix: Optional[str] = field(default=None)
    # mim_pos_prefix: Optional[str] = field(default=None)
    # ud_prefix: Optional[str] = field(default=None)
    # mim_ner_prefix: Optional[str] = field(default=None)
    # borrow from top level
    _bpe: Any = II("bpe")
    _dataset: Optional[Any] = II("dataset")
    _seed: int = II("common.seed")


@register_task("multi_incremental_parser_hydra", dataclass=MultiIncrementalParserHydraConfig)
class MultiIncrementalParserHydraTask(FairseqTask):
    cfg: MultiIncrementalParserHydraConfig

    def __init__(
        self,
        cfg: MultiIncrementalParserHydraConfig,
        data_dictionary: Dictionary,
        label_dictionary: Dictionary,
        is_word_initial,
    ):
        super().__init__(cfg)
        self.dictionary = data_dictionary

        if not hasattr(self, "args") and hasattr(self, "cfg"):
            self.args = self.cfg

        self.label_dictionary = label_dictionary

        self.bpe = encoders.build_bpe(self.cfg._bpe)
        self.is_word_initial = is_word_initial
        self.num_labels = len(self.label_dictionary)

    @classmethod
    def setup_task(cls, cfg: MultiIncrementalParserHydraConfig, **_kwargs):
        data_dict = cls.load_dictionary(cls, os.path.join(cfg.data, "dict.txt"))
        logger.info("[input] dictionary: {} types".format(len(data_dict)))
        label_dict = cls.load_dictionary(cls, cfg.label_file)
        label_dict.add_symbol(EOS_LABEL)
        label_dict.add_symbol(_GREYNIR_TASK_SYMBOL)
        label_dict.add_symbol(_ICEPAHC_TASK_SYMBOL)
        logger.info("[label] dictionary: {} types".format(len(label_dict)))

        assert cfg._bpe._name == cfg._parent.bpe._name

        is_word_initial = cls.get_word_beginnings(cfg._parent.bpe, data_dict)

        return MultiIncrementalParserHydraTask(
            cfg, data_dict, label_dictionary=label_dict, is_word_initial=is_word_initial
        )

    @classmethod
    def load_label_dictionary(cls, cfg: MultiIncrementalParserHydraConfig, filename: str, **_kwargs):
        """Load the dictionary from the filename
        Args:
            filename (str): the filename
        """
        assert Path(filename).exists(), f"Expected label_schema file at {filename}"
        label_schema = parse_label_schema(filename)

        return label_schema_as_dictionary(label_schema), label_schema

    @classmethod
    def load_dictionary(cls, cfg: MultiIncrementalParserHydraConfig, filename: str, add_mask: bool = True):
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

    def load_icepahc_dataset(self, path: str, *, dataset_id: int, use_augmentation: bool = False, limit=None):
        """Load a given dataset split (e.g., train, valid, test)."""
        trees_dataset = GreynirTreeJSONLDataset.from_path(path, limit=limit)

        if use_augmentation and self.cfg.case_punct_noise > 0.0:
            trees_dataset = GreynirTreeAugmentationDataset(trees_dataset, self.cfg.case_punct_noise, seed=self.cfg._seed)

        parsing_dataset = GreynirParsingDataset(
            trees_dataset,
            label_dictionary=self.label_dictionary,
            source_dictionary=self.source_dictionary,
            bpe=self.bpe,
            prepend_token=self.source_dictionary.bos(),
        )
        decomp = GreynirParsingDataset.decompose(parsing_dataset)

        # src_tokens = TextEncodingDataset(
        #     LambdaDataset(trees_dataset, lambda_fn=lambda x: x.text),
        #     self.source_dictionary,
        #     bpe=self.bpe,
        #     prepend_token=self.source_dictionary.bos(),
        # )

        src_tokens = decomp["src_tokens"]
        word_mask = WordEndMaskDataset(
            src_tokens, self.source_dictionary, self.is_word_initial, bos_value=0, eos_value=1
        )

        constant_dataset_id = ConstantDataset(dataset_id, collapse_collate=True)
        label_pad_idx = self.label_dictionary.pad()
        dataset = {
            "id": IdDataset(),
            "task_id": constant_dataset_id,
            "net_input": {
                "task_id": constant_dataset_id,
                "src_tokens": RightPadDataset(src_tokens, pad_idx=self.source_dictionary.pad()),
                "word_mask": RightPadDataset(word_mask, pad_idx=0),
                "preorder_nts": RightPadDataset(decomp["preorder_nts"], pad_idx=label_pad_idx),
                "preorder_depths": RightPad2dDataset(decomp["preorder_depths"], pad_idx=0),
                "preorder_mask": RightPad2dDataset(decomp["preorder_mask"], pad_idx=0),
                "chain_mask": RightPad2dDataset(decomp["chain_mask"], pad_idx=0),
                "preorder_spans": RightPad2dDataset(decomp["preorder_spans"], pad_idx=0),
                "preorder_flags": RightPad2dDataset(decomp["preorder_flags"], pad_idx=label_pad_idx),
                "nwords_per_step": RightPadDataset(decomp["nwords_per_step"], pad_idx=0),
            },
            "target_depths": RightPadDataset(decomp["target_depths"], pad_idx=label_pad_idx),
            "target_padding_mask": RightPadDataset(decomp["target_padding_mask"], pad_idx=1),
            "target_parents": RightPadDataset(decomp["target_parents"], pad_idx=label_pad_idx),
            "target_preterms": RightPadDataset(decomp["target_preterms"], pad_idx=label_pad_idx),
            "target_parent_flags": RightPad2dDataset(decomp["target_parent_flags"], pad_idx=label_pad_idx),
            "target_preterm_flags": RightPad2dDataset(decomp["target_preterm_flags"], pad_idx=label_pad_idx),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "nwords": NumWordsDataset(src_tokens, self.dictionary, self.is_word_initial),
        }

        dataset = NestedDictionaryDatasetFix(dataset, sizes=parsing_dataset.sizes)
        logger.info("Loaded {0} with #samples: {1}".format(path, len(dataset)))

        # dataset = NestedDictionaryDatasetFix(dataset, sizes=sizes)

        # with data_utils.numpy_seed(self.cfg._seed):
        #     shuffle = np.random.permutation(len(dataset))
        # dataset = SortDataset(dataset, sort_order=[shuffle])

        return dataset

    def load_greynir_dataset(self, path: str, *, dataset_id: int, use_augmentation: bool = False, limit=None):
        """Load a given dataset split (e.g., train, valid, test)."""
        trees_dataset = GreynirTreeJSONLDataset.from_path(path, limit=limit)
        # trees_dataset = GreynirTreeJSONLDataset.from_path(path)
        sizes = trees_dataset.sizes

        if use_augmentation and self.cfg.case_punct_noise > 0.0:
            trees_dataset = GreynirTreeAugmentationDataset(trees_dataset, self.cfg.case_punct_noise, seed=self.cfg._seed)

        parsing_dataset = GreynirParsingDataset(
            trees_dataset,
            label_dictionary=self.label_dictionary,
            source_dictionary=self.source_dictionary,
            bpe=self.bpe,
            prepend_token=self.source_dictionary.bos(),
            # word_initial=self.is_word_initial,
        )
        parsing_dataset[0]
        decomp = GreynirParsingDataset.decompose(parsing_dataset)

        # src_tokens = TextEncodingDataset(
        #     LambdaDataset(trees_dataset, lambda_fn=lambda x: x.text),
        #     self.source_dictionary,
        #     bpe=self.bpe,
        #     prepend_token=self.source_dictionary.bos(),
        # )

        src_tokens = decomp["src_tokens"]
        word_mask = WordEndMaskDataset(
            src_tokens, self.source_dictionary, self.is_word_initial, bos_value=0, eos_value=1
        )

        constant_dataset_id = ConstantDataset(dataset_id, collapse_collate=True)
        label_pad_idx = self.label_dictionary.pad()
        dataset = {
            "id": IdDataset(),
            "task_id": constant_dataset_id,
            "net_input": {
                "task_id": constant_dataset_id,
                "src_tokens": RightPadDataset(src_tokens, pad_idx=self.source_dictionary.pad()),
                "word_mask": RightPadDataset(word_mask, pad_idx=0),
                "preorder_nts": RightPadDataset(decomp["preorder_nts"], pad_idx=label_pad_idx),
                "preorder_depths": RightPad2dDataset(decomp["preorder_depths"], pad_idx=0),
                "preorder_mask": RightPad2dDataset(decomp["preorder_mask"], pad_idx=0),
                "chain_mask": RightPad2dDataset(decomp["chain_mask"], pad_idx=0),
                "preorder_spans": RightPad2dDataset(decomp["preorder_spans"], pad_idx=0),
                "preorder_flags": RightPad2dDataset(decomp["preorder_flags"], pad_idx=label_pad_idx),
                "nwords_per_step": RightPadDataset(decomp["nwords_per_step"], pad_idx=0),
            },
            "target_depths": RightPadDataset(decomp["target_depths"], pad_idx=label_pad_idx),
            "target_padding_mask": RightPadDataset(decomp["target_padding_mask"], pad_idx=1),
            "target_parents": RightPadDataset(decomp["target_parents"], pad_idx=label_pad_idx),
            "target_preterms": RightPadDataset(decomp["target_preterms"], pad_idx=label_pad_idx),
            "target_parent_flags": RightPad2dDataset(decomp["target_parent_flags"], pad_idx=label_pad_idx),
            "target_preterm_flags": RightPad2dDataset(decomp["target_preterm_flags"], pad_idx=label_pad_idx),
            "nsentences": NumSamplesDataset(),
            "ntokens": NumelDataset(src_tokens, reduce=True),
            "nwords": NumWordsDataset(src_tokens, self.dictionary, self.is_word_initial),
        }
        dataset = NestedDictionaryDatasetFix(dataset, sizes=parsing_dataset.sizes)
        # dataset = NestedDictionaryDatasetFix(dataset, sizes=sizes)
        logger.info("Loaded {0} with #samples: {1}".format(path, len(dataset)))

        # with data_utils.numpy_seed(self.cfg._seed):
        #     shuffle = np.random.permutation(len(dataset))
        # dataset = SortDataset(dataset, sort_order=[shuffle])

        return dataset

    def load_dataset(self, split: str, combine: bool = False, **_kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""

        # XXX: this approach is bad when NER, UD and POS tasks are added
        greynir_idx = self.label_dictionary.index(_GREYNIR_TASK_SYMBOL)
        icepahc_idx = self.label_dictionary.index(_ICEPAHC_TASK_SYMBOL)
        if "train" in split and self.cfg.sampling_weight_greynir > 0 and self.cfg.sampling_weight_icepahc > 0:
            greynir_path = Path(self.cfg.data) / f"{split}.greynir.jsonl"
            icepahc_path = Path(self.cfg.data) / f"{split}.icepahc.jsonl"
            # greynir_dataset = self.load_greynir_dataset(greynir_path, dataset_id=greynir_idx, use_augmentation=True, limit=50)
            # icepahc_dataset = self.load_greynir_dataset(icepahc_path, dataset_id=icepahc_idx, use_augmentation=True, limit=50)
            greynir_dataset = self.load_greynir_dataset(greynir_path, dataset_id=greynir_idx, use_augmentation=True, limit=self.cfg.limit_greynir)
            icepahc_dataset = self.load_greynir_dataset(icepahc_path, dataset_id=icepahc_idx, use_augmentation=True, limit=self.cfg.limit_icepahc)

            task_datasets = OrderedDict()
            # NOTE: MultitaskDatasetWrapper always shuffles
            task_datasets["greynir_parsing"] = MultitaskDatasetWrapper(
                greynir_dataset, sampling_weight=self.cfg.sampling_weight_greynir, name="greynir_parsing"
            )
            task_datasets["icepahc_parsing"] = MultitaskDatasetWrapper(
                icepahc_dataset, sampling_weight=self.cfg.sampling_weight_icepahc, name="icepahc_parsing"
            )
            _info = [
                ("greynir_parsing", self.cfg.sampling_weight_greynir, len(task_datasets["greynir_parsing"].ordered_indices())),
                ("icepahc_parsing", self.cfg.sampling_weight_icepahc, len(task_datasets["icepahc_parsing"].ordered_indices())),

            ]

            # from fairseq.data.multilingual.sampled_multi_epoch_dataset import SampledMultiEpochDataset
            # from fairseq.data.multilingual.sampled_multi_dataset import SampledMultiDataset, CollateFormat
            # breakpoint()
            # b /data/scratch/haukur/condaenvs/multiparser11.6/lib/python3.8/site-packages/fairseq/data/multilingual/sampled_multi_dataset.py:344
            # b /data/scratch/haukur/condaenvs/multiparser11.6/lib/python3.8/site-packages/fairseq/data/multilingual/sampled_multi_dataset.py:366
            # foo = SampledMultiDataset([greynir_dataset, icepahc_dataset], sampling_ratios=[0.5, 0.5], virtual_size=10000, collate_format=CollateFormat.ordered_dict)
            # self.datasets[split] = foo
            # return

            total = sum(length for name, weight, length in _info)
            for name, weight, length in _info:
                logger.info(f"Using {name} with downsampling ratio={weight}, #samples={length} in epoch, mixing ratio={100*length/total:2.1f}%")

            self.datasets[split] = task_datasets
        elif "train" in split and self.cfg.sampling_weight_greynir > 0:
            greynir_path = Path(self.cfg.data) / f"{split}.greynir.jsonl"
            greynir_dataset = self.load_greynir_dataset(greynir_path, dataset_id=greynir_idx, use_augmentation=True, limit=self.cfg.limit_greynir)
            self.datasets[split] = greynir_dataset
        elif "train" in split and self.cfg.sampling_weight_icepahc > 0:
            icepahc_path = Path(self.cfg.data) / f"{split}.icepahc.jsonl"
            icepahc_dataset = self.load_greynir_dataset(icepahc_path, dataset_id=icepahc_idx, use_augmentation=True, limit=self.cfg.limit_icepahc)
            self.datasets[split] = icepahc_dataset
        else:
            if "greynir" in split:
                dataset = self.load_greynir_dataset(Path(self.cfg.data) / f"{split}.jsonl", dataset_id=greynir_idx, use_augmentation=False, limit=1000)
            else:
                dataset = self.load_icepahc_dataset(Path(self.cfg.data) / f"{split}.jsonl", dataset_id=icepahc_idx, use_augmentation=False, limit=1000)
            self.datasets[split] = dataset

    # def build_model(self, model_cfg: FairseqDataclass, from_checkpoint=False):
    #     breakpoint()
    #     return super().build_model(model_cfg, from_checkpoint)

    def dataset(self, split: str):
        # this is the same as the super's method, except without a typecheck assertion
        # since our datasets are self.datasets[split] is an OrderedDict
        if split not in self.datasets:
            raise KeyError("Dataset not loaded: " + split)
        return self.datasets[split]

    def prepare_sentences(self, sentences: List[str]):
        src_tokens = TextEncodingDataset(
            sentences, self.source_dictionary, bpe=self.bpe, prepend_token=self.source_dictionary.bos()
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
                # tree = trees[seq_idx]
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

    # this function is heavily based on one from fairseq (fairseq/examples/laser/laser_src/laser_task.py)
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
        data_buffer_size=0,
        disable_iterator_cache=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
        **kwargs,
    ):
        # validation/test datasets are not multitask, we handle them normally
        if isinstance(dataset, FairseqDataset):
            assert ignore_invalid_inputs
            return super().get_batch_iterator(
                dataset,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                max_positions=min(max_positions or self.cfg.max_seq_len, self.cfg.max_seq_len),
                ignore_invalid_inputs=ignore_invalid_inputs,
                required_batch_size_multiple=required_batch_size_multiple,
                seed=seed,
                num_shards=num_shards,
                shard_id=shard_id,
                num_workers=num_workers,
                epoch=epoch,
                data_buffer_size=data_buffer_size,
                disable_iterator_cache=disable_iterator_cache,
                grouped_shuffling=grouped_shuffling,
                update_epoch_batch_itr=update_epoch_batch_itr,
                **kwargs,
            )
        assert isinstance(dataset, OrderedDict)
        assert len(dataset)
        assert isinstance(dataset[next(iter(dataset))], FairseqDataset)

        import time
        start = time.time()
        # initialize the dataset with the correct starting epoch
        for _, dt in dataset.items():
            dt.set_epoch(epoch)

        indices = OrderedDict()
        batch_sampler = OrderedDict()

        with data_utils.numpy_seed(seed, epoch):
            for key, dt in dataset.items():
                logger.info(f"\t ordered_indices {key}")
                indices[key] = dt.ordered_indices()

        # filter examples that are too large
        if max_positions is not None:
            for key, dt in dataset.items():
                logger.info(f"\t filter_by_size {key}")
                indices[key], _ignored = dt.filter_indices_by_size(indices[key], min(max_positions, self.cfg.max_seq_len))

        for key, dt in dataset.items():
            logger.info(f"\t batch_by_size {key}")
            batch_sampler[key] = data_utils.batch_by_size(
                indices[key],
                dt.num_tokens,
                max_tokens=max_tokens,
                max_sentences=max_sentences,
                required_batch_size_multiple=required_batch_size_multiple,
            )

        epoch_iter = MultidatasetEpochBatchIterator(
            dataset=dataset,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
        )

        end = time.time()
        print(f"Preparing next epoch took {end - start:8.2f}")
        return epoch_iter
