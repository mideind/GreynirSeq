"""Translation with Glossary Task. Enables a model to leverage glossaries during translation."""
#
# Based on fairseq/tasks/translation.py that has the following license
#
#    Copyright (c) Facebook, Inc. and its affiliates.
#
#    This source code is licensed under the MIT license found in the
#    LICENSE file in the root directory of this source tree.

import argparse
import logging
import random
import re
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import jaro
import torch
from fairseq.data import Dictionary, encoders
from fairseq.data.base_wrapper_dataset import BaseWrapperDataset
from fairseq.data.language_pair_dataset import LanguagePairDataset
from fairseq.tasks import register_task
from torch.functional import Tensor

from greynirseq.nicenlp.data.encoding import get_word_beginnings
from greynirseq.nicenlp.tasks.translation_with_backtranslation import TranslationWithBacktranslationTask
from greynirseq.nicenlp.utils.dictionary import ensure_symbols_are_present

logger = logging.getLogger(__name__)

non_letters_numbers = re.compile(r"\W", flags=re.UNICODE | re.IGNORECASE)


@dataclass
class GlossaryTaskConfig:
    """ Configuration for the glossary task."""

    enabled: bool
    ok_to_increase_dict_size: bool
    constraint_positional_start_idx: int
    seq_sample_ratio: float
    mean_whole_word: float
    stddev_whole_word: float
    glossary_file: str
    glossary_subset_prefix: str
    fuzzy_match_threshold: int

    @staticmethod
    def from_args(args):
        """Create a config from the args object. Call this in task __init__"""
        # It is not ok to increase the dictionary size if we are loading a pretrained model.
        # TODO: verify that this is the case.
        ok_to_increase_dict_size = args.restore_file is None
        return GlossaryTaskConfig(
            enabled=args.glossary_enabled,
            ok_to_increase_dict_size=ok_to_increase_dict_size,
            constraint_positional_start_idx=args.glossary_constraint_positional_start_idx,
            seq_sample_ratio=args.glossary_seq_sample_ratio,
            mean_whole_word=args.glossary_mean_whole_word,
            stddev_whole_word=args.glossary_stddev_whole_word,
            glossary_file=args.glossary_lookup_file,
            glossary_subset_prefix=args.glossary_glosspref,
            fuzzy_match_threshold=args.glossary_fuzzy_match_threshold,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """Add arguments to the parser for this task. Call this in the task's add_args()."""
        parser.add_argument(
            "--glossary-enabled",
            default=False,
            action="store_true",
            help="Should the glossary task be enabled? \
This will add soft constraints automatically during training and \
their effectiveness can be evaluated given a glossary.",
        )
        parser.add_argument(
            "--glossary-seq-sample-ratio",
            default=0.2,
            type=float,
            help="During training, soft constraints are automatically added to this fraction of sequences.",
        )
        parser.add_argument(
            "--glossary-mean-whole-word",
            default=1.0,
            type=float,
            help="During training, soft constraints are sampled from the target. \
The sampling selects whole-words (potentially multiple subwords). \
The sampling distribution is the normal distribution and this parameter is the mean of that distribution.",
        )
        parser.add_argument(
            "--glossary-stddev-whole-word",
            default=2.0,
            type=float,
            help="During training, soft constraints are sampled from the target. \
The sampling selects whole-words (potentially multiple subwords). \
The sampling distribution is the normal distribution and this parameter is the stddev of that distribution.",
        )
        parser.add_argument(
            "--glossary-constraint-positional-start-idx",
            default=0,
            type=int,
            help="When constraints are added as a part of the input, their positional indices are changed by the task. \
This parameter sets the starting index of the positional indices of the constraints. \
Starting from 0 will 'restart' the positional indicies for the constraints.",
        )
        parser.add_argument(
            "--glossary-lookup-file",
            default="",
            type=str,
            help="The file which defines the glossary/lookup which should be used when evaluating the glossary functionality. \
The file should be in DATA_DIR, i.e. a relative path. \
This file is expected to be a tsv file with SRC *tab* TGT, where TGT is the translation of SRC. \
Currently, we only support single word SRCs. \
The glossary is used when we translate the validation/test subset defined in --glossary-glosspref. \
We perform a fuzzy search for SRC in the input. \
If the fuzzy match is above --glossary-fuzzy-match-threshold the corresponding TGT is added as a constraint.",
        )
        parser.add_argument(
            "--glossary-glosspref",
            default="",
            type=str,
            help="The subset name of the validation or testing data which should use the --glossary-lookup-file. \
See --glossary-lookup-file help for more details",
        )
        parser.add_argument(
            "--glossary-fuzzy-match-threshold",
            default=100,
            type=int,
            help="The fuzzy match threshold when searching for SRC in the input. \
If the fuzzy match is greater than the threshold, we append the corresponding glossary TGT as constraints. \
The threshold is in [0, 100], 100 means exact match. \
See --glossary-lookup-file help for more details",
        )


def make_positions_with_constraints(
    tensor: Tensor,
    padding_idx: int,
    onnx_trace: bool = False,
    positional_idx_restart_offset=0,
    positional_marker_symbol_idx: Optional[int] = None,
):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.

    This code is based on fairseq's make_positions.
    The change allows the positional indices to be restarted when encountering the 'positional_marker_symbol_idx'.
    The positional indices after the restart are additionally offset by the 'positional_idx_restart_offset'.
    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    padding_mask = tensor.ne(padding_idx).int()
    positional_marker_mask = torch.ones_like(padding_mask)
    positional_marker_idx = torch.zeros_like(padding_mask).bool().nonzero()
    if positional_marker_symbol_idx is not None:
        positional_marker_idx = tensor.eq(positional_marker_symbol_idx).nonzero()
    # TODO: find a way to do this efficently using tensors
    for marker_idx in positional_marker_idx:
        positional_marker_mask[marker_idx[0], marker_idx[1] :] = 0
    before_positional_marker_mask = positional_marker_mask
    after_positional_marker_mask = (~positional_marker_mask.bool()).int()  # int -> bool -> negation -> int again

    # First the positions before the positional_marker_mask
    before_positional_marker_mask_and_padding = before_positional_marker_mask * padding_mask
    positions = (
        torch.cumsum(before_positional_marker_mask_and_padding, dim=1).type_as(padding_mask)
        * before_positional_marker_mask_and_padding
    )
    # Then the positions after the positional_marker_mask
    after_positional_marker_mask_and_padding = after_positional_marker_mask * padding_mask
    positions += (
        torch.cumsum(after_positional_marker_mask_and_padding, dim=1).type_as(padding_mask)
        + positional_idx_restart_offset
    ) * after_positional_marker_mask_and_padding
    positions += padding_idx
    return positions.long()


def apply_monkey_patch_for_make_positions(positional_marker_symbol_idx: int, positional_idx_restart_offset: int):
    """Monkey-patch the make_positions function in fairseq to be aware of the glossary constraints.
    The monkey patch only extends the original function."""
    from fairseq import utils  # pylint: disable=import-outside-toplevel

    utils.make_positions = partial(
        make_positions_with_constraints,
        positional_marker_symbol_idx=positional_marker_symbol_idx,
        positional_idx_restart_offset=positional_idx_restart_offset,
    )


def read_glossary(path: Path) -> Dict[str, str]:
    """Read the glossary from the given file. Return an empty dict if it does not exist."""
    if not path.exists():
        return {}
    with path.open("r") as f:
        return {line.split("\t")[0]: line.split("\t")[1].strip() for line in f}


@register_task("translation_with_glossary")
class TranslationWithGlossaryTask(TranslationWithBacktranslationTask):
    """
    Translate from one (source) language to another (target) language with soft constraints.

    """

    def __init__(
        self,
        args: argparse.Namespace,
        src_dict: Dictionary,
        tgt_dict: Dictionary,
    ):
        super().__init__(args, src_dict, tgt_dict)  # type: ignore
        config = GlossaryTaskConfig.from_args(args)
        if config.enabled:
            logger.info("Glossary is ENABLED")
            logger.info(f"Glossary config: {config}")
        else:
            logger.info("Glossary is DISABLED")
        self.glossary_task_config = config
        # Ensure that <sep> and <c> are defined in the dictionaries.
        ensure_symbols_are_present(
            self.source_dictionary,
            ["<c>", "<sep>"],
            self.glossary_task_config.ok_to_increase_dict_size,
        )
        ensure_symbols_are_present(
            self.target_dictionary,
            ["<c>", "<sep>"],
            self.glossary_task_config.ok_to_increase_dict_size,
        )
        assert (
            self.target_dictionary == self.source_dictionary
        ), "The target dictionary must be the same as the source dictionary, \
    because we use is_word_initial based on a single dictionary and use it for both src and tgt."
        is_word_initial = get_word_beginnings(args, self.source_dictionary)
        if is_word_initial is None:
            raise ValueError("The is_word_initial function is None.")
        self.is_word_initial = is_word_initial

        # Get the data path
        data_path = args.data.split(":")[0]
        self.glossary = read_glossary(Path(data_path) / self.glossary_task_config.glossary_file)

        apply_monkey_patch_for_make_positions(
            positional_marker_symbol_idx=self.source_dictionary.index("<sep>"),
            positional_idx_restart_offset=self.glossary_task_config.constraint_positional_start_idx,
        )
        self.bpe = encoders.build_bpe(args)

    @staticmethod
    def add_args(parser) -> None:
        """Add task-specific arguments to the parser."""
        super(TranslationWithGlossaryTask, TranslationWithGlossaryTask).add_args(parser)  # type: ignore
        GlossaryTaskConfig.add_args(parser)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs) -> None:
        """Load a given dataset split."""
        # Load the super stuff
        super().load_dataset(split=split, epoch=epoch, combine=combine, **kwargs)
        logger.info(f"Split: {split}")
        if not self.glossary_task_config.enabled:
            return
        is_glossary_pref = split == self.glossary_task_config.glossary_subset_prefix
        is_train_subset = split == getattr(self.args, "train_subset", None)

        # We make sure that the loaded ds is LanguagePairDataset - since we make assumptions about the ds structure.
        ds = self.datasets[split]
        src_dataset = ds.src
        assert isinstance(ds, LanguagePairDataset)
        if is_train_subset:
            logger.info("Training subset. Sampling whole words from target")
            # We sample whole words from the target side to use as constraints.
            constraints = SampleWholeWordDataset(
                dataset=ds.tgt,
                whole_word_masker=self.is_word_initial,
                seq_sample_ratio=self.glossary_task_config.seq_sample_ratio,
                mean_whole_word=self.glossary_task_config.mean_whole_word,
                stddev_whole_word=self.glossary_task_config.stddev_whole_word,
                pad_idx=self.target_dictionary.index("<pad>"),
                bpe=self.bpe,
            )
            logger.info("Training subset. Appending constraints to SRC")
            # We append the constraints to the source.
            src_dataset = AppendConstraintsDataset(
                dataset=ds.src,
                constraints=constraints,
                sep_symbol_idx=self.target_dictionary.index("<sep>"),
                constraint_symbol_idx=self.target_dictionary.index("<c>"),
                max_seq_len=self.args.max_source_positions,
                bpe=self.bpe,
            )
        elif is_glossary_pref:
            logger.info("Glossary subset. Fuzzy matching glossary SRC to input.")
            constraints = FuzzyGlossaryConstraintsDataset(
                dataset=ds.src,
                glossary=self.glossary,
                fuzzy_match_threshold=self.glossary_task_config.fuzzy_match_threshold,
                bpe=self.bpe,
            )
            src_dataset = AppendConstraintsDataset(
                dataset=ds.src,
                constraints=constraints,
                sep_symbol_idx=self.target_dictionary.index("<sep>"),
                constraint_symbol_idx=self.target_dictionary.index("<c>"),
                max_seq_len=self.args.max_source_positions,
                bpe=self.bpe,
            )
            logger.info("Glossary subset. Appending constraints to SRC")

        self.datasets[split] = LanguagePairDataset(
            src_dataset,
            ds.src.sizes,
            ds.src_dict,
            ds.tgt,
            ds.tgt.sizes,
            ds.tgt_dict,
            left_pad_source=ds.left_pad_source,
            left_pad_target=ds.left_pad_target,
            align_dataset=ds.align_dataset,
            eos=ds.eos,
            num_buckets=self.args.num_batch_buckets,
            shuffle=ds.shuffle,
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        """Build a dataset used for inference."""
        return super().build_dataset_for_inference(src_tokens, src_lengths, constraints=None)  # type: ignore


class FuzzyGlossaryConstraintsDataset(BaseWrapperDataset):
    """A Dataset class which fuzzy matches glossary entries to 'dataset'.

    TODO: More explaination"""

    def __init__(self, dataset, glossary: Dict[str, str], fuzzy_match_threshold: int, bpe):
        super().__init__(dataset)
        self.glossary = glossary
        self.fuzzy_match_threshold = fuzzy_match_threshold
        self.bpe = bpe

    def __getitem__(self, idx):
        # Dim = (seq_len)
        example: Tensor = self.dataset[idx]
        # TODO: Remove assumption that the fairseq source dictionary is irrelevant.
        sentence = self.bpe.decode(" ".join(str(x) for x in tensor_to_list(example)))
        fuzzy_matches = fuzzy_match_glossary(sentence, glossary=self.glossary)
        glossary_constraints = [x[0] for x in filter(lambda x: x[1] >= self.fuzzy_match_threshold, fuzzy_matches)]
        constraints = [
            torch.Tensor([int(x) for x in self.bpe.encode(constraint).split(" ")]).long()
            for constraint in glossary_constraints
        ]
        logger.info(f"Original sentence: {sentence}\nConstraints: {constraints}")
        return constraints


class AppendConstraintsDataset(BaseWrapperDataset):
    """A Dataset class which appends constraints to 'dataset'.

    A separator symbol is inserted between the original example and the constraints.
    Between each constraint, a constraint symbol is inserted.
    If the example goes over the max_seq_len, the example is truncated.
    Assumes that the 'constraints' dataset returns a List[Tensor] when indexed."""

    def __init__(
        self,
        dataset: BaseWrapperDataset,
        constraints: BaseWrapperDataset,
        sep_symbol_idx: int,
        constraint_symbol_idx: int,
        max_seq_len: int,
        bpe,
    ):
        super().__init__(dataset)
        self.constraints = constraints
        self.sep_symbol_idx = sep_symbol_idx
        self.constraint_symbol_idx = constraint_symbol_idx
        self.max_seq_len = max_seq_len
        # Just for debugging now
        self.bpe = bpe

    def __getitem__(self, idx):
        # Dim = (seq_len)
        example: Tensor = self.dataset[idx]
        constraints: List[Tensor] = self.constraints[idx]
        random.shuffle(constraints)
        constraint_src = apply_constraints(constraints, example, self.constraint_symbol_idx, self.sep_symbol_idx)
        constraint_src = constraint_src[: self.max_seq_len]
        # logger.info(
        #     f"constraint_src: {self.bpe.decode(' '.join(str(x) for x in constraint_src.tolist() if x <= 32000))}"
        # )
        return constraint_src


class SampleWholeWordDataset(BaseWrapperDataset):
    """A Dataset class which samples whole words from 'dataset'.

    For each example, we sample a random number (0, 1] and if it is less than the 'seq_sample_ratio'
    then we will sample whole words from that example.
    The whole words samples are drawn based on a normal distribution.
    The mean and standard deviation are given by the 'mean_whole_word' and 'stddev_whole_word' parameters.
    This Dataset returns List[Tensor] when indexed."""

    def __init__(
        self,
        dataset: BaseWrapperDataset,
        whole_word_masker: Dict[int, int],
        seq_sample_ratio: float,
        mean_whole_word: float,
        stddev_whole_word: float,
        pad_idx: int,
        bpe,
    ):
        super().__init__(dataset)
        self.whole_word_masker = whole_word_masker
        self.seq_sample_ratio = seq_sample_ratio
        self.mean_whole_word = mean_whole_word
        self.stddev_whole_word = stddev_whole_word
        self.pad_idx = pad_idx
        # Just for debugging now
        self.bpe = bpe

    def __getitem__(self, idx):
        # Dim = (seq_len)
        example: Tensor = self.dataset[idx]
        # For debugging
        # logger.info(f"example {example}")
        # logger.info(f"idx {idx}")
        # Pure Python for a single number is faster
        random_whole_word_count = random.gauss(self.mean_whole_word, self.stddev_whole_word)
        constraints = whole_word_sampling(
            example,
            self.whole_word_masker,
            self.seq_sample_ratio,
            word_count_to_sample=random_whole_word_count,
            contains_eos=True,
        )
        return constraints


def apply_constraints(constraints: List[Tensor], t: Tensor, constraint_idx: int, sep_idx: int) -> Tensor:
    """
    Apply constraints to a source sequence.
    """
    # t = t_1, t_2, ..., t_n (t_i are subwords)
    # t = t + <sep>
    t = torch.cat((t, torch.Tensor([sep_idx]).long()), dim=0)
    for constraint in constraints:
        # t = t + c_i + <c>
        t = torch.cat((t, constraint, torch.Tensor([constraint_idx]).long()), dim=0)
    return t


def whole_word_sampling(
    example: Tensor,
    whole_word_masker: Dict[int, int],
    seq_sample_ratio: float,
    word_count_to_sample: float,
    contains_eos: bool,
) -> List[Tensor]:
    """
    Create sampling constraints.
    """
    sampled_whole_words: List[Tensor] = []
    if contains_eos:
        # We need to remove the eos symbol from the target sequence
        example = example[:-1]
    tgt_whole_word_mask = [whole_word_masker[elm] for elm in tensor_to_list(example)]
    # The number whole words in the target
    tgt_whole_word_count = sum(tgt_whole_word_mask)
    # The number of whole words in the target that we want to use as contraints
    word_count_to_sample = round(min(max(0, word_count_to_sample), tgt_whole_word_count))

    fraction_sequences_to_constrain = seq_sample_ratio
    # Should we use this example as a constraint?
    use_example_as_constraint = random.random() < fraction_sequences_to_constrain
    if not use_example_as_constraint and word_count_to_sample == 0:
        return sampled_whole_words
    # We sample the indices of the whole words we want to use.
    sampled_idxs = sorted(list(random.sample(range(tgt_whole_word_count), word_count_to_sample)))
    # The number of subwords in the whole words
    tgt_whole_word_lengths = whole_word_lengths(tgt_whole_word_mask)
    # The indices of the whole words in the original target
    tgt_whole_word_start_idxs = tensor_to_list(torch.Tensor(tgt_whole_word_mask).nonzero().squeeze())
    for sampled_idx in sampled_idxs:
        tgt_whole_word_start_idx = tgt_whole_word_start_idxs[sampled_idx]
        tgt_whole_word_length = tgt_whole_word_lengths[sampled_idx]
        sampled_whole_words.append(example[tgt_whole_word_start_idx : tgt_whole_word_start_idx + tgt_whole_word_length])
    return sampled_whole_words


def whole_word_lengths(whole_word_mask: List[int]) -> List[int]:
    """Return the masks length, i.e. a padding_mask=[1,0,0,1,1,0] should return [3,1,2]"""
    lengths: List[int] = []
    if len(whole_word_mask) == 0:
        return lengths
    mask_idxs_ten = (
        torch.Tensor(whole_word_mask).nonzero().squeeze()
    )  # For some reason, nonzero returns a tensor of size (1,2)
    mask_idxs = tensor_to_list(mask_idxs_ten)
    prev_idx: Optional[int] = None
    for idx in mask_idxs:
        if prev_idx is None:
            prev_idx = idx
            continue
        lengths.append(idx - prev_idx)
        prev_idx = idx
    if prev_idx is not None:
        lengths.append(len(whole_word_mask) - prev_idx)
    return lengths


Translation = str
Source = str
Original = str
FuzzyResults = List[Tuple[Translation, float, Source, Original]]


def fuzzy_match_glossary(sentence: str, glossary: Dict[str, str], token_limit=5) -> FuzzyResults:
    """
    Return a list of glossary translations that are fuzzy matches of the sentence.

    Threshold is an integer between 0 and 100.
    """
    glossary_translations: FuzzyResults = []

    # Since we just 'split', some tokens will contain punctuation, we remove it.
    sentence = re.sub(non_letters_numbers, " ", sentence)
    for token in sentence.split():
        token_matches: FuzzyResults = []
        for key, value in glossary.items():
            score = jaro.jaro_winkler_metric(token, key)
            token_matches.append((value, score, key, token))
        token_matches.sort(key=lambda x: x[1], reverse=True)
        glossary_translations.extend(token_matches[:token_limit])
    return glossary_translations


def tensor_to_list(tensor: Tensor) -> List:
    """
    Convert a tensor to a list.

    tensor.tolist() returns a list of numbers OR a single number if the tensor only contains one element.
    """
    t = tensor.tolist()
    if isinstance(t, list):
        return t
    return [t]
