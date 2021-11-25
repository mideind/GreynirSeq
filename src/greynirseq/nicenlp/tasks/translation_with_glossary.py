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
from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional

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


@dataclass
class GlossaryTaskConfig:
    """ Configuration for the glossary task."""

    enabled: bool
    ok_to_increase_dict_size: bool
    constraint_positional_shift: int
    seq_sample_ratio: float
    mean_whole_word: float
    stddev_whole_word: float

    @staticmethod
    def from_args(args):
        """Create a config from the args object. Call this in task __init__"""
        # It is not ok to increase the dictionary size if we are loading a pretrained model.
        # TODO: verify that this is the case.
        ok_to_increase_dict_size = args.restore_file is None
        return GlossaryTaskConfig(
            enabled=args.glossary_enabled,
            ok_to_increase_dict_size=ok_to_increase_dict_size,
            constraint_positional_shift=args.glossary_constraint_positional_shift,
            seq_sample_ratio=args.glossary_seq_sample_ratio,
            mean_whole_word=args.glossary_mean_whole_word,
            stddev_whole_word=args.glossary_stddev_whole_word,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """Add arguments to the parser for this task. Call this in the task's add_args()."""
        parser.add_argument(
            "--glossary-enabled", default=False, action="store_true", help="Should the glossary task be enabled?"
        )
        parser.add_argument(
            "--glossary-constraint-positional-shift",
            default=1024,
            type=int,
            help="By how much should we shift the positional encoding of the glossary constraints?",
        )
        parser.add_argument(
            "--glossary-seq-sample-ratio",
            default=1.0,
            type=float,
            help="The ratio of training sequences which we will add constraints to.",
        )
        parser.add_argument(
            "--glossary-mean-whole-word",
            default=1.0,
            type=float,
            help="When sampling target whole words using a normal distribution, the mean of the distribution.",
        )
        parser.add_argument(
            "--glossary-stddev-whole-word",
            default=2.0,
            type=float,
            help="When sampling target whole words using a normal distribution, the stddev of the distribution.",
        )


def make_positions_with_constraints(
    tensor: Tensor, padding_idx: int, onnx_trace: bool = False, shift_amount=0, shift_from_symbol=None
):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.

    This code is based on fairseq's make_positions.
    Addition includes shifting all symbols from and with the "shift_from_symbol" by "shift_amount".
    Comment makes it is clear what changes are added.

    """
    # The series of casts and type-conversions here are carefully
    # balanced to both work with ONNX export and XLA. In particular XLA
    # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
    # how to handle the dtype kwarg in cumsum.
    mask = tensor.ne(padding_idx).int()
    # CHANGES START HERE
    positions = torch.cumsum(mask, dim=1).type_as(mask)

    if shift_from_symbol is not None and shift_amount > 0:
        # This makes sure that the constraints are not simply added at the end of the target.
        # Find if the shift_symbol is present in the tensor
        shift_symbol_idxs = tensor.eq(shift_from_symbol).nonzero()
        # TODO: maybe we can do this directly using tensor operations
        # Using positions[shift_symbol_idxs] += shift_amount only adds the shift to a single element
        for shift_symbol_idx in shift_symbol_idxs:
            positions[int(shift_symbol_idx[0].item()), int(shift_symbol_idx[1].item()) :] += shift_amount
    positions = positions * mask + padding_idx
    return positions.long()


def apply_monkey_patch_for_make_positions(shift_from_symbol: int, shift_amount: int):
    """Monkey-patch the make_positions function in fairseq to be aware of the glossary constraints.
    The monkey patch only extends the original function."""
    from fairseq import utils  # pylint: disable=import-outside-toplevel

    utils.make_positions = partial(
        make_positions_with_constraints, shift_from_symbol=shift_from_symbol, shift_amount=shift_amount
    )


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

        apply_monkey_patch_for_make_positions(
            shift_from_symbol=self.source_dictionary.index("<sep>"),
            shift_amount=self.glossary_task_config.constraint_positional_shift,
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
        if not self.glossary_task_config.enabled:
            return
        is_train_subset = split == getattr(self.args, "train_subset", None)
        if not is_train_subset:
            # We are not training - no need to add training constraints
            return
        # We make sure that the loaded ds is LanguagePairDataset - since we make assumptions about the ds structure.
        ds = self.datasets[split]
        assert isinstance(ds, LanguagePairDataset)
        src_dataset = TargetSamplingWholeWordDataset(
            src_dataset=ds.src,
            tgt_dataset=ds.tgt,
            whole_word_masker=self.is_word_initial,
            seq_sample_ratio=self.glossary_task_config.seq_sample_ratio,
            mean_whole_word=self.glossary_task_config.mean_whole_word,
            stddev_whole_word=self.glossary_task_config.stddev_whole_word,
            sep_symbol_idx=self.target_dictionary.index("<sep>"),
            constraint_symbol_idx=self.target_dictionary.index("<c>"),
            pad_idx=self.target_dictionary.index("<pad>"),
            max_seq_len=self.args.max_source_positions,
            bpe=self.bpe,
        )
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


class TargetSamplingWholeWordDataset(BaseWrapperDataset):
    def __init__(
        self,
        src_dataset: BaseWrapperDataset,
        tgt_dataset: BaseWrapperDataset,
        whole_word_masker: Dict[int, int],
        seq_sample_ratio: float,
        mean_whole_word: float,
        stddev_whole_word: float,
        sep_symbol_idx: int,
        constraint_symbol_idx: int,
        pad_idx: int,
        max_seq_len: int,
        bpe,
    ):
        super().__init__(src_dataset)
        self.tgt_dataset = tgt_dataset
        self.whole_word_masker = whole_word_masker
        self.seq_sample_ratio = seq_sample_ratio
        self.mean_whole_word = mean_whole_word
        self.stddev_whole_word = stddev_whole_word
        self.sep_symbol_idx = sep_symbol_idx
        self.constraint_symbol_idx = constraint_symbol_idx
        self.pad_idx = pad_idx
        self.max_seq_len = max_seq_len
        # Just for debugging now
        self.bpe = bpe

    def __getitem__(self, idx):
        # Dim = (seq_len)
        tgt: Tensor = self.tgt_dataset[idx]
        src: Tensor = self.dataset[idx]
        # For debugging
        # logger.info(f"tgt {tgt}")
        # logger.info(f"idx {idx}")
        # logger.info(f"src: {self.bpe.decode(' '.join(str(x) for x in src.tolist()))}")
        # logger.info(f"tgt: {self.bpe.decode(' '.join(str(x) for x in tgt.tolist()))}")
        constraints = whole_word_target_sampling(
            tgt,
            self.whole_word_masker,
            self.seq_sample_ratio,
            self.mean_whole_word,
            self.stddev_whole_word,
            contains_eos=True,
        )
        constraint_src = apply_constraints(constraints, src, self.constraint_symbol_idx, self.sep_symbol_idx)
        constraint_src = constraint_src[: self.max_seq_len]
        # logger.info(
        #     f"constraint_src: {self.bpe.decode(' '.join(str(x) for x in constraint_src.tolist() if x <= 32000))}"
        # )
        return constraint_src


def apply_constraints(constraints: List[Tensor], src: Tensor, constraint_idx: int, sep_idx: int) -> Tensor:
    """
    Apply constraints to a source sequence.
    """
    src = torch.cat((src, torch.Tensor([sep_idx]).long()), dim=0)
    for constraint in constraints:
        src = torch.cat((src, constraint, torch.Tensor([constraint_idx]).long()), dim=0)
    return src


def whole_word_target_sampling(
    tgt: Tensor,
    whole_word_masker: Dict[int, int],
    seq_sample_ratio: float,
    mean_whole_word: float,
    stddev_whole_word: float,
    contains_eos: bool,
) -> List[Tensor]:
    """
    Create target sampling constraints.
    """
    fraction_sequences_to_constrain = seq_sample_ratio
    mean_whole_words_as_constraints = mean_whole_word
    std_dev_whole_words_as_contraints = stddev_whole_word
    # Should we use this example as a constraint?
    use_example_as_constraint = random.random() < fraction_sequences_to_constrain
    if not use_example_as_constraint:
        return []
    if contains_eos:
        # We need to remove the eos symbol from the target sequence
        tgt = tgt[:-1]

    tgt_whole_word_mask = torch.Tensor([whole_word_masker[elm] for elm in tgt.tolist()]).long()
    # The number whole words in the target
    tgt_whole_word_count = int(tgt_whole_word_mask.sum().item())
    # The number of whole words in the target that we want to use as contraints
    tgt_contraints_whole_word_counts = int(
        (
            torch.normal(
                mean_whole_words_as_constraints, std_dev_whole_words_as_contraints, size=(1,)
            )  # Single point from a Normal distribution
            .round()
            .int()  # Round to integers
            .clamp(
                min=0, max=tgt_whole_word_count
            )  # Map negative values to 0 and values greater than the whole word count to the whole word count
        ).item()
    )
    # We sample the indices of the whole words we want to use.
    sampled_whole_word_orders = torch.Tensor(
        sorted(list(random.sample(range(tgt_whole_word_count), tgt_contraints_whole_word_counts)))
    ).long()
    # The number of subwords in the whole words
    whole_word_lengths = masks_lengths(tgt_whole_word_mask)
    # The indices of the whole words in the original target
    whole_word_idxs = tgt_whole_word_mask.nonzero().squeeze()
    sampled_whole_word_idxs = whole_word_idxs[sampled_whole_word_orders].tolist()
    sampled_whole_word_lengths = whole_word_lengths[sampled_whole_word_orders].tolist()
    sampled_whole_words = [
        tgt[whole_word_idx : whole_word_idx + whole_word_length]
        for (whole_word_idx, whole_word_length) in zip(sampled_whole_word_idxs, sampled_whole_word_lengths)
    ]
    return sampled_whole_words


def masks_lengths(mask: Tensor) -> Tensor:
    """Return the masks length, i.e. a mask=[1,0,0,1,1,0] should return [3,1,2]"""
    # TODO: This implementation could be improved.
    mask_idxs = mask.nonzero().squeeze()  # For some reason, nonzero returns a tensor of size (1,2)
    lengths = []
    prev_idx: Optional[int] = None
    for idx in mask_idxs.tolist():
        if prev_idx is None:
            prev_idx = idx
            continue
        lengths.append(idx - prev_idx)
        prev_idx = idx
    if prev_idx is not None:
        lengths.append(int(mask.shape[0]) - prev_idx)
    return torch.Tensor(lengths).long()
