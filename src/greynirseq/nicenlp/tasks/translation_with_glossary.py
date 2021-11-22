#
# Based on fairseq/tasks/translation.py that has the following license
#
#    Copyright (c) Facebook, Inc. and its affiliates.
#
#    This source code is licensed under the MIT license found in the
#    LICENSE file in the root directory of this source tree.

import argparse
import enum
import logging
from dataclasses import dataclass
from functools import partial
from typing import Callable, List

import torch
from fairseq.data import Dictionary
from fairseq.tasks import register_task
from torch.functional import Tensor

from greynirseq.nicenlp.data.encoding import get_word_beginnings
from greynirseq.nicenlp.tasks.translation_with_backtranslation import TranslationWithBacktranslationTask

logger = logging.getLogger(__name__)

Src = Tensor
Tgt = Tensor
Constraints = List[Tensor]
CreateConstraintFunction = Callable[[Src, Tgt], Constraints]


@dataclass
class GlossaryTaskConfig:
    """ Configuration for the glossary task."""

    enabled: bool = False
    ok_to_increase_dict_size: bool = True
    constraint_positional_shift: int = 0

    @staticmethod
    def from_args(args):
        # It is not ok to increase the dictionary size if we are loading a pretrained model.
        # TODO: verify that this is the case.
        ok_to_increase_dict_size = args.restore_file is None
        return GlossaryTaskConfig(
            enabled=args.glossary_enabled,
            ok_to_increase_dict_size=ok_to_increase_dict_size,
            constraint_positional_shift=args.glossary_constraint_positional_shift,
        )

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """Add arguments to the parser for this task."""
        parser.add_argument(
            "--glossary-enabled", default=False, action="store_true", help="Should the glossary task be enabled?"
        )
        parser.add_argument(
            "--glossary-training-constraints",
            default=None,
            type=str.lower,
            choices=["target-sampling"],
            help="How should the glossary training constraints be made?",
        )
        parser.add_argument(
            "--glossary-constraint-positional-shift",
            default=0,
            type=int,
            help="By how much should we shift the positional encoding of the glossary constraints?",
        )


def make_positions_with_constraints(
    tensor: Tensor, padding_idx: int, onnx_trace: bool = False, shift_amount=0, shift_from_symbol=None
):
    """Replace non-padding symbols with their position numbers.

    Position numbers begin at padding_idx+1. Padding symbols are ignored.

    This code is based on fairseq's make_positions with the addition of shifting all symbols from and with the "shift_from_symbol" by "shift_amount".
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
            positions[int(shift_symbol_idx[0].item()), int(shift_symbol_idx[1].item()):] += shift_amount
    positions = positions * mask + padding_idx
    return positions.long()


def apply_monkey_patch_for_make_positions(shift_from_symbol: int, shift_amount: int):
    # Monkey-patch the make_positions function to be aware of the glossary constraints.
    from fairseq import utils

    utils.make_positions = partial(
        make_positions_with_constraints, shift_from_symbol=shift_from_symbol, shift_amount=shift_amount
    )


def remove_madeupwords_from_dictionary(dictionary: Dictionary) -> int:
    """
    Remove madeupwords from the dictionary. Changes are done in-place. Returns an int of the number of madeupwords removed.
    """
    # We remove the madeupwords from the dictionary.
    # In symbols, a list of string symbols is stored.
    # In indices, a dict of sym -> int is stored.
    # In count, a list of ints is stored, corresponding to the symbols.
    idx = 0
    while (madeupword := f"madeupword{idx:04d}") in dictionary:
        sym_idx = dictionary.indices[madeupword]
        del dictionary.symbols[sym_idx - idx]
        del dictionary.indices[madeupword]
        del dictionary.count[sym_idx - idx]
        idx += 1
    return idx


def ensure_symbols_are_present(
    src_dict: Dictionary, tgt_dict: Dictionary, symbols: List[str], ok_to_increase_dict_size: bool
) -> None:
    """
    Ensure that the symbols in the source and target dictionary are present.

    Makes changes to the dictionaries in-place.
    """
    original_src_size, original_tgt_size = len(src_dict), len(tgt_dict)
    _ = remove_madeupwords_from_dictionary(src_dict)
    _ = remove_madeupwords_from_dictionary(tgt_dict)
    for symbol in symbols:
        src_dict.add_symbol(symbol)
        tgt_dict.add_symbol(symbol)
    src_dict.pad_to_multiple_(8)
    tgt_dict.pad_to_multiple_(8)
    if not ok_to_increase_dict_size:
        # Let's not crash - but rather point out that we are not allowed to increase the dictionary size.
        if len(src_dict) != original_src_size:
            logger.warning("The dictionary size changed. The model loading will probably fail.")
        if len(tgt_dict) != original_tgt_size:
            logger.warning("The dictionary size changed. The model loading will probably fail.")



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
        glossary_task_config: GlossaryTaskConfig,
    ):
        super().__init__(args, src_dict, tgt_dict)
        self.glossary_task_config = glossary_task_config
        # Ensure that <sep> and <c> are defined in the dictionaries.
        ensure_symbols_are_present(
            self.source_dictionary,
            self.target_dictionary,
            ["<c>", "<sep>"],
            self.glossary_task_config.ok_to_increase_dict_size,
        )
        assert (
            self.target_dictionary == self.source_dictionary
        ), "The target dictionary must be the same as the source dictionary, \
    because we use is_word_initial based on a single dictionary and use it for both src and tgt."
        self.is_word_initial = get_word_beginnings(args, self.source_dictionary)

        apply_monkey_patch_for_make_positions(
            shift_from_symbol=self.source_dictionary["<sep>"],
            shift_amount=self.glossary_task_config.constraint_positional_shift,
        )

    @staticmethod
    def add_args(parser):
        super(TranslationWithGlossaryTask, TranslationWithGlossaryTask).add_args(parser)
        """Add task-specific arguments to the parser."""
        GlossaryTaskConfig.add_args(parser)

    @classmethod
    def setup_task(cls, args: argparse.Namespace, **kwargs):
        config = GlossaryTaskConfig.from_args(args)
        if config.enabled:
            logger.info("Glossary is ENABLED")
        else:
            logger.info("Glossary is DISABLED")
        kwargs["glossary_task_config"] = config

        # We use the fact that super().setup_task() calls cls(args, kwargs).
        return super().setup_task(args, **kwargs)


def apply_constraints(constraints: Constraints, src: Src, constraint_idx, sep_idx) -> Src:
    """
    Apply constraints to a source sequence.
    """
    return src


def target_sampling_constraints(_: Src, tgt: Tgt) -> Constraints:
    """
    Create target sampling constraints.
    """
