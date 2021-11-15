#
# Based on fairseq/tasks/translation.py that has the following license
#
#    Copyright (c) Facebook, Inc. and its affiliates.
#
#    This source code is licensed under the MIT license found in the
#    LICENSE file in the root directory of this source tree.

import logging

from fairseq.data.encoders.utils import get_whole_word_mask
from fairseq.tasks import register_task

from greynirseq.nicenlp.tasks.translation_with_backtranslation import TranslationWithBacktranslationTask

EVAL_BLEU_ORDER = 4


logger = logging.getLogger(__name__)


@register_task("translation_with_glossary")
class TranslationWithGlossaryTask(TranslationWithBacktranslationTask):
    """
    Translate from one (source) language to another (target) language.

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

    @staticmethod
    def add_args(parser):
        super().add_args(parser)
        """Add task-specific arguments to the parser."""
        parser.add_argument(
            "--glossary-enabled", default=False, action="store_true", help="Should the glossary task be enabled?"
        )

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(self, args, src_dict, tgt_dict)
        # Ensure that <sep> is defined in the dictionary
        # Ensure that <c> is defined in the dictionary
        # self.bt_idx = self.src_dict.add_symbol("<bt>")
        # self.tgt_dict.add_symbol("<bt>")
        # self.src_dict.pad_to_multiple_(8)
        # self.tgt_dict.pad_to_multiple_(8)
