# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

# flake8: noqa

import itertools
import math
import time
from collections import namedtuple
from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data import Dictionary
from fairseq.models import FairseqModel
from torch import LongTensor, Tensor

import greynirseq.nicenlp.utils.constituency.chart_parser as chart_parser  # pylint: disable=no-name-in-module
import greynirseq.nicenlp.utils.constituency.greynir_utils as greynir_utils
import greynirseq.nicenlp.utils.constituency.tree_dist as tree_dist  # pylint: disable=no-name-in-module
from greynirseq.nicenlp.criterions.multilabel_token_classification_criterion import (
    MultiLabelTokenClassificationCriterion,
)
from greynirseq.nicenlp.criterions.parser_criterion import (
    Numeric,
    ParserCriterion,
    ParseResult,
    ParseStats,
    compute_parse_stats,
    f1_score,
    make_gold_label_mask,
    safe_div,
)
from greynirseq.nicenlp.models.multiparser import MultiParserOutput
from greynirseq.nicenlp.utils.label_schema.label_schema import make_dict_idx_to_vec_idx

# import pyximport; pyximport.install()


_GREYNIR_PARSER_PREFIX = "gpa."
_GREYNIR_POS_PREFIX = "gpo."


@register_criterion("multiparser")
class MultiParserCriterion(FairseqCriterion):

    common_log_keys = ("ntokens", "nsentences", "sample_size", "nwords")

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--greynir-parser-weight', type=float, metavar='FLOAT', required=False, default=1.0,
                            help='Weight of this objective function in the combined objective')
        parser.add_argument('--greynir-pos-weight', type=float, metavar='FLOAT', required=False, default=0.5,
                            help='Weight of this objective function in the combined objective')
        # fmt: on

    def __init__(self, task, greynir_parser_weight=1.0, greynir_pos_weight=1.0):
        super().__init__(task)
        self.task = task
        self.greynir_parser_weight = greynir_parser_weight
        self.greynir_pos_weight = greynir_pos_weight
        self.num_nterm_cats = task.num_nterm_cats
        self.nterm_label_shift = task.nterm_dictionary.nspecial
        self.greynir_parser_dict_idx_to_vec_idx = make_dict_idx_to_vec_idx(
            task.nterm_dictionary, task.nterm_schema.label_categories, device="cpu"
        )
        self.greynir_parser_criterion = ParserCriterion(task, log_valid_dists=False)
        self.greynir_pos_criterion = MultiLabelTokenClassificationCriterion(
            task, label_dictionary=task.term_dictionary, label_schema=task.term_schema
        )

    def forward(self, model: FairseqModel, sample: Dict[str, Any], reduce: bool = True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(model, "task_head") or hasattr(
            model, "task_heads"
        ), "model must provide task specific classification head"

        multiparser_output = model(**sample["net_input"], features_only=True)
        (
            greynir_parser_loss,
            _nwords_total,
            greynir_parser_logging_output,
        ) = self.greynir_parser_criterion.compute_loss(sample, multiparser_output.greynir_parser, reduce=reduce)
        cat_logits, attr_logits = multiparser_output.greynir_pos
        (greynir_pos_loss, _nwords_total, greynir_pos_logging_output,) = self.greynir_pos_criterion.compute_loss(
            sample,
            cat_logits,
            attr_logits,
            reduce=reduce,
            target_cats=sample["target_term_cats"],
            target_exclude_mask=sample["target_term_cats_exclude"],
            target_attrs=sample["target_term_attrs"],
        )
        merged_logging_output = {
            key: value for key, value in greynir_parser_logging_output.items() if key in self.common_log_keys
        }
        for key, value in greynir_parser_logging_output.items():
            new_key = _GREYNIR_PARSER_PREFIX + key
            merged_logging_output[new_key] = value
        for key, value in greynir_pos_logging_output.items():
            new_key = _GREYNIR_POS_PREFIX + key
            merged_logging_output[new_key] = value

        greynir_parser_loss *= self.greynir_parser_weight
        greynir_pos_loss *= self.greynir_pos_weight
        loss = greynir_parser_loss + greynir_pos_loss
        merged_logging_output["loss"] = loss
        return loss, sample["nwords"].sum(), merged_logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs: List[Dict[str, Numeric]]):
        """Aggregate logging outputs from data parallel training."""
        # grab common items
        greynir_parser_logs = []
        greynir_pos_logs = []

        def extract_from_dict(mydict, prefix, keys):
            ret = {}
            for key, value in mydict.items():
                if prefix is not None and key.startswith(prefix):
                    new_key = key.replace(prefix, "", 1)
                    ret[new_key] = value
                elif key in keys:
                    ret[key] = value
            return ret

        for log in logging_outputs:
            greynir_parser_logs.append(
                extract_from_dict(log, _GREYNIR_PARSER_PREFIX, MultiParserCriterion.common_log_keys)
            )
            greynir_pos_logs.append(extract_from_dict(log, _GREYNIR_POS_PREFIX, MultiParserCriterion.common_log_keys))

        greynir_parser_agg = ParserCriterion.aggregate_logging_outputs(greynir_parser_logs)
        agg_output = extract_from_dict(greynir_parser_agg, None, MultiParserCriterion.common_log_keys)
        agg_output["loss"] = float(sum(log.get("loss", 0.0) for log in logging_outputs))

        for key, value in greynir_parser_agg.items():
            if key in MultiParserCriterion.common_log_keys:
                continue
            agg_output[_GREYNIR_PARSER_PREFIX + key] = value
        for key, value in MultiLabelTokenClassificationCriterion.aggregate_logging_outputs(greynir_pos_logs).items():
            if key in MultiParserCriterion.common_log_keys:
                continue
            agg_output[_GREYNIR_POS_PREFIX + key] = value

        return agg_output
