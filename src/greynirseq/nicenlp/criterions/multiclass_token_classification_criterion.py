# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import math
from typing import Any, Dict, List, Union

import torch.nn.functional as F
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.models import FairseqModel

Numeric = Union[float, int]  # pylint: disable=unsubscriptable-object


@register_criterion("multi_class_token_classification")
class MultiClassTokenClassificationCriterion(FairseqCriterion):
    def forward(self, model: FairseqModel, sample: Dict[str, Any], reduce: bool = True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(model, "task_head"), "model must provide task specific classification head"

        target_attrs = sample["target_attrs"]
        nwords = sample["nwords"]

        bsz, _max_nwords = target_attrs.shape

        logits, _extra = model(**sample["net_input"], features_only=True)

        pad_idx = model.task.label_dictionary.pad()
        padding_mask = target_attrs.ne(pad_idx)

        # Batch x Time x Depth -> Batch x Depth x Time
        loss = F.cross_entropy(
            logits.transpose(2, 1),
            target_attrs,
            reduction="none",
        )

        # padding_value is -100 which as special semantics in cross_entropy
        loss = (loss * padding_mask).sum()

        nwords_total = nwords.sum().item()

        correct = (logits.max(-1).indices == target_attrs) * padding_mask.bool()
        logging_output = {
            "loss": loss.item(),
            "ntokens": sample["ntokens"],
            "nwords": nwords_total,
            "nsentences": bsz,
            "sample_size": nwords_total,
            "correct": correct.sum(),
        }

        return loss, nwords_total, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs: List[Dict[str, Numeric]]):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        nwords = int(sum(log.get("nwords", 0) for log in logging_outputs))
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        correct = sum(log.get("correct", 0) for log in logging_outputs)
        loss = float(sum(log.get("loss", 0.0) for log in logging_outputs))

        agg_output = {
            "loss": float((loss) / float(sample_size) / math.log(2)),
            "ppl": float(loss) / float(sample_size) / math.log(2),
            "acc_exact": float(correct) / float(sample_size),
            "ntokens": ntokens,
            "nwords": nwords,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        return agg_output
