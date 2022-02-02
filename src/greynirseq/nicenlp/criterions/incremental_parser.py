from dataclasses import dataclass, field
from typing import Dict, Any

import torch.nn.functional as F

from fairseq.models import FairseqModel
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq.data.data_utils import lengths_to_padding_mask
from torch.nn.utils.rnn import pad_sequence

from icecream import ic


IGNORE_INDEX = -100
NEGLIGIBLE_LOGIT_VALUE = -100


@dataclass
class IncrementalParserCriterionConfig(FairseqDataclass):
    attention_loss_weight: float = field(default=0.0, metadata={"help": "Include attention in loss calculation."})


@register_criterion("incremental_parser", dataclass=IncrementalParserCriterionConfig)
class IncrementalParserCriterion(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)
        self.task = task

    # @staticmethod
    # def add_args(parser):
    #     """Add criterion-specific arguments to the parser."""
    #     # fmt: off
    #     parser.add_argument('--log-dists', action="store_true", required=False,
    #                         help='Print out tree distances as part of metrics of parsers')
    #     # fmt: on

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
        span_logits = model(**sample["net_input"], features_only=True)
        loss, nwords_total, logging_output = self.compute_loss(sample, span_logits, reduce=reduce)
        return loss, nwords_total, logging_output

    @classmethod
    def compute_whole(
        cls, tgt_padding_mask, tgt_parents, tgt_preterms, tgt_depths, parent_logits, preterm_logits, attention, chain_mask
    ):
        ic(parent_logits.shape, tgt_parents.shape, tgt_padding_mask.shape)
        ic(preterm_logits.shape, tgt_preterms.shape, tgt_padding_mask.shape)
        ic(tgt_padding_mask.long())

        ic(tgt_depths.shape, tgt_depths)
        ic([att.shape for att in attention])

        tgt_parents_w_ignore = tgt_parents.clone()
        tgt_parents_w_ignore[tgt_padding_mask] = IGNORE_INDEX

        tgt_preterms_w_ignore = tgt_preterms.clone()
        tgt_preterms_w_ignore[tgt_padding_mask] = IGNORE_INDEX

        # Batch x Time x Channel -> Batch x Channel x Time
        parent_loss = F.cross_entropy(
            parent_logits.transpose(2, 1), tgt_parents_w_ignore, reduction="sum", ignore_index=IGNORE_INDEX
        )
        # Batch x Time x Channel -> Batch x Channel x Time
        preterm_loss = F.cross_entropy(
            preterm_logits.transpose(2, 1), tgt_preterms_w_ignore, reduction="sum", ignore_index=IGNORE_INDEX
        )

        right_chain_lengths = chain_mask.sum(-1)
        attention_padding = [lengths_to_padding_mask(s) for s in right_chain_lengths]

        attachment_losses = []
        for step, attn in enumerate(attention):
            if step == 0:
                # there is only one target, no need to do anything
                continue

            attn_ = attn.clone()
            # while not zero, padding now has negligible effect on loss
            attn_[attention_padding[step]] = NEGLIGIBLE_LOGIT_VALUE
            attachment_losses.append(F.cross_entropy(attn_, tgt_depths[step]))

        loss = parent_loss + preterm_loss + sum(attachment_losses)
        return loss

    @classmethod
    def compute_step(
        cls,
        step_preterm_mask,
        step_parent_mask,
        step_word_padding_mask,
        step_parents,
        step_preterms,
        tgt_depths,
        step_nwords,
        step_logits1,
        step_logits2,
    ):
        pass
