from dataclasses import dataclass, field
from typing import Dict, Any

import torch.nn.functional as F

from fairseq.models import FairseqModel
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


from icecream import ic


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
        cls, tgt_padding_mask, tgt_parents, tgt_preterms, tgt_depths, parent_logits, preterm_logits, attention
    ):
        ic(parent_logits.shape, tgt_parents.shape, tgt_padding_mask.shape)
        ic(preterm_logits.shape, tgt_preterms.shape, tgt_padding_mask.shape)
        ic(tgt_padding_mask.long())

        ic(tgt_depths.shape, tgt_depths)
        ic([att.shape for att in attention])

        IGNORE_INDEX = -100
        tgt_parents_w_ignore = tgt_parents.clone()
        tgt_parents_w_ignore[tgt_padding_mask] = IGNORE_INDEX

        tgt_preterms_w_ignore = tgt_preterms.clone()
        tgt_preterms_w_ignore[tgt_padding_mask] = IGNORE_INDEX

        # Batch x Time x Channel -> Batch x Channel x Time
        parent_loss = F.cross_entropy(
            parent_logits.transpose(2, 1), tgt_parents_w_ignore, reduction="sum", ignore_index=IGNORE_INDEX
        )
        # Batch x Time x Channel -> Batch x Channel x Time
        parent_loss = F.cross_entropy(
            preterm_logits.transpose(2, 1), tgt_preterms_w_ignore, reduction="sum", ignore_index=IGNORE_INDEX
        )

        # we can also flatten the attention and solve it that way
        for attn in attention:
            # seqlen =
            # F.cross_entropy(attn, )
            pass
        breakpoint()

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

        # ic.enable()
        # ic(step_logits1.shape, step_parents.shape)
        # ic(step_logits2.shape, step_preterms.shape)
        # breakpoint()
        # loss = F.cross_entropy(
        #     step_logits1,
        #     step_parents,
        #     # step_
        # )

        pass
