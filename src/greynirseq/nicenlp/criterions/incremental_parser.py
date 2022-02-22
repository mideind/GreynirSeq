from dataclasses import dataclass, field
import math
from typing import Dict, Any

import torch
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.models import FairseqModel
from fairseq.tasks import FairseqTask
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
    def __init__(self, cfg: IncrementalParserCriterionConfig, task: FairseqTask, padding_idx=1):
        super().__init__(task)
        self.task = task
        self.cfg = cfg
        self.padding_idx = padding_idx

    def forward(self, model: FairseqModel, sample: Dict[str, Any], reduce=True, **kwargs):
        constit_output = model(**sample["net_input"])
        loss, logging_outputs = self.compute_loss(
            sample, constit_output, attention_loss_weight=self.cfg.attention_loss_weight
        )
        sample_size = sample["nwords"].sum()
        return loss, sample_size, logging_outputs

    @classmethod
    def compute_loss(cls, sample, constit_output, attention_loss_weight=1.0, reduce=True, padding_idx=1):
        # bsz x nsteps x nodes -> nsteps x bsz x nodes
        chain_mask_without_root = sample["net_input"]["chain_mask"].transpose(0, 1)
        nwords_per_step = sample["net_input"]["nwords_per_step"].T  # bsz x nsteps -> nsteps x bsz
        padding_mask = sample["target_padding_mask"]

        # prepend root where the sequence is still alive, for each, for each step
        chain_mask = torch.cat([nwords_per_step.gt(0).unsqueeze(-1), chain_mask_without_root], dim=2)

        tgt_parents_w_ignore = sample["target_parents"].clone()
        tgt_parents_w_ignore[padding_mask] = IGNORE_INDEX
        tgt_preterms_w_ignore = sample["target_preterminals"].clone()
        tgt_preterms_w_ignore[padding_mask] = IGNORE_INDEX

        # bsz x nsteps x features -> bsz x features x nsteps
        parent_loss = F.cross_entropy(
            constit_output.parent_logits.transpose(2, 1),
            tgt_parents_w_ignore,
            reduction="sum" if reduce else "none",
            ignore_index=IGNORE_INDEX,
        )
        # Batch x Time x Channel -> Batch x Channel x Time
        preterm_loss = F.cross_entropy(
            constit_output.preterm_logits.transpose(2, 1),
            tgt_preterms_w_ignore,
            reduction="sum" if reduce else "none",
            ignore_index=IGNORE_INDEX,
        )

        # make multi-hot flag targets
        tgt_parent_flags = torch.zeros_like(constit_output.parent_flag_logits).scatter_(-1, sample["target_parent_flags"], 1)
        tgt_preterm_flags = torch.zeros_like(constit_output.preterm_flag_logits).scatter_(-1, sample["target_preterm_flags"], 1)
        # set weight of padding to zero
        tgt_preterm_flags[:, :, padding_idx] = 1
        tgt_parent_flags[:, :, padding_idx] = 1
        # XXX: this should probably use weighing since negative samples are way more common than positive ones for many flags
        preterm_flag_loss = F.binary_cross_entropy_with_logits(constit_output.preterm_flag_logits, tgt_preterm_flags, weight=None, reduction="sum" if reduce else "none", pos_weight=None)
        parent_flag_loss = F.binary_cross_entropy_with_logits(constit_output.preterm_flag_logits, tgt_parent_flags, weight=None, reduction="sum" if reduce else "none", pos_weight=None)

        right_chain_lengths = chain_mask.sum(-1)
        attention_padding = [lengths_to_padding_mask(s) for s in right_chain_lengths]

        attachment_losses = []
        num_attachments = 0
        ncorrect_attachments = 0
        for step, attn in enumerate(constit_output.attention):
            if step == 0:
                # there is only one attendable vector per seq, no need to do anything
                continue

            attn_ = attn.clone()
            # while not zero, padding now has negligible effect on loss, we need to do it this way since cross_entropy assumes
            # all sequences have the same number of "classes" (which is equal to input length)
            attn_[attention_padding[step]] = NEGLIGIBLE_LOGIT_VALUE
            attachment_losses.append(F.cross_entropy(attn_, sample["target_depths"][:, step], reduction="sum" if reduce else "none"))

            # dont count attachments on finished sequences
            step_correct_attachments = attn_.argmax(-1).eq(sample["target_depths"][:, step]) * attention_padding[
                step
            ].logical_not().sum(-1).gt(0)
            ncorrect_attachments += step_correct_attachments.sum()
            num_attachments += attention_padding[step].logical_not().sum(-1).gt(0).sum()

        target_mask = padding_mask.logical_not()
        num_label_targets = target_mask.sum(-1)
        ncorrect_parents = (constit_output.parent_logits.argmax(-1).eq(tgt_parents_w_ignore) * target_mask).sum().float()
        ncorrect_preterms = (constit_output.preterm_logits.argmax(-1).eq(tgt_preterms_w_ignore) * target_mask).sum().float()

        attachment_loss = sum(attachment_losses)
        multiclass_loss = parent_loss + preterm_loss
        loss = multiclass_loss + (preterm_flag_loss + parent_flag_loss)
        if attention_loss_weight > 0:
            loss += attention_loss_weight * attachment_loss

        logging_output = {
            "loss": loss,
            "multiclass_loss": multiclass_loss,
            "parent_loss": parent_loss,
            "preterm_loss": preterm_loss,
            "ntargets": num_label_targets,
            "ncorrect_parents": ncorrect_parents,
            "ncorrect_preterms": ncorrect_preterms,
            "nattach": num_attachments,
            "ncorrect_attach": ncorrect_attachments,
            "nwords": sample["nwords"].sum(),
            "sample_size": sample["nwords"].sum(),
            "nsentences": sample["nsentences"],
        }
        if attention_loss_weight > 0:
            logging_output["attach_loss"] = attachment_loss

        return loss, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        loss_sum = sum(log.get("loss", 0.0) for log in logging_outputs)
        multiclass_loss = sum(log.get("multiclass_loss", 0.0) for log in logging_outputs)
        parent_loss = sum(log.get("parent_loss", 0.0) for log in logging_outputs)
        preterm_loss = sum(log.get("preterm_loss", 0.0) for log in logging_outputs)
        attach_loss = sum(log.get("preterm_loss", 0.0) for log in logging_outputs)
        nwords = sum(log.get("nwords", 0.0) for log in logging_outputs)

        ncorrect_parents = sum(log.get("ncorrect_parents", 0.0) for log in logging_outputs)
        ncorrect_preterms = sum(log.get("ncorrect_preterms", 0.0) for log in logging_outputs)
        ncorrect_attach = sum(log.get("ncorrect_preterms", 0.0) for log in logging_outputs)
        ntargets = sum(log.get("ncorrect_preterms", 0.0) for log in logging_outputs)
        nattach = sum(log.get("ncorrect_preterms", 0.0) for log in logging_outputs)

        # we convert from base e to base 2
        metrics.log_scalar("loss", loss_sum / nwords / math.log(2), nwords, round=3)
        metrics.log_scalar("parent_loss", parent_loss / ntargets / math.log(2), nwords, round=3)
        metrics.log_scalar("preterm_loss", preterm_loss / ntargets / math.log(2), nwords, round=3)
        if nattach > 0:
            metrics.log_scalar("attach_loss", attach_loss / nattach / math.log(2), nwords, round=3)
        # loss stored in meters is already in base 2
        metrics.log_scalar("_multiclass_loss", multiclass_loss / nwords / math.log(2), nwords, round=3)
        metrics.log_derived("ppl", lambda meters: utils.get_perplexity(meters["_multiclass_loss"].avg))

        metrics.log_scalar("_ncorrect_parents", ncorrect_parents)
        metrics.log_scalar("_ncorrect_preterms", ncorrect_preterms)
        metrics.log_scalar("_ncorrect_attach", ncorrect_attach)
        metrics.log_scalar("_ntargets", ntargets)
        metrics.log_scalar("_nattach", nattach)

        metrics.log_derived(
            "parent_acc",
            lambda meters: meters["_ncorrect_parents"].sum * 100 / meters["_ntargets"].sum
            if meters["_ntargets"].sum > 0
            else float("nan"),
        )
        metrics.log_derived(
            "preterm_acc",
            lambda meters: meters["_ncorrect_preterms"].sum * 100 / meters["_ntargets"].sum
            if meters["_ntargets"].sum > 0
            else float("nan"),
        )
        metrics.log_derived(
            "attach_acc",
            lambda meters: meters["_ncorrect_attach"].sum * 100 / meters["_nattach"].sum
            if meters["_nattach"].sum > 0
            else float("nan"),
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
