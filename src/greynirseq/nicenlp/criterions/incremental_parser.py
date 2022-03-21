import math
from dataclasses import dataclass, field
from typing import Any, Dict

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass import FairseqDataclass
from fairseq.models import FairseqModel
from fairseq.tasks import FairseqTask

IGNORE_INDEX = -100
NEGLIGIBLE_LOGIT_VALUE = -100


@dataclass
class IncrementalParserCriterionConfig(FairseqDataclass):
    attention_loss_weight: float = field(default=1.0, metadata={"help": "Include attention in loss calculation."})


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
        parent_mask = sample["target_parents"].not_equal(padding_idx)
        preterm_mask = sample["target_preterms"].not_equal(padding_idx)

        # targets are ordered by step first,then batch
        #   Eg. [targets for step1, targets for step2, ..., targets for stepk]
        # NB that not all sequences in batch are of equal length
        # XXX: add class weights
        flat_tgt_preterms = sample["target_preterms"].T[preterm_mask.T]
        preterm_loss = F.cross_entropy(
            constit_output.preterm_logits, flat_tgt_preterms, reduction="sum" if reduce else "none"
        )
        flat_tgt_parents = sample["target_parents"].T[parent_mask.T]
        parent_loss = F.cross_entropy(
            constit_output.parent_logits, flat_tgt_parents, reduction="sum" if reduce else "none"
        )

        _, nclasses = constit_output.preterm_flag_logits.shape
        class_weights = constit_output.preterm_flag_logits.new_ones(nclasses)
        class_weights[padding_idx] = 0

        sparse_flat_tgt_parent_flags = sample["target_parent_flags"].transpose(0, 1)[parent_mask.T]
        dense_flat_tgt_parent_flags = torch.zeros_like(constit_output.parent_flag_logits.squeeze(1)).scatter_(
            -1, sparse_flat_tgt_parent_flags, 1
        )
        parent_flag_loss = F.binary_cross_entropy_with_logits(
            constit_output.parent_flag_logits,
            dense_flat_tgt_parent_flags,
            weight=class_weights,
            reduction="sum" if reduce else "none",
            pos_weight=None,
        )

        sparse_flat_tgt_preterm_flags = sample["target_preterm_flags"].transpose(0, 1)[preterm_mask.T]
        dense_flat_tgt_preterm_flags = torch.zeros_like(constit_output.preterm_flag_logits.squeeze(1)).scatter_(
            -1, sparse_flat_tgt_preterm_flags, 1
        )
        preterm_flag_loss = F.binary_cross_entropy_with_logits(
            constit_output.preterm_flag_logits,
            dense_flat_tgt_preterm_flags,
            weight=class_weights,
            reduction="sum" if reduce else "none",
            pos_weight=None,
        )

        attachment_losses = []
        num_attachments = 0
        ncorrect_attachments = 0
        if attention_loss_weight > 0:
            # bsz x nsteps x nodes -> nsteps x bsz x nodes
            chain_mask_without_root = sample["net_input"]["chain_mask"].transpose(0, 1)
            nwords_per_step = sample["net_input"]["nwords_per_step"].T  # bsz x nsteps -> nsteps x bsz
            # padding_mask = sample["target_padding_mask"]

            # prepend root where the sequence is still alive, for each, for each step
            chain_mask = torch.cat([nwords_per_step.gt(0).unsqueeze(-1), chain_mask_without_root], dim=2)

            for step, attn in enumerate(constit_output.attention):
                if step == 0:
                    # there is only one attendable vector per seq, no need to do anything
                    continue

                # while not zero, padding now has negligible effect on loss, we need to do it this way
                # since cross_entropy assumes all sequences have the same number of "classes"
                # (which is equal to input length)
                is_alive = nwords_per_step[step] > 0
                attn_padding = lengths_to_padding_mask(chain_mask[step, is_alive].sum(-1))
                attn_ = attn.clone()
                attn_[attn_padding] = NEGLIGIBLE_LOGIT_VALUE
                step_tgt_depths = sample["target_depths"][is_alive, step]
                # XXX: add class weights
                attachment_losses.append(F.cross_entropy(attn_, step_tgt_depths, reduction="sum" if reduce else "none"))

                ncorrect_attachments += attn_.argmax(-1).eq(step_tgt_depths).sum()
                num_attachments += attn_padding.logical_not().sum(-1).gt(0).sum()

        ncorrect_parents = constit_output.parent_logits.argmax(-1).eq(flat_tgt_parents).sum().float()
        ncorrect_preterms = constit_output.preterm_logits.argmax(-1).eq(flat_tgt_preterms).sum().float()
        num_label_targets = flat_tgt_preterms.numel()

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
            "ntokens": sample["ntokens"],
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
        attach_loss = sum(log.get("attach_loss", 0.0) for log in logging_outputs)
        nwords = sum(log.get("nwords", 0.0) for log in logging_outputs)

        ncorrect_parents = sum(log.get("ncorrect_parents", 0.0) for log in logging_outputs)
        ncorrect_preterms = sum(log.get("ncorrect_preterms", 0.0) for log in logging_outputs)
        ncorrect_attach = sum(log.get("ncorrect_attach", 0.0) for log in logging_outputs)
        ntargets = sum(log.get("ntargets", 0.0) for log in logging_outputs)
        nattach = sum(log.get("nattach", 0.0) for log in logging_outputs)

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
