from dataclasses import dataclass, field
from typing import Dict, Any

import torch
import torch.nn.functional as F

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
    def __init__(self, cfg: IncrementalParserCriterionConfig, task: FairseqTask):
        super().__init__(task)
        self.task = task
        self.cfg = cfg

    def forward(self, model: FairseqModel, sample: Dict[str, Any], **kwargs):
        # span_logits = model(**sample["net_input"], features_only=True)
        dec, encoder_out = kwargs["dec"], kwargs["encoder_out"]
        constit_output = dec(encoder_out=encoder_out, sample=sample)
        loss, logging_outputs = self.compute_loss(sample, constit_output, attention_loss_weight=self.cfg.attention_loss_weight)
        sample_size = sample["nwords"]
        return loss, sample_size, logging_outputs

    @classmethod
    def compute_loss(cls, sample, constit_output, attention_loss_weight=1.0):
        chain_mask_without_root = sample["chain_mask"]  # nwords x bsz x num_nodes
        nwords_per_step = sample["nwords_per_step"]  # nwords x bsz

        # prepend root where the sequence is still alive, for each, for each step
        chain_mask = torch.cat([nwords_per_step.gt(0).unsqueeze(-1), chain_mask_without_root], dim=2)

        tgt_padding_mask = sample["target_padding_mask"]
        tgt_parents = sample["target_parents"]
        tgt_preterms = sample["target_preterminals"]
        tgt_depths = sample["target_depths"]
        parent_logits = constit_output.parent_logits
        preterm_logits = constit_output.preterm_logits
        attention = constit_output.attention

        # ic(parent_logits.shape, tgt_parents.shape, tgt_padding_mask.shape)
        # ic(preterm_logits.shape, tgt_preterms.shape, tgt_padding_mask.shape)
        # ic(tgt_padding_mask.long())
        # ic(tgt_depths.shape, tgt_depths)
        # ic([att.shape for att in attention])

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
        # ic(parent_logits.shape, tgt_parents.shape, tgt_parents, tgt_padding_mask.shape, tgt_padding_mask.long())

        attachment_losses = []
        num_attachments = 0
        ncorrect_attachments = 0
        for step, attn in enumerate(attention):
            if step == 0:
                # there is only one attendable vector per seq, no need to do anything
                continue

            attn_ = attn.clone()
            # while not zero, padding now has negligible effect on loss
            attn_[attention_padding[step]] = NEGLIGIBLE_LOGIT_VALUE
            attachment_losses.append(F.cross_entropy(attn_, tgt_depths[step]))

            # dont count attachments on finished sequences
            step_correct_attachments = attn_.argmax(-1).eq(tgt_depths[step]) * attention_padding[
                step
            ].logical_not().sum(-1).gt(0)
            ncorrect_attachments += step_correct_attachments.sum()
            num_attachments += attention_padding[step].logical_not().sum(-1).gt(0).sum()

        target_mask = tgt_padding_mask.logical_not()
        num_label_targets = target_mask.sum(-1)
        ncorrect_parents = (parent_logits.argmax(-1).eq(tgt_parents) * target_mask).sum().float()
        ncorrect_preterms = (preterm_logits.argmax(-1).eq(tgt_preterms) * target_mask).sum().float()

        loss = parent_loss + preterm_loss
        if attention_loss_weight > 0:
            loss += attention_loss_weight * sum(attachment_losses)

        logging_output = {
            "loss": loss,
            "ntargets": num_label_targets,
            "ncorrect_parents": ncorrect_parents,
            "ncorrect_preterms": ncorrect_preterms,
            "parent_acc": ncorrect_parents / num_label_targets,
            "preterm_acc": ncorrect_preterms / num_label_targets,
            "nattach": num_attachments,
            "ncorrect_attach": ncorrect_attachments,
            "attach_acc": ncorrect_attachments / num_attachments,
            "nwords": sample["nwords"],
            "nsentences": sample["nsentences"],
        }

        return loss, logging_output
