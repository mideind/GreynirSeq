import itertools
import math
import time
from collections import namedtuple

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import utils

import torch
import torch.nn.functional as F

from greynirseq.nicenlp.utils.label_schema.label_schema import make_dict_idx_to_vec_idx


@register_criterion("pos_ice")
class POSCriterion(FairseqCriterion):
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(
            model, "pos_head"
        ), "model must provide sentence classification head for pos_ice"

        torch.autograd.set_detect_anomaly(True)
        torch.set_printoptions(precision=4, linewidth=160)

        target_cats = sample["target_cats"]
        target_attrs = sample["target_attrs"]
        nwords = sample["nwords"]

        bsz, _max_nwords = target_cats.shape
        bsz, _max_nwords, num_attrs = target_attrs.shape

        (cat_logits, attr_logits), _extra = model(
            **sample["net_input"], features_only=True
        )

        label_shift = self.task.label_dictionary.nspecial
        schema = self.task.label_schema
        label_dict = model.task.label_dictionary
        pad_idx = label_dict.pad()

        group_names = schema.group_name_to_labels.keys()
        group_name_to_mapped_vec_idxs = {
            gname: torch.tensor(
                [
                    label_dict.index(gitem) - label_shift
                    for gitem in schema.group_name_to_labels[gname]
                ]
            )
            for gname in group_names
        }

        target_mask = target_cats.ne(pad_idx)

        cat_dict_to_vec_idx = make_dict_idx_to_vec_idx(
            label_dict, schema.label_categories, device=target_cats.device
        )

        # Batch x Time x Depth -> Batch x Depth x Time
        cat_loss = F.cross_entropy(
            cat_logits.transpose(2, 1),
            cat_dict_to_vec_idx[target_cats],
            reduction="sum",
        )

        # attr_logits:  Batch x Time x Attr
        group_losses = []
        name_to_group_attr_vec_idxs = group_name_to_mapped_vec_idxs
        correct_attrs = torch.ones_like(target_cats).bool()
        # we want fixed iteration order of group names
        for group_name in schema.group_names:
            group_idxs = name_to_group_attr_vec_idxs[group_name]
            group_loss_mask, group_targets = target_attrs[:, :, group_idxs].max(dim=-1)
            group_logits = attr_logits[:, :, group_idxs]

            if len(group_idxs) == 1:
                group_loss = F.binary_cross_entropy_with_logits(
                    group_logits, group_targets.type_as(attr_logits), reduction="mean"
                )
            else:
                # Batch x Time x Depth -> Batch x Depth x Time
                group_loss = F.cross_entropy(group_logits.transpose(2, 1), group_targets, reduction="none") * group_loss_mask.type_as(group_logits)

            group_losses.append(group_loss)
            correct = (
                group_logits.max(dim=-1).indices == group_targets
            ) * group_loss_mask.bool()
            ignore_mask = group_loss_mask.bool().bitwise_not()
            correct_attrs *= ignore_mask + correct

        correct_attrs *= target_mask
        attrs_divisor = target_attrs.sum(-1)
        attrs_divisor[attrs_divisor == 0] = 1
        # average attributes per word, sum across sequence&batch
        attr_loss = (torch.stack(group_losses, dim=2).sum(-1) / attrs_divisor).sum()

        loss = cat_loss + attr_loss

        correct_cat = (
            cat_logits.max(-1).indices == cat_dict_to_vec_idx[target_cats]
        ) * target_mask.bool()
        correct_all = correct_attrs * correct_cat

        nwords_total = nwords.sum().item()
        logging_output = {
            "cat_loss": cat_loss.item(),
            "attr_loss": attr_loss.item(),
            "ntokens": sample["ntokens"],
            "nwords": nwords_total,
            "nsentences": bsz,
            "sample_size": nwords_total,
            "ncorrect_cat": correct_cat.sum(),
            "ncorrect_exact": correct_all.sum(),
            "ncorrect_attrs": correct_attrs.sum(),
        }

        return loss, nwords_total, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        nwords = int(sum(log.get("nwords", 0) for log in logging_outputs))
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        ncorrect_cat = sum(log.get("ncorrect_cat", 0) for log in logging_outputs)
        ncorrect_exact = sum(log.get("ncorrect_exact", 0) for log in logging_outputs)
        ncorrect_attrs = sum(log.get("ncorrect_attrs", 0) for log in logging_outputs)

        cat_loss = float(sum(log.get("cat_loss", 0.0) for log in logging_outputs))
        attr_loss = float(sum(log.get("attr_loss", 0.0) for log in logging_outputs))

        agg_output = {
            "loss": float((cat_loss + attr_loss) / float(sample_size) / math.log(2)),
            "ppl": float(cat_loss) / float(sample_size) / math.log(2),
            "attr_loss": float(attr_loss) / float(sample_size) / math.log(2),
            "acc_cat": float(ncorrect_cat) / float(sample_size),
            "acc_exact": float(ncorrect_exact) / float(sample_size),
            "acc_attrs": float(ncorrect_attrs) / float(sample_size),
            "ntokens": ntokens,
            "nwords": nwords,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        return agg_output
