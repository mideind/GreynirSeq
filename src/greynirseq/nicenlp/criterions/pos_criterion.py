import itertools
import math
import time
from collections import namedtuple

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import utils

import torch
import torch.nn.functional as F

try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from greynirseq.utils.ner import EvalNER
import greynirseq.nicenlp.utils.greynir.greynir_utils as greynir_utils

import pyximport
pyximport.install()
import greynirseq.nicenlp.utils.greynir.tree_dist as tree_dist


def targets_to_flat_mask_no_bos(targets, pad_idx):
    bsz, _ = targets.shape
    padding_mask = torch.cat(
        [targets.new_empty(bsz, 1).fill_(pad_idx + 1), targets], 1
    ).eq(pad_idx)

    no_bos_mask = torch.cat([targets.new_empty(bsz, 1).fill_(pad_idx), targets], 1).ne(
        pad_idx
    )

    return no_bos_mask[padding_mask.bitwise_not()]


@register_criterion("pos_ice")
class POSCriterion(FairseqCriterion):
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, "classification_heads")
            # and "multi_label_word_classification" in model.classification_heads
            and "pos_ice" in model.classification_heads
        ), "model must provide sentence classification head for --criterion=pos_ice"

        torch.autograd.set_detect_anomaly(True)
        torch.set_printoptions(precision=4, linewidth=160)

        target_cats = sample["target_cats"]
        target_attrs = sample["target_attrs"]
        nwords = sample["nwords"]

        bsz, _max_nwords = target_cats.shape
        bsz, _max_nwords, num_attrs = target_attrs.shape

        (cat_logits, attr_logits, _words_w_bos), _extra = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name="pos_ice",
        )

        """
            separate strength and degree (esb evb)

            (ao does not allow empty in annotald)
        """
        label_shift  = self.task.label_dictionary.nspecial
        label_schema = self.task.label_schema
        pad_idx = self.task.label_dictionary.pad()
        group_names = label_schema.group_name_to_labels.keys()
        group_name_to_mapped_vec_idxs = {
            gname: torch.tensor(
                [
                    self.task.label_dictionary.index(gitem) - label_shift
                    for gitem in label_schema.group_name_to_labels[gname]
                ]
            )
            for gname in group_names
        }

        cat_target_to_vec_idx = target_cats.new_empty(len(self.task.label_dictionary)).fill_(
            len(self.task.label_dictionary) - 1
        )
        for vec_idx, lbl in enumerate(label_schema.label_categories):
            cat_target_to_vec_idx[self.task.label_dictionary.index(lbl)] = vec_idx

        target_padding_mask = target_cats.ne(pad_idx)
        target_cats_flat = target_cats[target_padding_mask]

        flat_mask_no_bos = targets_to_flat_mask_no_bos(
            target_cats, pad_idx
        ).unsqueeze(-1)

        pred_flat = cat_logits.masked_select(flat_mask_no_bos).reshape(
            -1, self.task.num_cats
        )
        mapped_target_cats_flat = cat_target_to_vec_idx[target_cats_flat]
        # TODO: use weights?
        cat_loss = F.cross_entropy(
            pred_flat, mapped_target_cats_flat, reduction="sum"
        )

        target_attrs_flat = target_attrs.masked_select(
            target_padding_mask.unsqueeze(-1)
        ).reshape(-1, num_attrs)
        attr_logits = attr_logits.masked_select(flat_mask_no_bos).reshape(-1, num_attrs)

        # attr_logits:  (bsz * words) x attrs
        group_losses = []
        correct_attrs = torch.ones_like(target_cats_flat).bool()
        for group_name in label_schema.group_names:
            # we want fixed iteration order of group names
            mapped_group_idxs = group_name_to_mapped_vec_idxs[group_name]
            group_targets = target_attrs_flat[:, mapped_group_idxs].max(dim=-1)[1]
            group_target_mask = target_attrs_flat[:, mapped_group_idxs].bool().any(dim=-1)
            group_logits = attr_logits[:, mapped_group_idxs]
            if len(mapped_group_idxs) == 1:
                group_loss = F.binary_cross_entropy_with_logits(
                    group_logits.squeeze(-1),
                    group_targets.type_as(attr_logits),
                    reduction="none"
                )
            else:
                group_loss = F.cross_entropy(group_logits, group_targets, reduction="none")

            group_losses.append(group_loss * group_target_mask.type_as(group_loss))
            correct = (group_logits.max(dim=-1)[1] == group_targets) * group_target_mask
            ignore_mask = group_target_mask.bool().bitwise_not()
            correct_attrs *= (ignore_mask + correct.bool())

        # group_losses after:  (bsz * words) x group
        group_losses = torch.stack(group_losses, dim=1)
        attr_loss = group_losses.sum()

        loss = cat_loss + attr_loss

        correct_cat = pred_flat.max(dim=-1)[1] == mapped_target_cats_flat
        correct_exact = correct_attrs * correct_cat

        nwords_total = nwords.sum().data
        logging_output = {
            "cat_loss": cat_loss.data,
            "attr_loss": attr_loss.data,
            "ntokens": sample["ntokens"],
            "nwords": nwords_total,
            "nsentences": bsz,
            "sample_size": nwords_total,
            "ncorrect_cat": correct_cat.sum(),
            "ncorrect_exact": correct_exact.sum(),
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
            "accuracy_cat": float(ncorrect_cat) / float(sample_size),
            "accuracy_exact": float(ncorrect_exact) / float(sample_size),
            "accuracy_attrs": float(ncorrect_attrs) / float(sample_size),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        return agg_output
