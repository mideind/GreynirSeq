# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import math
from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.models import FairseqModel

from greynirseq.nicenlp.utils.label_schema.label_schema import (
    make_dict_idx_to_vec_idx,
    make_group_masks,
    make_group_name_to_group_attr_vec_idxs,
)

Numeric = Union[float, int]  # pylint: disable=unsubscriptable-object


@register_criterion("multilabel_token_classification")
class MultiLabelTokenClassificationCriterion(FairseqCriterion):
    def __init__(self, task, label_dictionary=None, label_schema=None):
        super().__init__(task)
        self.task = task
        self.label_dictionary = label_dictionary if label_dictionary is not None else self.task.label_dictionary
        self.label_schema = label_schema if label_schema is not None else self.task.label_schema
        self.group_masks = make_group_masks(self.label_dictionary, self.label_schema)
        self.group_name_to_group_attr_vec_idxs = make_group_name_to_group_attr_vec_idxs(
            self.label_dictionary, self.label_schema
        )
        self.cat_dict_to_vec_idx = make_dict_idx_to_vec_idx(self.label_dictionary, self.label_schema.label_categories)

    def forward(self, model: FairseqModel, sample: Dict[str, Any], reduce: bool = True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(model, "task_head"), "model must provide task specific classification head"
        (cat_logits, attr_logits), _extra = model(**sample["net_input"], features_only=True)
        loss, nwords_total, logging_output = self.compute_loss(
            sample,
            cat_logits,
            attr_logits,
            target_cats=sample["target_cats"],
            target_exclude_mask=sample["target_exclude_mask"],
            target_attrs=sample["target_attrs"],
            reduce=reduce,
        )
        return loss, nwords_total, logging_output

    def compute_loss_from_logits(
        self,
        sample: Dict[str, Any],
        cat_logits: Any,
        attr_logits: Any,
        target_attrs: Any,
        target_cats: Any,
        target_exclude_mask: Any,
        reduce: bool = True,
    ):
        bsz, _max_nwords = target_cats.shape
        bsz, _max_nwords, _num_attrs = target_attrs.shape

        pad_idx = self.label_dictionary.pad()
        padding_mask = target_cats.ne(pad_idx)

        device = cat_logits.device
        for k, v in self.group_name_to_group_attr_vec_idxs.items():
            if device == v.device:
                break
            self.group_name_to_group_attr_vec_idxs[k] = v.to(device)
        if self.group_masks.device != device:
            self.group_masks = self.group_masks.to(device)
        if self.cat_dict_to_vec_idx.device != device:
            self.cat_dict_to_vec_idx = self.cat_dict_to_vec_idx.to(device)

        # Batch x Time x Depth -> Batch x Depth x Time
        cat_loss = F.cross_entropy(cat_logits.transpose(2, 1), self.cat_dict_to_vec_idx[target_cats], reduction="none")

        # padding_value is -100 which as special semantics in cross_entropy
        cat_loss = (cat_loss * target_exclude_mask).sum()

        # each attribute group is one-hot, so Attr is multi-hot
        # attr_logits:  Batch x Time x Attr
        group_losses = []
        correct_attrs = torch.ones_like(target_cats).bool()

        missing_binary_targets = torch.zeros_like(target_attrs)
        cat_vec_idxs = self.cat_dict_to_vec_idx[target_cats.clone()]
        target_pad_val = -100
        cat_vec_idxs = cat_vec_idxs * cat_vec_idxs.ne(target_pad_val)  # make padding select 0th index

        # we want fixed iteration order of group names
        for i, group_name in enumerate(self.label_schema.group_names):
            group_idxs = self.group_name_to_group_attr_vec_idxs[group_name]
            group_loss_mask = self.group_masks[cat_vec_idxs][:, :, i]
            group_loss_mask *= (
                padding_mask * target_exclude_mask
            )  # reset padding to zero, since 0th vector was selected where padding is
            group_logits = attr_logits[:, :, group_idxs]

            if group_idxs.numel() == 1:
                missing_binary_targets[:, :, group_idxs[0]] = 1
                missing_binary_targets[:, :, group_idxs[0]] *= group_loss_mask
                group_targets = target_attrs[:, :, group_idxs[0]]

                group_logits = attr_logits[:, :, group_idxs[0]]
                group_loss = F.binary_cross_entropy_with_logits(
                    group_logits, group_targets.type_as(group_logits), reduction="none"
                )
                group_loss = group_loss * group_loss_mask

                group_losses.append(group_loss)

                correct = (
                    (group_logits.ge(0).int() == group_targets)
                    * group_loss_mask.bool()
                    * target_exclude_mask.bool()
                    * padding_mask.bool()
                )

            else:
                # Batch x Time x Depth -> Batch x Depth x Time
                group_targets = target_attrs[:, :, group_idxs].max(dim=-1).indices
                bsz, num_words, num_members = group_logits.shape
                group_loss = F.cross_entropy(group_logits.transpose(2, 1), group_targets, reduction="none")
                group_loss *= group_loss_mask.type_as(group_logits)
                group_loss *= target_exclude_mask * padding_mask
                group_losses.append(group_loss)

                correct = (
                    (group_logits.max(dim=-1).indices == group_targets)
                    * group_loss_mask.bool()
                    * target_exclude_mask.bool()
                    * padding_mask
                )

            # Correct attrs starts true for all words
            # then a single incorrect should flip the word
            # Just need to make sure to only flip those that have current group attrs!

            keep_unchanged = group_loss_mask.bool().bitwise_not()
            correct_attrs *= correct + keep_unchanged

        correct_attrs *= padding_mask * target_exclude_mask.bool()  # Only count words, not padding
        correct_cat = (
            (cat_logits.max(-1).indices == self.cat_dict_to_vec_idx[target_cats]) * padding_mask.bool()
        ) * target_exclude_mask.bool()
        nwords_total = (target_exclude_mask == 1).sum().item()
        correct_all = correct_attrs * correct_cat

        # NOTE: Just target attrs does not suffice for the binary labels
        # since 0 has a meaning, hence adding missing_binary_targets
        attrs_divisor = target_attrs.sum(-1) + missing_binary_targets.sum(-1)
        attrs_divisor[attrs_divisor == 0] = 1

        # average attributes per word, sum across sequence&batch
        attr_loss = (torch.stack(group_losses, dim=-1).sum(-1) / attrs_divisor).sum()
        loss = cat_loss + attr_loss

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
    def aggregate_logging_outputs(logging_outputs: List[Dict[str, Numeric]]):
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
            "loss": round(float((cat_loss + attr_loss) / float(sample_size) / math.log(2)), 3),
            "ppl": round(float(cat_loss) / float(sample_size) / math.log(2), 3),
            "attr_loss": round(float(attr_loss) / float(sample_size) / math.log(2), 3),
            "cat_acc": round(float(ncorrect_cat) / float(sample_size), 3),
            "acc_exact": round(float(ncorrect_exact) / float(sample_size), 3),
            "attrs_acc": round(float(ncorrect_attrs) / float(sample_size), 3),
            "ntokens": ntokens,
            "nwords": nwords,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        return agg_output
