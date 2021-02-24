# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import itertools
import math
import time
from collections import namedtuple
from typing import Any, Dict, List, Union

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data import Dictionary
from fairseq.models import FairseqModel
from torch import LongTensor, Tensor

import greynirseq.nicenlp.utils.constituency.chart_parser as chart_parser
import greynirseq.nicenlp.utils.constituency.greynir_utils as greynir_utils
import greynirseq.nicenlp.utils.constituency.tree_dist as tree_dist
from greynirseq.nicenlp.utils.label_schema.label_schema import (
    make_dict_idx_to_vec_idx
)
from greynirseq.types import Numeric


def gen_2d_diags(chart_width: Union[int, LongTensor]):
    """Generator for all diagonal positions in a 2d matrix, starting with right of center diagonal from pos (0,1),
       then diagonal starting at (0,2), etc."""
    for span_length in range(1, chart_width):
        for start in range(chart_width - span_length):
            ii = start
            jj = start + span_length
            yield (ii, jj)


ParseResult = namedtuple("ParseResult", ["score", "spans", "labels", "masked_lchart"])
ParseStats = namedtuple("ParseStats", ["ncorrect", "ncorrect_spans", "npred", "ngold"])


def compute_parse_stats(
    lmask: Tensor,
    mapped_target_labels: LongTensor,
    target_spans: LongTensor,
    ntargets: LongTensor,
    ignore_idxs: List[int] = [0, 1],
):
    # assumes target_spans are (nseqs, 2, nspans) and leaf_idx is shifted
    pad_val = -1
    npred, ngold = 0, 0
    ncorrect_lbls, ncorrect_spans = 0, 0

    for idx in range(ntargets.numel()):
        seq_ntargets = ntargets[idx]
        seq_targets = mapped_target_labels[idx, :seq_ntargets]
        seq_tgt_ii = target_spans[idx, 0, :seq_ntargets]
        seq_tgt_jj = target_spans[idx, 1, :seq_ntargets]
        nrows, ncols = seq_tgt_ii.max() + 1, seq_tgt_jj.max() + 1

        tgt_chart = seq_targets.new_full((nrows, ncols), fill_value=pad_val, dtype=torch.long)
        tgt_chart[seq_tgt_ii, seq_tgt_jj] = seq_targets
        lchart = lmask[idx, :nrows, :ncols, :].max(dim=-1)[1]

        tgt_mask = tgt_chart.ne(pad_val)
        pred_mask = lmask[idx, :nrows, :ncols, :].any(dim=-1).bool()
        for ignore_idx in ignore_idxs or []:
            tgt_mask *= tgt_chart.ne(ignore_idx)
            pred_mask *= lchart.ne(ignore_idx)

        # equivalent to: (is_target and is_predicted and is_same_label)
        ncorrect_lbls += ((lchart == tgt_chart) * tgt_mask * pred_mask).sum()
        npred += pred_mask.sum()
        ngold += tgt_mask.sum()
        ncorrect_spans += (tgt_mask * pred_mask).sum()

    return ParseStats(
        ncorrect=ncorrect_lbls.item(), ncorrect_spans=ncorrect_spans.item(), npred=npred.item(), ngold=ngold.item(),
    )


def safe_div(nom: Numeric, denom: Numeric):
    if not denom:
        return 0.0
    return float(nom / denom)


def f1_score(precision: Numeric, recall: Numeric):
    return safe_div(2 * precision * recall, precision + recall)


def make_gold_label_mask(sample: Dict[str, Any], dict_to_vec: LongTensor, num_labels: Union[int, LongTensor]):
    bsz = sample["nsentences"]
    ntarget_span_labels = sample["ntarget_span_labels"]
    target_span_labels = sample["target_span_labels"]
    target_spans = sample["target_spans"].reshape(bsz, -1, 2).permute(0, 2, 1)  # (bsz, 2, max_ntargets)
    mask_width = sample["nwords"].max() + 1
    mask_shape = (bsz, mask_width, mask_width, num_labels)
    gold_lmask = target_span_labels.new_zeros(mask_shape, dtype=torch.bool)
    # gold_lmask: (bsz, seq_len, seq_len, nlabels)
    for seq_idx in range(bsz):
        seq_ntargets = ntarget_span_labels[seq_idx]
        seq_labels = target_span_labels[seq_idx, :seq_ntargets]
        seq_tgt_ii, seq_tgt_jj = target_spans[seq_idx, :, :seq_ntargets].chunk(2, dim=0)
        seq_tgt_ii, seq_tgt_jj = seq_tgt_ii.squeeze(0), seq_tgt_jj.squeeze(0)
        gold_lmask[seq_idx, seq_tgt_ii, seq_tgt_jj, dict_to_vec[seq_labels]] = 1
    return gold_lmask


@register_criterion("parser")
class ParserCriterion(FairseqCriterion):
    def forward(self, model: FairseqModel, sample: Dict[str, Any], reduce: bool = True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(model, "task_head"), "model must provide task specific classification head"

        ntarget_span_labels = sample["ntarget_span_labels"]
        target_span_labels = sample["target_span_labels"]
        target_spans = sample["target_spans"]
        nwords = sample["nwords"]
        src_tokens = sample["net_input"]["src_tokens"]

        bsz, _ = src_tokens.shape

        span_logits = model(**sample["net_input"], features_only=True)

        scores = span_logits

        ncats_nterm = model.task.num_nterm_cats
        label_shift = model.task.nterm_dictionary.nspecial

        nterm_dict_idx_to_vec_idx = make_dict_idx_to_vec_idx(
            model.task.nterm_dictionary, model.task.nterm_schema.label_categories, device=ntarget_span_labels.device,
        )

        # extract targets from padded inputs
        gold_lmask = make_gold_label_mask(sample, nterm_dict_idx_to_vec_idx, ncats_nterm).type_as(scores)
        null_leaf_dict_idx = model.task.nterm_dictionary.leaf()

        _, best_lspans, _best_mask, best_lmask = chart_parser.parse_many(scores.cpu().detach(), nwords.cpu())

        # this augmentation forces max to select any label except the gold label, so that
        # parsing constructs a high scoring bad tree
        _aug_bad_tree_scores, bad_lspans, _, bad_lmask = chart_parser.parse_many(
            (scores + (1 - gold_lmask)).cpu().detach(), nwords.cpu()
        )

        parse_stats = compute_parse_stats(
            best_lmask,
            nterm_dict_idx_to_vec_idx[target_span_labels].cpu(),
            target_spans.reshape(bsz, -1, 2).transpose(2, 1).cpu(),
            ntarget_span_labels.cpu(),
        )

        # the reason we don't just reduce directly with sum to get the tree scores,
        # is so we can clamp each sentence separately
        best_lmask = best_lmask.to(scores.device).type_as(scores)
        bad_lmask = bad_lmask.to(scores.device).type_as(scores)
        best_lbl_chart = best_lmask.long().max(-1)[1]

        ignore_leaves_mask = best_lmask.new_ones(*best_lbl_chart.shape)
        ignore_leaves_mask[:, torch.arange(nwords.max()), torch.arange(1, 1 + nwords.max())] = 0
        illegal_leaves_mask = (best_lbl_chart * ignore_leaves_mask) == null_leaf_dict_idx
        illegal_leaves_loss = span_logits[:, :, :, null_leaf_dict_idx].masked_select(illegal_leaves_mask).sum()

        tree_scores_best = (best_lmask * scores).sum_to_size(bsz, 1, 1, 1).squeeze()
        tree_scores_bad = (bad_lmask * scores).sum_to_size(bsz, 1, 1, 1).squeeze()
        tree_scores_gold = (gold_lmask * scores).sum_to_size(bsz, 1, 1, 1).squeeze()

        # this is for margin loss, this is not optimized by itself
        hamming_loss = (
            ntarget_span_labels.type_as(scores) - (bad_lmask * gold_lmask).sum_to_size(bsz, 1, 1, 1).squeeze()
        )

        # clamp for hinge loss
        hinge_loss = (hamming_loss + tree_scores_bad - tree_scores_gold).clamp(0).sum()
        nterm_loss = hinge_loss + illegal_leaves_loss

        loss = nterm_loss
        nwords_total = nwords.sum().data
        logging_output = {
            "loss": loss.data,
            "hinge_loss": hinge_loss.data,
            "gold_tree": tree_scores_gold.sum().data,
            "bad_tree": tree_scores_bad.sum().data,
            "best_tree": tree_scores_best.sum().data,
            "ntokens": sample["ntokens"],
            "nwords": nwords_total,
            "nsentences": bsz,
            "sample_size": nwords_total,
            "ncorrect_labels": parse_stats.ncorrect,
            "ncorrect_spans": parse_stats.ncorrect_spans,
            # filter out NULL labels
            "gold_nlabels": target_span_labels.gt(label_shift).sum(),
            "gold_nlabels_roof": parse_stats.ngold,
            "best_nlabels_roof": parse_stats.npred,
        }

        if not model.training:

            def parse_result_to_tree(lspans):
                labels_str = [model.task.nterm_dictionary.symbols[idx + label_shift] for idx in lspans[:, 2]]
                tree = greynir_utils.Node.from_labelled_spans(lspans[:, :2].tolist(), labels_str).debinarize()
                return tree

            trees_best = list(map(parse_result_to_tree, best_lspans))

            _, tgold_lspans, _, _ = chart_parser.parse_many(gold_lmask.cpu().detach(), nwords.cpu())
            trees_gold = list(map(parse_result_to_tree, tgold_lspans))
            trees_bad = list(map(parse_result_to_tree, bad_lspans))

            logging_output["dist_gold_bad"] = sum(
                [tree_dist.tree_dist(gold, bad, None) for gold, bad in zip(trees_gold, trees_bad)]
            )
            logging_output["dist_best_gold"] = sum(
                [tree_dist.tree_dist(best, gold, None) for best, gold in zip(trees_best, trees_gold)]
            )

        return loss, nwords_total, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs: List[Dict[str, Numeric]]):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0.0) for log in logging_outputs)
        hinge_loss = float(sum(log.get("hinge_loss", 0.0) for log in logging_outputs))
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        nwords = int(sum(log.get("nwords", 0) for log in logging_outputs))
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        gold_tree = float(sum(log.get("gold_tree", 0.0) for log in logging_outputs))
        best_tree = float(sum(log.get("best_tree", 0.0) for log in logging_outputs))
        bad_tree = float(sum(log.get("bad_tree", 0.0) for log in logging_outputs))

        gold_nlabels_roof = float(sum(log.get("gold_nlabels_roof", float("inf")) for log in logging_outputs))
        best_nlabels_roof = float(sum(log.get("best_nlabels_roof", float("inf")) for log in logging_outputs))
        ncorrect_labels = float(sum(log.get("ncorrect_labels", 0) for log in logging_outputs))
        ncorrect_spans = float(sum(log.get("ncorrect_spans", 0) for log in logging_outputs))
        bracketing_errors = best_nlabels_roof + gold_nlabels_roof - 2 * ncorrect_spans

        ncorrect_cat = sum(log.get("ncorrect_cat", 0) for log in logging_outputs)
        ncorrect_exact = sum(log.get("ncorrect_exact", 0) for log in logging_outputs)
        ncorrect_attrs = sum(log.get("ncorrect_attrs", 0) for log in logging_outputs)
        cat_loss = float(sum(log.get("cat_loss", 0.0) for log in logging_outputs))
        attr_loss = float(sum(log.get("attr_loss", 0.0) for log in logging_outputs))

        label_recall = float(ncorrect_labels) / float(gold_nlabels_roof)
        label_precision = safe_div(ncorrect_labels, best_nlabels_roof)
        bracketing_precision = safe_div(ncorrect_spans, best_nlabels_roof)
        bracketing_recall = float(ncorrect_spans) / float(gold_nlabels_roof)

        agg_output = {
            "loss": float(loss_sum / math.log(2)) / float(sample_size),
            "hinge_loss": float(hinge_loss / float(sample_size)),
            "t_ppl": float(cat_loss) / float(sample_size) / math.log(2),
            "t_attr_ce": float(attr_loss) / float(sample_size) / math.log(2),
            "nt_label_precision": label_recall,
            "nt_label_recall": label_precision,
            "nt_label_f1": f1_score(label_precision, label_recall),
            "bracketing_precision": bracketing_precision,
            "bracketing_recall": bracketing_recall,
            "bracketing_f1": f1_score(bracketing_precision, bracketing_recall),
            "avg_bracketing_errors": bracketing_errors / (nsentences),
            "delta_gold_bad": (gold_tree - bad_tree) / nsentences,
            "delta_best_gold": (best_tree - gold_tree) / (nsentences),
            "t_prec_cat": float(ncorrect_cat) / float(sample_size),
            "t_prec_exact": float(ncorrect_exact) / float(sample_size),
            "t_prec_attrs": float(ncorrect_attrs) / float(sample_size),
            "gold_tree": gold_tree / nsentences,
            "bad_tree": bad_tree / nsentences,
            "best_tree": best_tree / nsentences,
            "ntokens": ntokens,
            "nwords": nwords,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        if len(logging_outputs) > 0:
            if "dist_gold_bad" in logging_outputs[0]:
                dist_diff = float(sum(log.get("dist_gold_bad", 0) for log in logging_outputs))
                agg_output.update(dist_best_gold=dist_diff / nsentences)
            if "dist_best_gold" in logging_outputs[0]:
                dist_diff = float(sum(log.get("dist_best_gold", 0) for log in logging_outputs))
                agg_output.update(dist_best_gold=dist_diff / nsentences)
            if "dist_best_gold_unif" in logging_outputs[0]:
                dist_diff = float(sum(log.get("dist_best_gold_unif", 0) for log in logging_outputs))
                agg_output.update(dist_dist_best_gold=dist_diff / nsentences)

        return agg_output
