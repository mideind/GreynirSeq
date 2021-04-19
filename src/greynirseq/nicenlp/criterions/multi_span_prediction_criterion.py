# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

# flake8: noqa

import math
from collections import namedtuple

import torch
import torch.nn.functional as F
from fairseq.criterions import FairseqCriterion, register_criterion

from greynirseq.nicenlp.utils.constituency import greynir_utils, tree_dist  # pylint: disable=no-name-in-module


def gen_2d_diags(chart_width):
    """Generator for all diagonal positions in a 2d matrix, starting with right of center diagonal from pos (0,1),
    then diagonal starting at (0,2), etc."""
    for span_length in range(1, chart_width):
        for start in range(chart_width - span_length):
            ii = start
            jj = start + span_length
            yield (ii, jj)


ParseResult = namedtuple("ParseResult", ["score", "spans", "labels", "masked_lchart"])


def parse_from_chart(scores, widths, scores_only=False, score_shift=None):
    # walk diagonal line starting at position (0, start)
    if score_shift is not None:
        chart, lchart = (scores + score_shift).max(dim=-1)
        chart = chart - score_shift.max(dim=-1)[0]
    else:
        chart, lchart = scores.max(dim=-1)

    bsz, chart_width, _ = chart.shape
    best_chart = torch.zeros_like(chart)
    kchart = torch.zeros_like(lchart)

    for ii, jj in gen_2d_diags(chart_width):
        if ii >= jj:
            continue
        label_score = chart[:, ii, jj]
        if ii + 1 == jj:
            best_chart[:, ii, jj] = label_score
            continue
        kk = jj - ii - 1
        k_horiz = best_chart[:, ii, (ii + 1) : jj]
        k_vert = best_chart[:, (ii + 1) : jj, jj]
        best_split, argmax_k = (k_horiz + k_vert).max(dim=-1)
        kchart[:, ii, jj] = argmax_k
        mask = widths.ge(jj).type(chart.dtype)  # mask out shorter sequences
        best_chart[:, ii, jj] = (label_score + best_split) * mask

    tree_scores = torch.stack([best_chart[seq_idx, 0, widths[seq_idx]] for seq_idx in range(bsz)])

    if scores_only:
        return tree_scores, None

    results = []
    masked_lcharts = []
    for seq_idx in range(bsz):
        ii, jj = 0, int(widths[seq_idx])
        stack = [(ii, jj)]
        spans = []
        while stack:
            ii, jj = stack.pop()
            if ii + 1 == jj:
                spans.append((ii, jj))
                continue
            kk = int(kchart[seq_idx, ii, jj] + 1)
            spans.append((ii, jj))
            stack.append((ii, ii + kk))
            stack.append((ii + kk, jj))

        # sort spans by pre-order
        spans = sorted(spans, key=lambda x: (x[0], -x[0]))
        spans_ii, spans_jj = zip(*spans)
        labels = lchart[seq_idx, spans_ii, spans_jj]
        masked_lchart = widths.new_zeros(widths[seq_idx], widths[seq_idx] + 1)
        masked_lchart[spans_ii, spans_jj] = labels

        res = ParseResult(tree_scores[seq_idx], spans, labels, masked_lchart)
        results.append(res)
    return tree_scores, results


@register_criterion("multi_span")
class MultiSpanCriterion(FairseqCriterion):
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
            and "multi_span_classification" in model.classification_heads
        ), "model must provide sentence classification head for --criterion=multi_label_word_classification"

        torch.autograd.set_detect_anomaly(True)

        ntargets = sample["ntargets"]
        target_labels = sample["targets"]
        target_spans = sample["target_spans"]
        nwords = sample["nwords"]
        ninp = sample["net_input"]
        sspans = ninp["src_spans"]
        nsspans = ninp["nsrc_spans"]
        nswidths = nsspans.float().sqrt().long()
        # loss_masks = sample["loss_masks"]

        bsz, _ = target_labels.shape
        max_ntargets = ntargets.max()

        max_nwords = nwords.max()
        # target_spans (bsz, 2 * max_ntargets)
        tl = target_labels
        tspans = target_spans
        tspans = tspans.reshape(bsz, -1, 2).permute(0, 2, 1)  # (bsz, 2, max_ntargets)
        ii = tspans[:, 0, :]
        jj = tspans[:, 1, :]

        (logits, cat_logits), _extra = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name="multi_span_classification",
        )

        scores = logits
        logprobs = F.log_softmax(logits, dim=-1)

        # assemble upper triangular score chart for CKY parsing algorithm
        # go from (bsz, max_nsrc_spans, num_labels)
        # to (bsz, max_nwords +1, max_nwords, num_labels) aligned for each sequence
        num_labels = model.task.num_labels
        assert num_labels == model.classification_heads["multi_span_classification"].num_labels
        chart_scores = scores.new_zeros(bsz, max_nwords + 1, max_nwords + 1, num_labels)
        # we dont copy gradient, we only use these to force the
        # parser to select the right labels/spans
        max_chart_score = scores.max().data
        min_chart_score = scores.min().data
        chart_mask = torch.zeros_like(chart_scores)
        chart_logprobs = scores.new_zeros(bsz, max_nwords + 1, max_nwords + 1, num_labels)
        for seq_idx in range(bsz):
            width = nswidths[seq_idx]
            square = scores[seq_idx, : nsspans[seq_idx], :].reshape(width, width, num_labels)
            chart_scores[seq_idx, :width, 1 : 1 + width, :] = square
            # we prohibit chart decoding to select NULL as the root label
            chart_scores[seq_idx, 0, width, 0] = min_chart_score - 1
            chart_mask[seq_idx, :width, 1 : 1 + width, :] = (
                torch.triu(chart_mask.new_ones(width, width)).unsqueeze(-1).expand_as(square)
            )
            chart_logprobs[seq_idx, :width, 1 : 1 + width, :] = logprobs[seq_idx, : nsspans[seq_idx], :].reshape(
                width, width, num_labels
            )
        chart_logprobs *= chart_mask  # make upper triangular
        chart_scores *= chart_mask

        # extract targets from padded inputs
        # we could instead flatten with reshape(-1) and make use of ignore_index in nll_loss
        # but we want various accuracy metrics?
        chart_mask_not_gold = chart_mask.clone()
        label_shift = model.task.label_dictionary.nspecial
        rect_pred, rect_tgts = [], []
        for seq_idx in range(bsz):
            seq_ntargets = ntargets[seq_idx]
            seq_labels = target_labels[seq_idx, :seq_ntargets]
            seq_tgt_ii, seq_tgt_jj = tspans[seq_idx, :, :seq_ntargets].chunk(2, dim=0)
            seq_pred_tgts = chart_logprobs[seq_idx, seq_tgt_ii, seq_tgt_jj, :].squeeze(0)
            if not seq_labels.le(num_labels + label_shift).all():
                import pdb

                pdb.set_trace()
            chart_mask_not_gold[seq_idx, seq_tgt_ii, seq_tgt_jj, seq_labels - label_shift] = 0
            rect_pred.append(seq_pred_tgts)
            rect_tgts.append(seq_labels)

        chart_mask_gold = chart_mask - chart_mask_not_gold
        rect_pred = torch.cat(rect_pred, dim=0)
        rect_tgts = torch.cat(rect_tgts, dim=0)

        # this augmentation forces max to return the gold labels, so that parsing automatically selects gold tree
        max_dist = 1.1 * (max_chart_score - min_chart_score)
        augment_gold = chart_mask_gold * max_dist
        tree_scores_gold, results_gold = parse_from_chart(
            chart_scores, nswidths, scores_only=model.training, score_shift=augment_gold
        )
        tree_scores_best, results_best = parse_from_chart(
            chart_scores,
            nswidths,
            # we extract labels to calculate label_precision
            scores_only=False,
        )
        # this augmentation forces max to select any label except the gold label, so that
        # parsing constructs a high scoring bad tree
        # tree_scores_bad, tree_spans_bad, tree_labels_bad = parse_from_chart(
        tree_scores_bad, results_bad = parse_from_chart(
            chart_scores,
            nswidths,
            scores_only=model.training,
            score_shift=chart_mask_not_gold,
        )

        def parse_result_to_tree(parse_result):
            labels_str = [model.task.label_dictionary.symbols[idx + label_shift] for idx in parse_result.labels]
            tree = greynir_utils.Node.from_labelled_spans(parse_result.spans, labels_str).debinarize()
            return tree

        trees_best = list(map(parse_result_to_tree, results_best))

        """
        in kitaevs implementation:
        hamming_loss is scaled up by 5
        span classifier/scorer consists of (note: no softmax):
            linear > layernorm > relu > linear
        TODO:
            NOTE: not yet implemented
            kitaevs bert-based parser uses self-attention on top of bert representations!
            thus we need to implement:
            2 layers (self-attention -> ffn) before MLP span classification
            the parser part is at word level (instead of subword level) using last vector of the word
                the (pos/char/affix) features are then (concatenated/added) to reduced bert features before feeding to parser
            uses separate MLP span classifier for different languages (tagsets)
            ensembling (4) adds betwen 0.15 and 0.2 to F1

        consider using a combining dataset similar to this:
            /home/haukur/github/fairseq/fairseq/data/language_pair_dataset.py
            {
                index: ..., ?
                source: ...,
                nterm_spans: ...,
                nterm_targets: ...,
                term_spans: ...,
                term_targets: ...,
            }

        until then:
            terms are can be fit to labelled span dataset
            need to reintroduce label groups, label categories
            implement loss mask datasets for label groups/categories
            change span_classification_head into constituency_classification_head
                project into word cats
                concatenate word cat features with span features
                    compute word attribute features

        NOTE: see notes on kitaevs paper above!
        now there is a slight dilemma, we can condition nonterminal span labeling on POS features but
        the question is how to do that?
            - one approach is to compute POS features from layer ~6 features, add or concatenate them to layer 6 features
                before projecting them to encoder_embed_dim and feeding them to layer 7
                (add them to initial bpe token or all bpe tokens of a given word?)
            - two feed-throughs, first to compute POS, second augmented with fresh POS features to compute constituency labels?
        concatenating POS features to roberta extracted_features is not good enough since the
            nonterminal classifier only sees the endpoints of a span


        span_classification_head can then be for (NER/grammaticality/terminals as a separate task)



        """
        label_hamming_losses = 5 * F.nll_loss(rect_pred, rect_tgts - label_shift, reduction="none")
        label_hamming_losses = torch.stack(
            [seq_loss.sum() for seq_loss in label_hamming_losses.split(ntargets.tolist())]
        )

        # clamp for hinge loss
        loss = (label_hamming_losses + tree_scores_bad.view(-1).sum() - tree_scores_gold.view(-1).sum()).clamp(0).sum()

        null_leaf_idx = model.task.label_dictionary.index("LEAF")
        ncorrect_labels_best_roof = []
        ncorrect_spans_best_roof = []
        nlabels_best_roof = []
        bracketing_errors = []
        for seq_idx, presult in enumerate(results_best):
            seq_ntargets = ntargets[seq_idx]
            seq_targets = target_labels[seq_idx, :seq_ntargets]
            seq_tgt_ii, seq_tgt_jj = tspans[seq_idx, :, :seq_ntargets].chunk(2, dim=0)
            pred_labels_best = presult.masked_lchart[seq_tgt_ii, seq_tgt_jj]
            seq_ncorrect_labels = pred_labels_best.gt(1) * (pred_labels_best == (seq_targets - label_shift))
            seq_ncorrect_spans = pred_labels_best.gt(1)
            ncorrect_spans_best_roof.append(seq_ncorrect_spans.sum())
            ncorrect_labels_best_roof.append(seq_ncorrect_labels.sum())
            seq_nlabels = presult.labels.gt(1)
            nlabels_best_roof.append(seq_nlabels.sum())
            seq_bracketing_errors = seq_nlabels.sum() + seq_ntargets - 2 * seq_ncorrect_spans.sum()
            bracketing_errors.append(seq_bracketing_errors)

        nwords_total = nwords.sum().data
        logging_output = {
            "loss": loss.data,
            "label_loss": label_hamming_losses.sum().data,
            "gold_tree": tree_scores_gold.sum().data,
            "worst_tree": tree_scores_bad.sum().data,
            "best_tree": tree_scores_best.sum().data,
            "ntokens": sample["ntokens"],
            "nwords": nwords_total,
            "nsentences": bsz,
            "sample_size": nwords_total,
            "ncorrect_labels": sum(ncorrect_labels_best_roof),
            "ncorrect_spans": sum(ncorrect_spans_best_roof),
            # filter out NULL labels
            "gold_nlabels": target_labels.gt(label_shift).sum(),
            "gold_nlabels_roof": (target_labels[target_labels != null_leaf_idx] > label_shift).sum(),
            "best_nlabels_roof": sum(nlabels_best_roof),
            "bracketing_errors": sum(bracketing_errors),
        }

        if not model.training:
            trees_gold = list(map(parse_result_to_tree, results_gold))
            trees_bad = list(map(parse_result_to_tree, results_bad))

            logging_output["dist_gold_bad"] = sum(
                [tree_dist.tree_dist(gold, bad, None) for gold, bad in zip(trees_gold, trees_bad)]
            )
            logging_output["dist_best_gold"] = sum(
                [tree_dist.tree_dist(best, gold, None) for best, gold in zip(trees_best, trees_gold)]
            )

            dist_best_gold_unif = 0.0
            for best, gold in zip(trees_best, trees_gold):
                best = best.roof()
                gold = gold.roof()
                dist = 0.0
                if best is not None and gold is not None:
                    dist = tree_dist.tree_dist(best.uniform(), gold.uniform(), None)
                dist_best_gold_unif += dist
            logging_output["dist_best_gold_unif"] = dist_best_gold_unif
        return loss, nwords_total, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0.0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        nwords = int(sum(log.get("nwords", 0) for log in logging_outputs))
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        gold_tree = float(sum(log.get("gold_tree", 0) for log in logging_outputs))
        best_tree = float(sum(log.get("best_tree", 0) for log in logging_outputs))
        worst_tree = float(sum(log.get("worst_tree", 0) for log in logging_outputs))

        gold_nlabels = float(sum(log.get("gold_nlabels", float("inf")) for log in logging_outputs))
        gold_nlabels_roof = float(sum(log.get("gold_nlabels_roof", float("inf")) for log in logging_outputs))
        best_nlabels_roof = float(sum(log.get("best_nlabels_roof", float("inf")) for log in logging_outputs))
        ncorrect_labels = float(sum(log.get("ncorrect_labels", 0) for log in logging_outputs))
        ncorrect_spans = float(sum(log.get("ncorrect_spans", 0) for log in logging_outputs))
        bracketing_errors = int(sum(log.get("bracketing_errors", 0) for log in logging_outputs))

        label_loss = float(sum(log.get("label_loss", 0.0) for log in logging_outputs))
        # loss for NULL is always 0 so we dont count it in total
        label_loss_ = float(label_loss / gold_nlabels / math.log(2))

        label_recall = float(ncorrect_labels) / float(gold_nlabels_roof)
        label_precision = float(ncorrect_labels) / float(best_nlabels_roof)
        bracketing_precision = float(ncorrect_spans) / float(best_nlabels_roof)
        bracketing_recall = float(ncorrect_spans) / float(gold_nlabels_roof)

        agg_output = {
            "loss": float(loss_sum / math.log(2)),  # / sample_size / math.log(2),
            "label_hamming": label_loss_,
            "ppl": label_loss_,
            "label_precision": label_recall,
            "label_recall": label_precision,
            "label_f1": 2 / (1 / (label_precision or 1) + 1 / (label_recall or 1)),
            "bracketing_precision": bracketing_precision,
            "bracketing_recall": bracketing_recall,
            "bracketing_f1": 2 / (1 / (bracketing_precision or 1) + 1 / (bracketing_recall or 1)),
            "avg_bracketing_errors": bracketing_errors / (nsentences),
            "delta_gold_worst": (gold_tree - worst_tree) / nsentences,
            "delta_best_gold": (best_tree - gold_tree) / (nsentences),
            "gold_tree": gold_tree / nsentences,
            "worst_tree": worst_tree / nsentences,
            "best_tree": best_tree / nsentences,
            "ntokens": ntokens,
            "nwords": nwords,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        if len(logging_outputs) > 0:
            if "dist_gold_bad" in logging_outputs[0]:
                dist_diff = float(sum(log.get("dist_gold_bad", 0) for log in logging_outputs))
                agg_output.update(dist_gold_bad=dist_diff)
            if "dist_best_gold" in logging_outputs[0]:
                dist_diff = float(sum(log.get("dist_best_gold", 0) for log in logging_outputs))
                agg_output.update(dist_best_gold=dist_diff)
            if "dist_best_gold_unif" in logging_outputs[0]:
                dist_diff = float(sum(log.get("dist_best_gold_unif", 0) for log in logging_outputs))
                agg_output.update(dist_best_gold_unif=dist_diff)

        return agg_output
