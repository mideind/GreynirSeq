import itertools
import math

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq import utils

import torch
import torch.nn.functional as F

from greynirseq.utils.ifd_utils import FEATS_MUTEX
from greynirseq.nicenlp.utils.logits_filter import (
    word_classes_to_mask,
    LABEL_GROUPS,
    max_tensor_by_bins,
)


@register_criterion("multi_label")
class BinaryMultiLabelCriterion(FairseqCriterion):
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert (
            hasattr(model, "classification_heads")
            and "multi_label_word_classification" in model.classification_heads
        ), "model must provide sentence classification head for --criterion=multi_label_word_classification"

        logits, _ = model(
            **sample["net_input"],
            features_only=True,
            classification_head_name="multi_label_word_classification",
        )
        word_logits, binary_logits = logits

        # (batchs, bpe_token)
        targets = model.get_targets(sample, [logits])
        bsz, num_words, num_word_cls = word_logits.shape
        _, _, num_bin_cls = binary_logits.shape
        # targets_hot:  (batches, word, word_classes + binary_classes)
        num_all_labels = num_word_cls + num_bin_cls
        targets_hot = torch.zeros(bsz, num_words, num_all_labels).cuda()
        num_special = self.task.label_dictionary.nspecial

        def targets_to_n_hot(sent, dim):
            words = [
                list(x[1])
                for x in itertools.groupby(
                    sent,
                    lambda x: x == self.task.span_separator
                    or x == self.task.label_dictionary.pad()
                    or x == 4 # Hack because double sep....,
                )
                if not x[0]
            ]
            nh = torch.zeros(dim[1], dim[2])
            for i, w in enumerate(words):
                for k in w:
                    nh[i][k.item() - num_special] = 1
            return nh
        
        for i in range(bsz):
            targets_hot[i] = targets_to_n_hot(targets[i], targets_hot.shape)

        word_sep = self.task.span_separator
        unknown_word_id = 2  # x token in data


        # Ignore unknown words
        x_count = (targets == unknown_word_id).sum()
        # Number of words to match, works since word_sep is also at end of last word

        # TODO RESOLVE TEMPORARY HACK
        word_count = (targets == word_sep).int().sum() / 2 - x_count # Divided by 2 because sep around sep... FIX
        # Number of labels times number of words
        sample_size = word_count * num_all_labels

        # (batch, words, word_class_labels)
        word_targets = targets_hot[:, :, :num_word_cls]

        # Cast max gives indice in indices type
        word_targets_idxs = (
            (
                word_targets.max(dim=2)[1].type(word_targets.dtype)
                + word_targets.max(dim=2)[0]
                - 1
            )
            .long()
            .view(-1)
        )

        # Null loss from unknown 'x' label
        word_cls_weights = word_logits.new_ones(num_word_cls)
        word_cls_weights[unknown_word_id] = 0

        # Todo, fix foo things
        foo0, foo1, foo2 = word_logits.shape
        word_cls_loss = F.cross_entropy(
            word_logits.reshape(foo0 * foo1, foo2),
            word_targets_idxs,
            reduction="sum",
            ignore_index=-1,
            weight=word_cls_weights
        )

        # bin_targets: (bsz, num_words, num_bin_labels)
        bin_targets = targets_hot[:, :, num_word_cls:]
        pos_weights = bin_targets.new_ones(num_bin_cls) * 10
        binary_class_loss = self.calc_class_loss(
            bin_targets, binary_logits, pos_weights, word_targets_idxs
        )
        loss = word_cls_loss + binary_class_loss

        logging_output = {
            "loss": utils.item(loss.data) if reduce else loss.data,
            "ntokens": sample["ntokens"],
            "sample_size": sample_size.item(),
            "nwords": word_count.data.item(),
            "nsentences": word_targets.shape[0],
        }

        _, wc_preds = word_logits.max(dim=-1)

        # Works because -1 above on target idxs, i.e. padding does not match
        word_targets_idx_reshaped = word_targets_idxs.reshape(bsz, num_words)
        wc_matches = wc_preds == word_targets_idx_reshaped
        ncs_1 = wc_matches.sum().item()

        b_preds = self.get_binary_predictions(
            binary_logits, word_targets_idxs, num_bin_cls
        ).type(binary_logits.dtype)

        correct = 0
        sc_correct = 0

        for sent_idx in range(bsz):
            # To ensure not matching over padding
            words_in_sent = word_targets[sent_idx].sum().int().item()
            for word_idx in range(words_in_sent):

                # Ignore unkown words when calculating accuracy
                if word_targets_idx_reshaped[sent_idx][word_idx] == unknown_word_id:
                    continue

                # TODO: rethink this hack for a* labels
                if wc_preds[sent_idx][word_idx].item() in list(range(27, 33)):
                    b_preds[sent_idx, word_idx, 18] = 0
                elif wc_preds[sent_idx][word_idx].item() == 23:
                    b_preds[sent_idx, word_idx, 11:12] = 0

                sc_match = (
                    bin_targets[sent_idx][word_idx].type(b_preds.dtype)
                    == b_preds[sent_idx][word_idx]
                ).all()
                sc_correct += int(sc_match)
                if wc_matches[sent_idx][word_idx] and sc_match:
                    correct += 1

        logging_output.update(wc_ncorrect=ncs_1)
        logging_output.update(bc_ncorrect=sc_correct)
        logging_output.update(ncorrect=correct)
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        nwords = sum(log.get("nwords", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        agg_output = {
            "loss": loss_sum / math.log(2),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "nwords": nwords,
        }

        if len(logging_outputs) > 0:
            if "ncorrect" in logging_outputs[0]:
                ncorrect = sum(log.get("ncorrect", 0) for log in logging_outputs)
                acc = ncorrect / nwords
                agg_output.update(accuracy=acc)
            if "bc_ncorrect" in logging_outputs[0]:
                ncorrect_bc = sum(log.get("bc_ncorrect", 0) for log in logging_outputs)
                acc_bc = ncorrect_bc / nwords
                agg_output.update(bc_accuracy=acc_bc)
            if "wc_ncorrect" in logging_outputs[0]:
                ncorrect_wc = sum(log.get("wc_ncorrect", 0) for log in logging_outputs)
                acc_wc = ncorrect_wc / nwords
                agg_output.update(wc_accuracy=acc_wc)

        if sample_size != ntokens:
            agg_output["nll_loss"] = loss_sum / ntokens / math.log(2)

        return agg_output

    def calc_class_loss(self, targets, logits, pos_weights, **wargs):
        return F.binary_cross_entropy_with_logits(
            logits,
            targets,
            weight=None,  # TODO possibly use weight
            reduction="sum",
            pos_weight=pos_weights,
        )

    @classmethod
    def get_binary_predictions(self, binary_logits, *args):
        return (binary_logits - 0.5 > 0).int()


@register_criterion("multi_label_idf")
class GeneralMultiLabelCriterion(BinaryMultiLabelCriterion):
    def calc_class_loss(self, targets, logits, pos_weights, word_targets_idxs):
        bsz, num_words, _ = logits.shape
        num_groups = len(LABEL_GROUPS)
        losses = logits.new_zeros((bsz, num_words, num_groups))
        # TODO group_idx
        for i, label_group in enumerate(LABEL_GROUPS):
            # TODO: Solve overlapping groups issue
            start, end = (
                label_group[0],
                label_group[-1] + 1,
            )  # + 1 to include last value
            label_group_logits = logits[:, :, start:end]
            # Use dimension of max value as target
            target_idxs = targets[:, :, start:end].max(dim=2)[1]
            num_members = end - start

            if num_members == 1:
                loss = F.binary_cross_entropy_with_logits(
                    label_group_logits, targets[:, :, start:end].type(label_group_logits.dtype), reduction="none"
                )
            else:
                loss = F.cross_entropy(
                    label_group_logits.reshape(bsz * num_words, num_members),
                    target_idxs.reshape(bsz * num_words),
                    reduction="none",
                )
            losses[:, :, i] = loss.reshape(bsz, num_words)

        label_mask = word_classes_to_mask(word_targets_idxs).reshape(
            bsz, num_words, len(LABEL_GROUPS)
        )
        return (losses * label_mask.type(losses.dtype)).sum()

    @classmethod
    def get_binary_predictions(cls, binary_logits, word_targets_idxs, num_bin_cls):

        binary_predictions = [13, 14]  # Definite article and proper noun

        lab_grp_cats = binary_logits.new_zeros(len(LABEL_GROUPS), num_bin_cls)
        bsz, words, _ = binary_logits.shape
        label_mask = word_classes_to_mask(word_targets_idxs).reshape(
            bsz, words, len(LABEL_GROUPS)
        )
        for i in range(len(LABEL_GROUPS)):
            lab_grp_cats[i, :] = F.one_hot(
                torch.tensor(LABEL_GROUPS[i]), num_bin_cls
            ).sum(dim=0)

        filtered_bin_logits = binary_logits.new_zeros(binary_logits.shape)
        for i in range(bsz):
            maxed = max_tensor_by_bins(
                binary_logits[i], LABEL_GROUPS, softmax_by_bin=True
            )

            filtered_bin_logits[i] = maxed * torch.mm(label_mask[i].float(), lab_grp_cats.float()).type(maxed.dtype)
            # Special case for when both gender and person
            for j in range(len(label_mask[i])):
                if label_mask[i][j][3] == 1:
                    max_v = filtered_bin_logits[i][j][:7].max(dim=-1)
                    filtered_bin_logits[i][j][:7] = filtered_bin_logits.new_zeros(7)
                    filtered_bin_logits[i][j][max_v[1]] = max_v[0]

        for bp in binary_predictions:
            filtered_bin_logits[:, :, bp][filtered_bin_logits[:, :, bp] < 0.5] = 0
        return (filtered_bin_logits > 0).int()
