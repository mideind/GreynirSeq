from typing import List, Union, Dict, Any
import itertools
import math
import time
from collections import namedtuple

from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.models import FairseqModel
from fairseq import utils

import torch
import torch.nn.functional as F

from greynirseq.types import Numeric


@register_criterion("multilabel_token_classification")
class MultiLabelTokenClassificationCriterion(FairseqCriterion):
    def forward(self, model: FairseqModel, sample: Dict[str, Any], reduce: bool = True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(
            model, "task_head"
        ), "model must provide task specific classification head"

        target_cats = sample["target_cats"] 
        target_cats_mask = sample["exclude_cats_mask"]
        target_attrs = sample["target_attrs"]
        nwords = sample["nwords"]

        bsz, _max_nwords = target_cats.shape
        bsz, _max_nwords, _num_attrs = target_attrs.shape

        (cat_logits, attr_logits), _extra = model(
            **sample["net_input"], features_only=True
        )

        pad_idx = model.task.label_dictionary.pad()
        target_mask = target_cats.ne(pad_idx)

        device = cat_logits.device
        cat_dict_to_vec_idx = model.task.cat_dict_idx_to_vec_idx.clone().to(device)

        # Batch x Time x Depth -> Batch x Depth x Time
        cat_loss = F.cross_entropy(
            cat_logits.transpose(2, 1),
            cat_dict_to_vec_idx[target_cats],
            reduction="none",
        )
        # MASK AWAY LOSS FOR PADD NOT NEEDED?
        cat_loss = (cat_loss * target_cats_mask).sum()

        # each attribute group is one-hot, so Attr is multi-hot
        
        # attr_logits:  Batch x Time x Attr
        group_losses = []
        correct_attrs = torch.ones_like(target_cats).bool()

        group_name_to_group_attr_vec_idxs = {}
        for k, v in self.task.group_name_to_group_attr_vec_idxs.items():
            group_name_to_group_attr_vec_idxs[k] = v.clone().to(device)
        group_names = self.task.label_schema.group_names
        group_masks = self.task.group_mask.clone().to(device)
        num_cats = len(self.task.label_schema.label_categories)

        missing_binary_targets = torch.zeros_like(target_attrs)

        # we want fixed iteration order of group names
        for i, group_name in enumerate(group_names):
            group_idxs = group_name_to_group_attr_vec_idxs[group_name]
            _wrong_group_loss_mask, group_targets = target_attrs[:, :, group_idxs].max(dim=-1)
            target_pad_val = -100
            cat_vec_idxs = cat_dict_to_vec_idx[target_cats.clone()]
            cat_vec_idxs = cat_vec_idxs * cat_vec_idxs.ne(target_pad_val)

            group_loss_mask = group_masks[cat_vec_idxs][:,:,i]  # disregard padding
            #group_loss_mask = group_loss_mask * group_loss_mask.ne(target_pad_val)  # set padding to zero
            group_logits = attr_logits[:, :, group_idxs]

            if group_idxs.numel() == 1:
                #group_loss_mask = group_targets = target_attrs[:, :, group_idxs[0]]
                missing_binary_targets[:, :, group_idxs[0]] = 1
                missing_binary_targets[:, :, group_idxs[0]] *= group_loss_mask
                group_targets = target_attrs[:, :, group_idxs[0]]

                # group_targets = group_loss_mask * group_idxs[0]
                group_logits = attr_logits[:, :, group_idxs[0]]

                group_loss = F.binary_cross_entropy_with_logits(
                    group_logits.squeeze(), group_targets.type_as(attr_logits).squeeze(), reduction="none"
                ) 
                group_loss = group_loss * group_loss_mask.type_as(group_logits).squeeze()
                group_loss = group_loss * target_cats_mask
                group_losses.append(group_loss)
    
                correct = (
                    group_logits.ge(0).int() == group_targets
                ) * group_loss_mask.bool() * target_cats_mask.bool()
                #import pdb; pdb.set_trace()
            
            else:
                # Batch x Time x Depth -> Batch x Depth x Time
                group_loss = F.cross_entropy(
                    group_logits.transpose(2, 1).squeeze(), group_targets.squeeze(), reduction="none"
                )
                group_loss = group_loss * group_loss_mask.type_as(group_logits).squeeze()
                group_loss = group_loss * target_cats_mask
                group_losses.append(group_loss)
                
                # import pdb; pdb.set_trace()

                correct = (
                    group_logits.max(dim=-1).indices == group_targets
                ) * group_loss_mask.bool() * target_cats_mask.bool()
            
            # Correct attrs starts true for all words
            # then a single incorrect should flip the word
            # Just need to make sure to only flip those that have current group attrs!

            ignore_mask = group_loss_mask.bool().bitwise_not()
            
            #import pdb; pdb.set_trace()
            correct_attrs *= correct + ignore_mask

        #from pprint import pprint
        preds = self.task.logits_to_labels(cat_logits, attr_logits, sample["net_input"]["word_mask"])
        #pprint(preds[0])

        correct_attrs *= target_mask * target_cats_mask.bool()  # Only count words, not padding
        #correct_attrs *= target_cats_mask.bool()  # Ignore loss on words with exclusion category
        
        correct_cat = ((
            cat_logits.max(-1).indices == cat_dict_to_vec_idx[target_cats]
        ) * target_mask.bool() ) * target_cats_mask.bool()

        nwords_total = nwords.sum().item() - (target_cats_mask == 0).sum().item()
       
        correct_all = correct_attrs * correct_cat 
        
        if False:
            correct = 0
            seen = 0
            correct_att = 0
            correct_both = 0
            for j in range(target_cats.shape[0]):
                targ_cats = [self.task._label_dictionary.symbols[i] for i in target_cats[j]]
                for i, pc in enumerate(preds[j]):
                    
                    targ_attrs = [self.task._label_dictionary.symbols[k + self.task._label_dictionary.nspecial] for k in target_attrs[j][i].nonzero()]
                    
                    if targ_cats[i] not in self.task.label_schema.ignore_categories:
                        seen += 1
                        same_cat = pc[0] == targ_cats[i]
                        same_attr = set(pc[1]) == set(targ_attrs)
                        if same_cat:
                            correct +=1
                        if same_attr:
                            correct_att += 1
                        if same_cat and same_attr:
                            correct_both += 1
                        else:
                           print("{}\t{}\t{}".format(pc[0], targ_cats[i], same_cat))
                           print("\t\t{}\t--\t{}\t\t{} - {}".format(pc[1], targ_attrs, same_attr, correct_attrs[j][i]))
                    print("-{}-{}-".format(j, i))
            print("CATEGORY Seen: {} vs {}. Correct: {} vs {}. Correct cats: \t{}".format(seen, nwords_total, correct, correct_cat.sum(), correct/seen))
            print("ATTRS Correct: {} vs {}. Correct attrs:\t{}".format(correct_att, correct_attrs.sum(), correct_att/seen))
            print("Correct both: {} vs {}.".format(correct_both, correct_all.sum()))
            import pdb; pdb.set_trace()
        # NOTE: does not suffice for the binary labels since 0 has a meaning, hence adding missing_binary_targets
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
