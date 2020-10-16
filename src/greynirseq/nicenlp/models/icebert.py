import itertools

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from fairseq import utils
from fairseq.models.roberta.model import (
    base_architecture,
    RobertaModel,
)
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models import (
    register_model,
    register_model_architecture,
)

from greynirseq.nicenlp.criterions.multi_label import GeneralMultiLabelCriterion
from greynirseq.utils.ifd_utils import vec2idf, FEATS_MUTEX_MAP_IDX
from greynirseq import settings


class BinaryClassifierChain(nn.Module):
    def __init__(self, embed_dim, num_word_classes, num_bin_features):
        super().__init__()
        self.num_bin_features = num_bin_features
        self.embed_dim = embed_dim
        self.num_word_classes = num_word_classes

        self.dense = nn.ModuleList(
            [
                nn.Linear(embed_dim + num_word_classes + num_bin_features, 1)
                for _ in range(num_bin_features)
            ]
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, features, word_class_features):
        # features:          (bsz, num_words, hidden_dim)
        # word_cls_features: (bsz, num_words, num_bin_labels)
        word_class_features = F.softmax(word_class_features)
        # (batch, words, bin_features)
        bsz, num_words, _ = features.shape
        bin_features = features.new_zeros(bsz, num_words, self.num_bin_features)
        # all_features:
        # (bsz, num_words, hidden_dim + word_class_features + bin_features)
        all_features = torch.cat((features, word_class_features, bin_features), 2)
        offset = self.embed_dim + self.num_word_classes
        for class__idx, dense in enumerate(self.dense):
            logit = dense(all_features)
            prob = self.sigmoid(logit)
            one_hot = logit.new_zeros(bsz, num_words, offset + self.num_bin_features)
            one_hot[:, :, offset + class__idx] = prob.squeeze()
            all_features = all_features + one_hot
        # (bsz, num_words, num_bin_labels)
        return all_features[:, :, offset:]


class MultiLabelClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim,
        inner_dim,
        num_classes_mutex,
        num_classes_binary,
        activation_fn,
        pooler_dropout,
    ):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_classes_mutex = num_classes_mutex
        self.num_classes_binary = num_classes_binary

        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes_mutex)
        self.binary_out_proj = nn.Linear(
            inner_dim + num_classes_mutex, num_classes_binary
        )
        # TODO: compare with BCC
        # self.classifier_chain = BinaryClassifierChain(
        #    inner_dim, num_classes_mutex, num_classes_binary
        # )

    @staticmethod
    def get_word_start_indexes(word_starts):
        return [i for i in range(len(word_starts)) if word_starts[i]]

    @staticmethod
    def tokens_to_range_sum(word_starts, bpe_token_features, inner_dim, device):
        nsentences = len(word_starts)
        max_word_length = max(sum(w) for w in word_starts)

        word_features = torch.ones(
            (nsentences, max_word_length, inner_dim), device=device
        )

        for j in range(nsentences):
            word_start_idxs = MultiLabelClassificationHead.get_word_start_indexes(
                word_starts[j]
            )
            nwords = len(word_start_idxs)
            sentence_bpe_tokens = bpe_token_features[j]

            for i in range(nwords):
                word_range = word_start_idxs[i]
                if i < nwords - 1:
                    word_end = word_start_idxs[i + 1]
                    tokens_in_word = sentence_bpe_tokens[word_range:word_end]
                else:
                    tokens_in_word = sentence_bpe_tokens[word_range:]
                word_features[j][i] = sum(tokens_in_word)
        return word_features.type(bpe_token_features.dtype)

    def forward(self, features, **kwargs):
        # [num_seqs, max_seq_len, hidden_dim]
        x = features[:, 1:-1]  # dont classify BOS token and EOS token !
        x = self.tokens_to_range_sum(
            kwargs["src_spans"], x, self.inner_dim, device=x.device
        )
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        word_class_logits = self.out_proj(x)
        binary_logits = self.binary_out_proj(torch.cat((word_class_logits, x), dim=-1))
        # TODO: see comment above on BCC
        # binary_logits = self.classifier_chain(x, word_class_logits)
        # (batch, words, word_class_features), (batch, words, binary_features)
        return word_class_logits, binary_logits


@register_model("icebert")
class IcebertModel(RobertaModel):

    @staticmethod
    def pos_from_settings():
        return IcebertModel.from_pretrained(
            settings.IceBERT_POS_PATH,
            **settings.IceBERT_POS_CONFIG
        )

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs
    ):

        if classification_head_name is not None:
            features_only = True
        x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x, **kwargs)
        return x, extra

    def register_classification_head(
        self,
        name,
        num_classes_mutex=None,
        num_classes_binary=None,
        inner_dim=None,
        **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes_mutex != prev_num_classes or inner_dim != prev_inner_dim:
                print(
                    'WARNING: re-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name,
                        num_classes_mutex,
                        prev_num_classes,
                        inner_dim,
                        prev_inner_dim,
                    )
                )
        self.classification_heads[name] = MultiLabelClassificationHead(
            self.args.encoder_embed_dim,
            inner_dim or self.args.encoder_embed_dim,
            num_classes_mutex,
            num_classes_binary,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        **kwargs
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return IcebertHubInterface(x["args"], x["task"], x["models"][0])


class IcebertHubInterface(RobertaHubInterface):
    def predict_file(self, input_path, label_path, batch_size=1, device="cuda"):
        hits = 0
        total = 0
        separator = "<SEP>"

        with open(input_path) as input_file, open(label_path) as label_file:
            batch_input = []
            label_input = []
            label_tail_input = []
            for sent, labels in zip(input_file, label_file):
                sent = sent.strip()
                labels = labels.strip().split(separator)
                head_labels = [lab.split()[0] for lab in labels]
                tail_labels = [lab.split()[1:] for lab in labels]

                batch_input.append(sent)
                label_input.append(head_labels)
                label_tail_input.append(tail_labels)

                if len(batch_input) == batch_size:
                    word_cats_pr, sub_cats_pr = self.predict(batch_input, device=device)

                    for sent, pr, scpr, wc_true, sc_true in zip(
                        batch_input,
                        word_cats_pr,
                        sub_cats_pr,
                        label_input,
                        label_tail_input,
                    ):
                        pr = ["{:>2s}".format(foo) for foo in pr[: len(wc_true)]]
                        wc_true = ["{:>2s}".format(foo) for foo in wc_true]
                        scpr = scpr[: len(sc_true)]

                        # print("-----")
                        # print(sent)
                        # print("Pred:\t{}".format(" ".join(pr)))
                        # print("True:\t{}".format(" ".join(wc_true)))

                        # print(
                        #    "Sub pred:\t{}".format(
                        #        " | ".join([" ".join(sc_pr) for sc_pr in scpr])
                        #    )
                        # )
                        # print(
                        #    "Sub true:\t{}".format(
                        #        " | ".join([" ".join(sc) for sc in sc_true])
                        #    )
                        # )

                        no_tokens = len(sc_true)
                        total += no_tokens
                        for i in range(no_tokens):
                            if set(scpr[i]) == set(sc_true[i]):
                                hits += 1

                    batch_input = []
                    label_input = []
                    label_tail_input = []
        print("Accuracy: {}%".format(100.0 * hits / total))

    def predict(self, sentences, offset=5, device="cuda", return_labels=True):
        device = torch.device(device)

        tokens = [self.encode(sentence).to(device=device) for sentence in sentences]
        tokens = pad_sequence(
            tokens, padding_value=self.task.label_dictionary.pad()
        ).t()

        spans = []
        for sentence in tokens:
            spans.append([self.task.is_word_begin[t.item()] for t in sentence])

        features = self.extract_features(torch.tensor(tokens)).to(device)

        # to make new tensor use same device as another tensor, use
        # new_tensor = old_tensor.new(dim1, dim2, ...)
        src_spans = (torch.tensor(spans)).to(device)
        nspans = torch.tensor([sum(span) for span in spans]).to(device)

        logits, binary_logits = self.model.classification_heads[
            "multi_label_word_classification"
        ](features, src_spans=src_spans, nspans=nspans)

        prs = F.softmax(logits, dim=-1)
        # Off by one to not predict BOS
        preds_idxs = prs.max(dim=-1)[1] - 1

        if not return_labels:
            preds = preds_idxs
        else:
            preds = []
            for sent in preds_idxs:
                pred = [
                    self.task.label_dictionary.string([idx + offset]) for idx in sent
                ]
            preds.append(pred)

        pr_bin = GeneralMultiLabelCriterion.get_binary_predictions(
            binary_logits, preds_idxs.view(-1), 26
        )
        # pr_bin = (binary_logits - 0.5 > 0).int()
        pred_bin = []
        for sent_idx in range(len(pr_bin)):
            pr = pr_bin[sent_idx]
            pred = []
            num_words, num_labels = pr.shape
            for j in range(num_words):
                word_pred = []
                v = [i for i in range(num_labels) if pr[j][i]]
                for k in v:

                    # Hack to fix POS (null bin) for a* labels
                    if (
                        preds_idxs[sent_idx][j].item() in list(range(27, 33))
                        and k == 18
                    ):
                        continue

                    # Hack for sþ, no acc, gen
                    if preds_idxs[sent_idx][j].item() == 23 and k in [11, 12]:
                        continue

                    # TODO remove this 34
                    if return_labels:
                        word_pred.append(
                            self.task.label_dictionary.string([k + 34 + offset])
                        )
                    else:
                        word_pred.append(k + 34)
                pred.append(word_pred)
            pred_bin.append(pred)
        return preds, pred_bin

    def predict_to_idf(self, sentence, device="cuda"):
        wc_pred, bin_pred = self.predict([sentence], return_labels=False, device=device)

        # TODO, remove hard coding
        num_labels = 61

        labels = []

        for wc, sc_labels in zip(wc_pred[0], bin_pred[0]):
            word_vec = [wc] + sc_labels
            one_hot_word = F.one_hot(torch.tensor(word_vec), num_labels)
            labels.append(vec2idf(one_hot_word))

        return labels[:-1]


@register_model_architecture("icebert", "icebert")
def roberta_base_architecture(args):
    base_architecture(args)
