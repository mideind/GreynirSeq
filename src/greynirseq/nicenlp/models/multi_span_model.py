import itertools

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from fairseq import utils
from fairseq.models.roberta.model import base_architecture, RobertaModel
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models import register_model, register_model_architecture
from fairseq.modules import LayerNorm
from fairseq.data import (
    RightPadDataset,
    BaseWrapperDataset,
    NestedDictionaryDataset,
    TokenBlockDataset,
    NumelDataset,
)

import nltk
import tokenizer

from greynirseq.nicenlp.data.datasets import (
    WordEndMaskDataset,
    LabelledSpanDataset,
    SpanDataset,
    SparseProductSpanDataset,
    ProductSpanDataset,
    NumSpanDataset,
    NestedDictionaryDatasetFix,
    NestedDictionaryDatasetFix2,
    LossMaskDataset,
    NumWordsDataset,
)

from greynirseq.nicenlp.criterions.multi_span_prediction_criterion import (
    parse_from_chart,
    sparse_to_chart,
)

from greynirseq.nicenlp.utils.greynir.greynir_utils import Node

try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


class MultiLabelSequenceTaggingHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, in_features, out_features, num_cats, pooler_dropout):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features  # num_labels
        self.num_cats = num_cats

        self.dense = nn.Linear(self.in_features, self.in_features)
        self.activation_fn = utils.get_activation_fn("relu")
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.layer_norm = LayerNorm(self.in_features)
        self.cat_proj = nn.Linear(self.in_features, self.num_cats)
        self.out_proj = nn.Linear(self.in_features + self.num_cats, self.out_features)

    def forward(self, features, **kwargs):
        mask = kwargs["word_mask_w_bos"]
        words_w_bos = features.masked_select(mask.unsqueeze(-1).bool()).reshape(
            -1, self.in_features
        )
        x = self.dropout(words_w_bos)
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.activation_fn(x)

        cat_logits = self.cat_proj(x)
        cat_probits = torch.softmax(cat_logits, dim=-1)
        attr_logits = self.out_proj(torch.cat((x, cat_probits), -1))

        return cat_logits, attr_logits, words_w_bos


class MultiSpanClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self, input_dim, inner_dim, num_cats, num_labels, activation_fn, pooler_dropout
    ):
        super().__init__()
        self.inner_dim = inner_dim
        self.num_labels = num_labels
        self.num_cats = num_cats

        # self.dense = nn.Linear(2 * input_dim, inner_dim)
        self.dense = nn.Linear(2 * input_dim, 2 * input_dim)
        assert ic
        # self.activation_fn = utils.get_activation_fn(activation_fn)  # defaults to tanh
        self.activation_fn = utils.get_activation_fn("relu")
        self.dropout = nn.Dropout(p=pooler_dropout)
        if self.num_cats:
            self.layer_norm = LayerNorm(2 * self.inner_dim)
            self.cat_proj = nn.Linear(2 * self.inner_dim, self.num_cats)
            self.out_proj = nn.Linear(
                2 * self.inner_dim + self.num_cats, self.num_labels
            )
        else:
            self.layer_norm = LayerNorm(2 * self.inner_dim)
            self.out_proj = nn.Linear(
                2 * self.inner_dim, self.num_labels - 1
            )  # don't include parameters for NULL label

    @classmethod
    def compute_span_features(cls, features, spans, nspans, **kwargs):
        """
        we need to compute all spans
        spans: (num_seqs, seq_len, num_words * num_words)
        spans : [..., (w_i_s, word_j_e), ...]  # indices into bpe tokens vector

        want to return:
        [w_i_s + word_j_e ; w_{i+1}_s + word_{j+1}_e]
        """
        bsz, ntokens, hidden_dim = features.shape
        max_nspans = nspans.max()
        # spans: (num_seqs, 2 * num_spans)
        spans = spans.reshape(bsz, max_nspans, 2)
        starts = spans[:, :, 0]
        ends = spans[:, :, 1]
        starts_ = torch.stack(
            [
                features[seq_idx].index_select(0, starts[seq_idx])
                for seq_idx in range(bsz)
            ],
            0,
        )
        ends_ = torch.stack(
            # we subtract 1 because these denote fence posts
            [
                features[seq_idx].index_select(0, ends[seq_idx] - 1)
                for seq_idx in range(bsz)
            ],
            0,
        )
        # ic((ends - starts)[-2])
        return torch.cat((starts_, ends_), dim=-1)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.compute_span_features(
            x, kwargs["src_spans"], kwargs["nsrc_spans"], **kwargs
        )
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.activation_fn(x)
        span_cat_logits = None
        if self.num_cats:
            raise NotImplementedError("")
            span_cat_logits = self.cat_proj(x)
            span_cat_probits = torch.softmax(span_cat_logits, dim=-1)
            final_logits = self.out_proj(torch.cat((x, span_cat_probits), -1))
        else:
            final_logits = self.out_proj(x)
            (
                bsz,
                nspans,
                _,
            ) = final_logits.shape
            # final_logits: (bsz, max_nwords * max_nwords, num_labels - 1)
            # make label_idx of 0 correspond to NULL label, fixed at label-span score of 0
            final_logits = torch.cat(
                [final_logits.new_zeros(bsz, nspans, 1), final_logits], -1
            )

        # nans = torch.any(torch.isnan(final_logits))
        # infs = torch.any(torch.isinf(final_logits))
        # assert not (nans or infs)), "Unexpected nan/inf"

        return final_logits, span_cat_logits


@register_model("icebert_const")
class IcebertConstModel(RobertaModel):
    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        classification_head_name=None,
        **kwargs,
    ):

        if classification_head_name is not None:
            features_only = True
        x, extra = self.encoder(src_tokens, features_only, return_all_hiddens, **kwargs)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x, **kwargs)
        return x, extra

    def register_classification_head(
        self, name, num_cats=None, num_labels=None, inner_dim=None, **kwargs
    ):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_labels != prev_num_classes or inner_dim != prev_inner_dim:
                print(
                    'WARNING: re-registering head "{}" with num_classes {} (prev: {}) '
                    "and inner_dim {} (prev: {})".format(
                        name, num_labels, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )

        # self.classification_heads[name] = MultiSpanClassificationHead(
        self.classification_heads[name] = MultiLabelSequenceTaggingHead(
            in_features=self.args.encoder_embed_dim,
            out_features=num_labels,
            num_cats=num_cats,
            pooler_dropout=self.args.pooler_dropout,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        **kwargs,
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
        return IcebertConstHubInterface(x["args"], x["task"], x["models"][0])


def tokenize(text, allow_multiword=False):
    mw_tokens = [tok.txt for tok in tokenizer.tokenize(text) if tok.txt is not None]
    if allow_multiword:
        return mw_tokens
    tokens = []
    for mw_token in mw_tokens:
        tokens.extend(mw_token.split(" "))
    return tokens


def tokenize_to_string(text):
    return " ".join(tokenize(text))


class IcebertConstHubInterface(RobertaHubInterface):
    def predict_file(self, input_path, label_path, batch_size=5, device="cuda"):
        pass

    def _encode_as_dataset_pos(self, sentences):
        gtokens = [tokenize(sentence) for sentence in sentences]  # greynir tokens
        tokens = [self.encode(tokenize_to_string(sentence)) for sentence in sentences]

        word_masks_w_bos = WordEndMaskDataset(
            tokens,
            self.task.is_word_initial,
            include_bos=True,
        )

        dataset = {
            "net_input": {
                "src_tokens": RightPadDataset(
                    tokens, pad_idx=self.task.source_dictionary.pad()
                ),
                "nsrc_tokens": NumelDataset(tokens),
                "word_mask_w_bos": RightPadDataset(
                    word_masks_w_bos,
                    pad_idx=0,
                ),
            },
            "ntokens": NumelDataset(tokens, reduce=True),
            "nwords": NumWordsDataset(
                tokens, is_word_initial=self.task.is_word_initial
            ),
        }
        dataset = NestedDictionaryDatasetFix2(dataset)
        return dataset

    def predict_sample_pos(self, sample, sentences, device="cuda"):
        bsz = len(sentences)
        sample = sample
        nwords = sample["nwords"]
        net_input = sample["net_input"]
        tokens = net_input["src_tokens"]
        ntokens = net_input["nsrc_tokens"]
        word_mask_w_bos = net_input["word_mask_w_bos"]
        label_shift = self.task.label_dictionary.nspecial
        ldict = self.task.label_dictionary

        label_schema = self.task.label_schema
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

        from greynirseq.nicenlp.tasks.pos_task import make_group_masks

        group_masks = make_group_masks(label_schema, self.task.label_dictionary)

        try:
            features = self.extract_features(net_input["src_tokens"])
        except ValueError:
            # TODO remove this...
            return None, ["O" for i in sentences[0]], sentences[0]

        cat_logits, attr_logits, _words_w_bos = self.model.classification_heads[
            "pos_ice"
        ](features.to(device), word_mask_w_bos=word_mask_w_bos.to(device))

        # remove bos token at start of each sequence (used for constituency parsing)
        zero = tokens.new_zeros(1)
        one = tokens.new_ones(1)
        no_bos_mask = (
            torch.cat(
                [
                    # (bsz * (nwords + 1))
                    torch.cat([zero, tokens.new_ones(seq_nwords)])
                    for seq_nwords in nwords.tolist()
                ]
            )
            .unsqueeze(-1)
            .bool()
            .to(device)
        )
        cat_logits = cat_logits.masked_select(no_bos_mask).reshape(
            -1, cat_logits.shape[-1]
        )
        attr_logits = attr_logits.masked_select(no_bos_mask).reshape(
            -1, attr_logits.shape[-1]
        )

        pred_cats = cat_logits.max(dim=-1)[1]

        map_cats = tokens.new_zeros(len(label_schema.label_categories))
        for vec_idx, lbl in enumerate(label_schema.label_categories):
            assert ldict.index(lbl) >= label_shift
            map_cats[vec_idx] = ldict.index(lbl)
        assert (map_cats.eq(0) + map_cats.ge(label_shift)).all()

        # attr_logits:  (bsz * words) x attrs
        group_preds = []
        mapped_attr_idxs = []
        binary_group_mask = tokens.new_zeros(len(label_schema.group_names))
        for group_idx, group_name in enumerate(label_schema.group_names):
            # we want fixed iteration order of group names
            mapped_group_idxs = group_name_to_mapped_vec_idxs[group_name]
            mapped_attr_idxs.append(mapped_group_idxs)
            group_logits = attr_logits[:, mapped_group_idxs]
            if len(mapped_group_idxs) == 1:
                binary_group_mask[group_idx] = 1
                # ic(group_name, group_logits)
                bool_preds = group_logits.sigmoid().ge(0.5).squeeze(-1)
                # NOTE: we flip zero to one, so that the gather operation below selects the
                # label_idx when bool_preds is True else selects 0
                group_preds.append(torch.where(bool_preds, zero, one))

                # ic(group_preds[-1].shape)
                # ic(group_preds[-1])
                # import pdb; pdb.set_trace()
            else:
                group_preds.append(group_logits.max(dim=-1)[1])
                # ic(group_preds[-1].shape)
        # import pdb; pdb.set_trace()
        group_preds = torch.stack(group_preds, dim=-1)

        mapped_attr_idxs = pad_sequence(mapped_attr_idxs, padding_value=0)

        group_pred_mask = group_masks[pred_cats]

        # ic(binary_group_mask)

        # ic(
        #     cat_logits.shape, attr_logits.shape, no_bos_mask.shape, group_preds.shape,
        #     group_mask.shape, group_pred_mask.shape, # mapped_attr_idxs.shape
        # )

        seq_idx = 0

        # pad_sequence(mapped_attr_idxs, padding_value=0)[group_preds] * group_pred_mask
        # ic(pad_sequence(mapped_attr_idxs, padding_value=0))
        # ic(group_preds[1:2])
        # ic(pad_sequence(mapped_attr_idxs, padding_value=0)[group_preds[1:2]])
        # ic(torch.gather(pad_sequence(mapped_attr_idxs, padding_value=0), 0, group_preds[1:3]))

        pred_cats_ = pred_cats
        pred_cats = map_cats[pred_cats]
        pred_attr_padded = torch.gather(
            (mapped_attr_idxs + label_shift).to(device), 0, group_preds
        )

        non_binary_pred_mask = (
            group_pred_mask.bool() * binary_group_mask.bool().bitwise_not()
        )
        binary_pred_mask = group_pred_mask.bool() * binary_group_mask.bool()
        # if (group_preds * binary_group_mask).bool().any():
        #     import pdb; pdb.set_trace()

        # import pdb; pdb.set_trace()
        # pred_attrs = pred_attr_padded.masked_select(non_binary_pred_mask).split(non_binary_pred_mask.sum(-1).tolist())
        pred_attrs = pred_attr_padded.masked_select(
            group_pred_mask.bool().to(device)
        ).split(group_pred_mask.sum(-1).tolist())
        # bin_pred_attrs = pred_attr_padded.masked_select(binary_pred_mask).split(binary_pred_mask.sum(-1).tolist())
        # bin_pred_attrs = (binary_pred_mask * group_preds * pred_attr_padded).masked_select(binary_pred_mask).split(binary_pred_mask.sum(-1).tolist())
        # bin_pred_attrs = pred_attr_padded.masked_select(binary_pred_mask * group_preds.gt(0)).split(binary_pred_mask.sum(-1).tolist())

        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        # pred_bin_attrs =

        sentence = [tokenize(sentence) for sentence in sentences][seq_idx]

        cat_idxs = []
        labels = []

        # for (tok, cat_idx, attrs_idxs, bin_idxs) in zip(sentence, pred_cats.tolist(), pred_attrs, bin_pred_attrs):
        for (tok, cat_idx, attrs_idxs) in zip(sentence, pred_cats.tolist(), pred_attrs):
            cat = ldict.symbols[cat_idx]
            cat_idxs.append(cat_idx)
            labels.append(cat)
            attrs = [
                ldict.symbols[lbl_idx]
                for lbl_idx in attrs_idxs[attrs_idxs.gt(4)].sort()[0]
            ]
            attrs = " ".join([attr for attr in attrs if "empty" not in attr])
            # bins = ""
            # if len(bin_idxs) > 0:
            #     bins = ldict.string(bin_idxs)
            # # ic(tok, cat, attrs)
            # print("    {:>20}   {:<10s}   {} {}".format(tok, cat, attrs, bins))
            # print("    {:>20}   {:<10s}   {}".format(tok, cat, attrs))

        return cat_idxs, labels, sentence

    def _encode_as_dataset(self, sentences):
        gtokens = [tokenize(sentence) for sentence in sentences]  # greynir tokens
        tokens = [self.encode(tokenize_to_string(sentence)) for sentence in sentences]

        nwords = NumWordsDataset(tokens, self.task.is_word_initial)
        word_spans = SpanDataset(tokens, self.task.is_word_initial)
        product_spans = RightPadDataset(
            ProductSpanDataset(word_spans), pad_idx=self.task.source_dictionary.pad()
        )

        dataset = NestedDictionaryDatasetFix2(
            {
                "net_input": {
                    "src_tokens": RightPadDataset(
                        tokens, pad_idx=self.task.source_dictionary.pad()
                    ),
                    "nsrc_tokens": NumelDataset(tokens, reduce=True),
                    "src_spans": RightPadDataset(
                        product_spans, pad_idx=self.task.label_dictionary.pad()
                    ),
                    "nsrc_spans": NumSpanDataset(product_spans),
                },
                "nwords": nwords,
                "word_spans": RightPadDataset(
                    word_spans, pad_idx=self.task.label_dictionary.pad()
                ),
            }
        )
        return dataset

    def predict_sample(self, sample, sentences, device="cuda"):
        bsz = len(sentences)
        nwords = sample["nwords"]
        word_spans = sample["word_spans"]
        ni = sample["net_input"]
        tokens = ni["src_tokens"]
        ntokens = ni["nsrc_tokens"]

        features = self.extract_features(ni["src_tokens"])

        logits, _extra = self.model.classification_heads["multi_span_classification"](
            features, **ni
        )
        label_shift = self.task.label_dictionary.nspecial
        scores, mask = sparse_to_chart(logits, nwords)
        mask = mask.max(dim=-1)[0]

        best_scores, presults = parse_from_chart(scores, nwords)

        seq_idx = 0
        presult = presults[seq_idx]

        bpe_tokens = [
            self.encode(tokens[seq_idx][i].unsqueeze(0))
            for i in range(1, ntokens[seq_idx] - 1)
        ]
        bpe_per_word = []
        for (start, end) in (
            (word_spans[seq_idx][: 2 * nwords[seq_idx]] - 1).reshape(-1, 2).tolist()
        ):
            bpe_per_word.append(bpe_tokens[start:end])

        seq_labels = [
            self.task.label_dictionary.symbols[l_idx + label_shift]
            for l_idx in presult.labels
        ]

        sentence = [tokenize(sentence) for sentence in sentences][seq_idx]

        tree = Node.from_labelled_spans(presult.spans, seq_labels, tokens=sentence)

        return tree, presult

    def predict(self, sentences, device="cuda"):
        device = torch.device(device)
        num_sentences = len(sentences)

        dataset = self._encode_as_dataset(sentences)
        sample = dataset.collater([dataset[i] for i in range(num_sentences)])

        return self.predict_sample(sample, sentences, device)

    def predict_pos(self, sentences, device="cuda"):
        device = torch.device(device)
        num_sentences = len(sentences)
        dataset = self._encode_as_dataset_pos(sentences)
        sample = dataset.collater([dataset[i] for i in range(num_sentences)])
        return self.predict_sample_pos(sample, sentences, device)

    def pretty_predict(self, sentence, device="cuda"):
        _ = self.predict([sentence])
        print(sentence)


@register_model_architecture("icebert_const", "icebert_const_base")
def roberta_base_architecture(args):
    base_architecture(args)
