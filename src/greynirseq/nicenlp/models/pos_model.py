import itertools
import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from fairseq import utils
from fairseq.models.roberta.model import base_architecture, RobertaModel
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models.roberta.model import RobertaEncoder
from fairseq.models import register_model, register_model_architecture

from greynirseq.nicenlp.utils.label_schema.label_schema import (
    make_vec_idx_to_dict_idx,
    make_bos_mask,
    make_mapped_group_masks,
    make_group_name_to_mapped_group_idxs,
)

from greynirseq.nicenlp.utils.constituency import token_utils
from greynirseq.nicenlp.models.multilabel_word_classification import MultiLabelTokenClassificationHead


@register_model("icebert_pos")
class IceBERTPOS(RobertaModel):

    def __init__(self, args, encoder, num_cats, num_labels):
        super().__init__(args, encoder)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        sentence_encoder = self.decoder.sentence_encoder
        if args.freeze_embeddings:
            freeze_module_params(sentence_encoder.embed_tokens)
            freeze_module_params(sentence_encoder.segment_embeddings)
            freeze_module_params(sentence_encoder.embed_positions)
            freeze_module_params(sentence_encoder.emb_layer_norm)

        for layer in range(args.n_trans_layers_to_freeze):
            freeze_module_params(sentence_encoder.layers[layer])

        self.pos_head = MultiLabelTokenClassificationHead(
            in_features=sentence_encoder.embedding_dim,
            out_features=num_labels,
            num_cats=num_cats,
            pooler_dropout=self.args.pooler_dropout,
        )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(IceBERTPOS, IceBERTPOS).add_args(parser)
        # fmt: off
        parser.add_argument('--freeze-embeddings', default=False,
                            help='Freeze transformer embeddings during fine-tuning')
        parser.add_argument('--n-trans-layers-to-freeze', default=0, type=int,
                            help='Number of transformer layers to freeze during fine-tuning')
        parser.add_argument('--tag-layer', default=12, type=int,
                            help='Which layer-activations to use to compute POS tags (note: 1-indexed)')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoder(args, task.source_dictionary)
        return cls(args, encoder, task.num_cats, task.num_labels)

    def forward(

        self, src_tokens, features_only=False, return_all_hiddens=False, **kwargs
    ):
        _x, _extra = self.decoder(
            src_tokens, features_only, return_all_hiddens=True, **kwargs
        )
        # _x = _x.transpose(0, 1)

        x_tag = _extra["inner_states"][self.args.tag_layer - 1]
        x_tag = x_tag.transpose(0, 1)
        x = x_tag

        _, _, nfeatures = x.shape
        mask = kwargs["word_mask_w_bos"]
        words_w_bos = x.masked_select(mask.unsqueeze(-1).bool()).reshape(-1, nfeatures)

        nwords_w_bos = kwargs["word_mask_w_bos"].sum(-1)

        words_w_bos_padded = pad_sequence(
            words_w_bos.softmax(-1).split((nwords_w_bos).tolist()),
            padding_value=0,
            batch_first=True,
        )

        assert "pos_head" in self.classification_heads, "task must be defined as pos_task"
        pos_features = self.pos_head(
            words_w_bos_padded, **kwargs
        )

        return pos_features

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
        return IceBERTPOSHubInterface(x["args"], x["task"], x["models"][0])

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + '.' if name != '' else ''

        for key, value in self.pos_head.state_dict().items():
            path = prefix + "pos_head." + key
            logger.info('Initializing pos_head.' + key)
            if path not in state_dict:
                state_dict[path] = value


class IceBERTPOSHubInterface(RobertaHubInterface):

    def predict_sample(self, sample):
        net_input = sample["net_input"]
        tokens = net_input["src_tokens"]

        word_mask_w_bos = sample["net_input"]["word_mask_w_bos"]
        nwords_w_bos = sample["net_input"]["word_mask_w_bos"].sum(-1)

        x, _extra = self.model.decoder(
            tokens, features_only=True, return_all_hiddens=False
        )

        term_cat_logits, term_attr_logits, _words_w_bos = self.model.classification_heads[
            "pos_tagger"
        ](
            x, word_mask_w_bos=word_mask_w_bos
        )
        term_cats, term_attrs = _term_predictions_from_logits(
            term_cat_logits,
            term_attr_logits,
            nwords_w_bos,
            self.model.task.term_dictionary,
            self.model.task.term_schema,
        )

        all_cats, all_attrs = zip(
            *[
                _clean_cats_attrs(
                    self.task.term_dictionary,
                    self.task.term_schema,
                    seq_cats,
                    seq_attrs,
                )
                for seq_cats, seq_attrs in zip(term_cats, term_attrs)
            ]
        )
        return all_cats, all_attrs

    def predict(self, sentences, device="cuda"):
        device = torch.device(device)
        num_sentences = len(sentences)

        dataset = self.prepare_sentences(sentences)
        sample = dataset.collater([dataset[i] for i in range(num_sentences)])

        return self.predict_sample(sample)


def _term_predictions_from_logits(cat_logits, attr_logits, nwords_w_bos, ldict, schema):
    no_bos_mask = make_bos_mask(nwords_w_bos).bool().bitwise_not().unsqueeze(-1)
    no_bos_mask = no_bos_mask.to(cat_logits.device)
    # logits: (bsz * nwords) x labels
    _, num_cats = cat_logits.shape
    _, num_attrs = attr_logits.shape
    assert num_attrs == len(schema.labels)
    assert num_cats == len(schema.label_categories)

    cat_logits = cat_logits.masked_select(no_bos_mask).reshape(-1, num_cats)
    attr_logits = attr_logits.masked_select(no_bos_mask).reshape(-1, num_attrs)

    cat_vec_idx_to_dict_idx = make_vec_idx_to_dict_idx(ldict, schema.label_categories)
    pred_cats = cat_vec_idx_to_dict_idx[cat_logits.max(dim=-1)[1]]

    group_name_to_mapped_group_idxs = make_group_name_to_mapped_group_idxs(
        ldict, schema.group_name_to_labels
    )

    group_mask = make_mapped_group_masks(schema, ldict)[pred_cats]

    pred_attrs = []
    for group_idx, group_name in enumerate(schema.group_names):
        mapped_group_idxs = group_name_to_mapped_group_idxs[group_name]
        # logits: (bsz * nwords) x labels
        group_logits = attr_logits[:, mapped_group_idxs]
        if len(mapped_group_idxs) == 1:
            group_preds = group_logits.sigmoid().ge(0.5).long()
        else:
            group_preds = group_logits.max(dim=-1)[1]
        mapped_group_preds = (
            mapped_group_idxs[group_preds] + ldict.nspecial
        ) * group_mask[:, group_idx]
        pred_attrs.append(mapped_group_preds)
    pred_attrs = torch.stack(pred_attrs).t()
    nwords_list = (nwords_w_bos - 1).tolist()
    return pred_cats.split(nwords_list), pred_attrs.split(nwords_list)


def _clean_cats_attrs(ldict, schema, pred_cats, pred_attrs):
    cats = ldict.string(pred_cats).split(" ")
    attrs = []
    for (_cat_idx, attr_idxs) in zip(pred_cats.tolist(), pred_attrs.split(1, dim=0)):
        seq_attrs = [
            lbl
            for lbl in ldict.string((attr_idxs.squeeze())).split(
                " "
            )  # if "empty" not in lbl
        ]
        if not any(it for it in seq_attrs):
            seq_attrs = []
        attrs.append(seq_attrs)
    return cats, attrs


@register_model_architecture("icebert_pos", "icebert_base_pos")
def roberta_base_architecture(args):
    base_architecture(args)
