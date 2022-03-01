# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import logging
from typing import List

import torch
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models.roberta.model import RobertaModel, base_architecture
from fairseq.modules import LayerNorm
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from greynirseq.nicenlp.utils.constituency import token_utils
from greynirseq.nicenlp.utils.constituency.greynir_utils import Node
from greynirseq.nicenlp.utils.label_schema.label_schema import make_vec_idx_to_dict_idx

logger = logging.getLogger(__name__)

try:
    import greynirseq.nicenlp.utils.constituency.chart_parser as chart_parser  # pylint: disable=no-name-in-module
except Exception as e:
    logger.warn(f"Failed to import chart_parser: {e}. Parsing will not work.")


class ChartParserHead(nn.Module):
    def __init__(self, in_features, out_features, dropout):
        super().__init__()
        self.in_features = in_features
        self.inner_dim = in_features
        self.out_features = out_features

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = LayerNorm(self.inner_dim)
        self.activation_fn = utils.get_activation_fn("relu")

        self.dense = nn.Linear(self.in_features, self.inner_dim)
        self.dense2 = nn.Linear(self.inner_dim, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, self.out_features - 1)

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.dropout(x)
        x = self.layer_norm(x)
        x = self.activation_fn(x)

        # This constructs all spans via broadcasting
        # (B x T1 x T2 x C) <- (B x T1 x C) - (B x T2 x C)
        # bijf <- bif - bjf
        x = x.unsqueeze(1) - x.unsqueeze(2)

        x = self.dense2(x)
        x = self.activation_fn(x)

        logits = self.out_proj(x)

        bsz, nwords, _, _ = logits.shape
        # logits: (bsz, nwords, nwords, nlabels-1)
        # make label_vec_idx of 0 correspond to NULL label, which we fix to label-span score of 0
        logits = torch.cat([logits.new_zeros(bsz, nwords, nwords, 1), logits], -1)

        return logits


@register_model("icebert_simple_parser")
class SimpleParserModel(RobertaModel):
    """Simple chart parser model based on Roberta. Simple in the sense that it does not
    add additional self-attentive layers or feature-engineering from Kitaev and Klein 2018."""

    def __init__(self, args, encoder, task):
        super().__init__(args, encoder)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        sentence_encoder = self.encoder.sentence_encoder

        # Not freezing embeddings degrades result according to multiple papers (e.g. Kitaev)
        freeze_module_params(sentence_encoder.embed_tokens)
        freeze_module_params(sentence_encoder.embed_positions)
        freeze_module_params(sentence_encoder.layernorm_embedding)

        for layer in range(args.n_trans_layers_to_freeze):
            freeze_module_params(sentence_encoder.layers[layer])

        _num_embeddings, embed_dim = sentence_encoder.embed_tokens.weight.shape
        self.task = task
        self.task_head = ChartParserHead(
            embed_dim,
            self.task.num_nterm_cats,
            self.args.pooler_dropout,
        )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(SimpleParserModel, SimpleParserModel).add_args(parser)
        # fmt: off
        parser.add_argument('--freeze-embeddings', default=False,
                            help='Freeze transformer embeddings during fine-tuning')
        parser.add_argument('--n-trans-layers-to-freeze', default=0, type=int,
                            help='Number of transformer layers to freeze during fine-tuning')
        parser.add_argument('--parser-layers', default=0, type=int,
                            help='Number of transformer layers to stack on-top of original encoder for parsing')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaModel.build_model(args, task).encoder

        return cls(args, encoder, task)

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, **kwargs):
        x, _extra = self.encoder(src_tokens, features_only, return_all_hiddens=True, **kwargs)
        _, _, nchannels = x.shape
        mask = kwargs["word_mask_w_bos"]
        words_w_bos = x.masked_select(mask.unsqueeze(-1).bool()).reshape(-1, nchannels)
        nwords_w_bos = kwargs["word_mask_w_bos"].sum(-1)
        words_w_bos_padded = pad_sequence(
            words_w_bos.split((nwords_w_bos).tolist()),
            padding_value=0,
            batch_first=True,
        )
        span_features = self.task_head(words_w_bos_padded)
        return span_features

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + "." if name != "" else ""

        for key, value in self.task_head.state_dict().items():
            path = prefix + "task_head." + key
            logger.info("Initializing task_head." + key)
            if path not in state_dict:
                state_dict[path] = value

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
        return SimpleParserHubInterface(x["args"], x["task"], x["models"][0])


class SimpleParserHubInterface(RobertaHubInterface):
    def predict_sample(self, sample):
        nwords = sample["nwords"]
        net_input = sample["net_input"]
        tokens = net_input["src_tokens"]

        span_features = self.model(**sample["net_input"], features_only=True)

        _tree_scores, lspans, _mask, _lmask = chart_parser.parse_many(span_features.cpu().detach(), nwords.cpu())

        cat_vec_idx_to_dict_idx = make_vec_idx_to_dict_idx(self.task.nterm_dictionary, self.task.nterm_schema.labels)
        span_labels = [
            self.task.nterm_dictionary.string(cat_vec_idx_to_dict_idx[seq_lspans[:, 2].long()]).split(" ")
            for seq_lspans in lspans
        ]
        spans = [seq_lspans[:, :2] for seq_lspans in lspans]
        src_tokens_unpadded = tokens[tokens != 1].split(sample["net_input"]["nsrc_tokens"].tolist())
        sentences_in_tokens = [self.decode(seq_tokens).strip().split(" ") for seq_tokens in src_tokens_unpadded]
        pred_trees = [
            Node.from_labelled_spans(seq_spans, seq_span_labels, sentence_tokens).debinarize()
            for seq_spans, seq_span_labels, sentence_tokens in zip(spans, span_labels, sentences_in_tokens)
        ]

        return pred_trees, (lspans, span_labels, _lmask)

    def predict(self, sentences, device="cuda"):
        device = torch.device(device)
        num_sentences = len(sentences)

        dataset = self.prepare_sentences(sentences)
        sample = dataset.collater([dataset[i] for i in range(num_sentences)])

        return self.predict_sample(sample)

    def prepare_sentences(self, sentences: List[str]):
        tokens = [
            self.encode(token_utils.tokenize_to_string(sentence, add_prefix_space=True)) for sentence in sentences
        ]
        return self.task.prepare_tokens(tokens)


@register_model_architecture("icebert_simple_parser", "icebert_base_simple_parser")
def roberta_base_architecture(args):
    base_architecture(args)
