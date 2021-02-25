# flake8: noqa

import itertools
import logging

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import BaseWrapperDataset, NestedDictionaryDataset, NumelDataset, RightPadDataset
from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models.roberta.model import RobertaEncoder, RobertaModel, base_architecture
from fairseq.modules import LayerNorm
from torch import nn
from torch.nn.utils.rnn import pad_sequence

import greynirseq.nicenlp.chart_parser as chart_parser  # pylint: disable=no-name-in-module
import greynirseq.nicenlp.utils.constituency.greynir_utils as greynir_utils
from greynirseq.nicenlp.data.datasets import (
    LabelledSpanDataset,
    NestedDictionaryDatasetFix,
    NestedDictionaryDatasetFix2,
    NumSpanDataset,
    ProductSpanDataset,
    WordEndMaskDataset,
)
from greynirseq.nicenlp.utils.constituency.greynir_utils import Node
from greynirseq.nicenlp.utils.label_schema.label_schema import make_vec_idx_to_dict_idx

logger = logging.getLogger(__name__)


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
        # logits: (bsz, nwords * nwords, nlabels-1)
        # make label_idx of 0 correspond to NULL label, fixed at label-span score of 0
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

        sentence_encoder = self.decoder.sentence_encoder

        # Not freezing embeddings degrades result according to multiple papers (e.g. Kitaev)
        freeze_module_params(sentence_encoder.embed_tokens)
        freeze_module_params(sentence_encoder.segment_embeddings)
        freeze_module_params(sentence_encoder.embed_positions)
        freeze_module_params(sentence_encoder.emb_layer_norm)

        for layer in range(args.n_trans_layers_to_freeze):
            freeze_module_params(sentence_encoder.layers[layer])

        self.task = task
        self.task_head = ChartParserHead(
            sentence_encoder.embedding_dim, self.task.num_nterm_cats, self.args.pooler_dropout,
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
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoder(args, task.source_dictionary)
        return cls(args, encoder, task)

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, **kwargs):
        x, _extra = self.decoder(src_tokens, features_only, return_all_hiddens=True, **kwargs)
        x = x.transpose(0, 1)
        _, _, nchannels = x.shape
        mask = kwargs["word_mask_w_bos"]
        words_w_bos = x.masked_select(mask.unsqueeze(-1).bool()).reshape(-1, nchannels)

        nwords_w_bos = kwargs["word_mask_w_bos"].sum(-1)

        words_w_bos_padded = pad_sequence(
            words_w_bos.softmax(-1).split((nwords_w_bos).tolist()), padding_value=0, batch_first=True,
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
        cls, model_name_or_path, checkpoint_file="model.pt", data_name_or_path=".", bpe="gpt2", **kwargs,
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

        word_mask_w_bos = sample["net_input"]["word_mask_w_bos"]

        # get roberta features
        x, _extra = self.model.decoder(tokens, features_only=True, return_all_hiddens=False)

        # use first bpe token of each "word" as contextual word vectors
        words_w_bos = x.masked_select(word_mask_w_bos.unsqueeze(-1).bool()).reshape(-1, self.in_features)

        parser_input_features = words_w_bos

        span_features = self.model.classification_heads["chart_parser_head"](parser_input_features)
        _tree_scores, lspans, _mask, _lmask = chart_parser.parse_many(span_features.cpu().detach(), nwords.cpu())

        cat_vec_idx_to_dict_idx = make_vec_idx_to_dict_idx(self.task.nterm_dictionary, self.task.nterm_schema.labels)

        span_labels = [
            self.task.nterm_dictionary.string(cat_vec_idx_to_dict_idx[seq_lspans[:, 2].long()]).split(" ")
            for seq_lspans in lspans
        ]

        return lspans, span_labels

    def predict(self, sentences, device="cuda"):
        device = torch.device(device)
        num_sentences = len(sentences)

        dataset = self.prepare_sentences(sentences)
        sample = dataset.collater([dataset[i] for i in range(num_sentences)])

        return self.predict_sample(sample)


@register_model_architecture("icebert_simple_parser", "icebert_base_simple_parser")
def roberta_base_architecture(args):
    base_architecture(args)
