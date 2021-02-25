# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import itertools
import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from fairseq import utils
from fairseq.models.roberta.model import base_architecture, roberta_base_architecture, roberta_large_architecture, RobertaModel
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models.roberta.model import RobertaEncoder
from fairseq.models import register_model, register_model_architecture
from fairseq.modules import LayerNorm

from greynirseq.nicenlp.utils.label_schema.label_schema import make_vec_idx_to_dict_idx
from greynirseq.nicenlp.data.encoding import get_word_beginnings
from greynirseq.nicenlp.utils.constituency import token_utils


logger = logging.getLogger(__name__)


class MultiClassTokenClassificationHead(nn.Module):
    """Head for word-level classification tasks."""

    def __init__(self, in_features, out_features, pooler_dropout):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features  # num_labels

        self.dense = nn.Linear(self.in_features, self.in_features)
        self.activation_fn = utils.get_activation_fn("relu")
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.layer_norm = LayerNorm(self.in_features)
        self.out_proj = nn.Linear(self.in_features, self.out_features)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.activation_fn(x)

        attr_logits = self.out_proj(x)
        return attr_logits


@register_model("multiclass_roberta")
class MultiClassRobertaModel(RobertaModel):
    def __init__(self, args, encoder, task):
        super().__init__(args, encoder)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        sentence_encoder = self.encoder.sentence_encoder
        if args.freeze_embeddings:
            freeze_module_params(sentence_encoder.embed_tokens)
            freeze_module_params(sentence_encoder.segment_embeddings)
            freeze_module_params(sentence_encoder.embed_positions)
            freeze_module_params(sentence_encoder.emb_layer_norm)

        for layer in range(args.n_trans_layers_to_freeze):
            freeze_module_params(sentence_encoder.layers[layer])

        self.task = task
        self.task_head = MultiClassTokenClassificationHead(
            in_features=sentence_encoder.embedding_dim,
            out_features=self.task.num_labels,
            pooler_dropout=self.args.pooler_dropout,
        )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        RobertaModel.add_args(parser)
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

    def forward(
        self, src_tokens, features_only=False, return_all_hiddens=False, **kwargs
    ):
        x, _extra = self.encoder(
            src_tokens, features_only, return_all_hiddens=True, **kwargs
        ) 

        _, _, inner_dim = x.shape
        word_mask = kwargs["word_mask"]

        # use first bpe token of word as representation
        x = x[:,1:-1]
        starts = word_mask[:,1:-1]  # remove bos, eos
        ends = starts.roll(-1,dims=[-1]).nonzero()[:,-1] + 1
        starts = starts.nonzero().tolist()
        mean_words = []
        for (seq_idx, token_idx), end in zip(starts, ends):
            mean_words.append(x[seq_idx, token_idx:end].mean(dim=0))
        mean_words = torch.stack(mean_words)
        words = mean_words

        nwords = word_mask.sum(-1)
        attr_logits = self.task_head(words)

        # (Batch * Time) x Depth -> Batch x Time x Depth
        attr_logits = pad_sequence(
            attr_logits.split((nwords).tolist()),
            padding_value=0,
            batch_first=True,
        )
        return attr_logits, _extra

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
        return MultiClassRobertaHubInterface(x["args"], x["task"], x["models"][0])

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + "." if name != "" else ""

        for key, value in self.task_head.state_dict().items():
            path = prefix + "task_head." + key
            logger.info("Initializing pos_head." + key)
            if path not in state_dict:
                state_dict[path] = value


class MultiClassRobertaHubInterface(RobertaHubInterface):
    def encode(self, sentence):
        if sentence[0] != " ":
            sentence = " " + sentence
        return super().encode(sentence)

    def decode(self, tokens, no_cut_space=False):
        if no_cut_space:
            return super().decode(tokens)
        return super().decode(tokens)[1:]

    def predict_labels(self, sentence):
      
        word_start_dict = get_word_beginnings(self.args, self.task.dictionary)
        tokens = self.encode(sentence)
        word_mask = torch.tensor([word_start_dict[t.item()] for t in tokens])
        word_mask[0] = 0
        word_mask[-1] = 0
        word_mask = word_mask.unsqueeze(0)
        tokens = tokens.unsqueeze(0)
        attr_logits, extra = self.model(tokens, features_only=True, word_mask=word_mask)
        pred_idxs = attr_logits.max(dim=-1).indices[0]    
        labels = [self.model.task.label_dictionary.symbols[i] for i in pred_idxs]
        return labels, pred_idxs



@register_model_architecture("multiclass_roberta", "multiclass_roberta_base")
def multiclass_roberta_base_architecture(args):
    roberta_base_architecture(args)


@register_model_architecture("multiclass_roberta", "multiclass_roberta_large")
def multiclass_roberta_large_architecture(args):
    roberta_large_architecture(args)

