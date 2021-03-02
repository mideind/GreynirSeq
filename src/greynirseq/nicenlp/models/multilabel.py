# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import logging

import torch
import torch.nn.functional as F
from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models.roberta.model import (
    RobertaEncoder,
    RobertaModel,
    base_architecture,
    roberta_base_architecture,
    roberta_large_architecture,
)
from fairseq.modules import LayerNorm
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from greynirseq.nicenlp.data.encoding import get_word_beginnings

logger = logging.getLogger(__name__)


class MultiLabelTokenClassificationHead(nn.Module):
    """Head for word-level classification tasks."""

    def __init__(self, in_features, out_features, num_cats, pooler_dropout):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features  # num_labels
        self.num_cats = num_cats

        self.dense = nn.Linear(self.in_features, self.in_features)
        self.activation_fn = F.relu
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.layer_norm = LayerNorm(self.in_features)
        self.cat_proj = nn.Linear(self.in_features, self.num_cats)
        self.out_proj = nn.Linear(self.in_features + self.num_cats, self.out_features)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.activation_fn(x)

        cat_logits = self.cat_proj(x)
        cat_probits = torch.softmax(cat_logits, dim=-1)
        attr_logits = self.out_proj(torch.cat((cat_probits, x), -1))

        return cat_logits, attr_logits


@register_model("multilabel_roberta")
class MultiLabelRobertaModel(RobertaModel):
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
        self.task_head = MultiLabelTokenClassificationHead(
            in_features=sentence_encoder.embedding_dim,
            out_features=self.task.num_labels,
            num_cats=self.task.num_cats,
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

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, **kwargs):
        x, _extra = self.encoder(src_tokens, features_only, return_all_hiddens=True, **kwargs)

        _, _, inner_dim = x.shape
        word_mask = kwargs["word_mask"]

        # use first bpe token of word as representation
        x = x[:, 1:-1, :]
        starts = word_mask[:, 1:-1]  # remove bos, eos
        ends = starts.roll(-1, dims=[-1]).nonzero()[:, -1] + 1
        starts = starts.nonzero().tolist()
        mean_words = []
        for (seq_idx, token_idx), end in zip(starts, ends):
            mean_words.append(x[seq_idx, token_idx:end, :].mean(dim=0))
        mean_words = torch.stack(mean_words)
        words = mean_words
        # Innermost dimension is mask for tokens at head of word.
        nwords = word_mask.sum(dim=-1)
        (cat_logits, attr_logits) = self.task_head(words)

        # (Batch * Time) x Depth -> Batch x Time x Depth
        cat_logits = pad_sequence(cat_logits.split((nwords).tolist()), padding_value=0, batch_first=True)
        attr_logits = pad_sequence(attr_logits.split((nwords).tolist()), padding_value=0, batch_first=True,)
        return (cat_logits, attr_logits), _extra

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
        return MultiLabelRobertaHubInterface(x["args"], x["task"], x["models"][0])

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + "." if name != "" else ""

        for key, value in self.task_head.state_dict().items():
            path = prefix + "task_head." + key
            logger.info("Initializing pos_head." + key)
            if path not in state_dict:
                state_dict[path] = value


class MultiLabelRobertaHubInterface(RobertaHubInterface):
    def predict_sample(self, sample):
        net_input = sample["net_input"]
        tokens = net_input["src_tokens"]

        word_mask = sample["net_input"]["word_mask"]

        (cat_logits, attr_logits), _extra = self.model(tokens, word_mask=word_mask)

        return self.task.logits_to_labels(cat_logits, attr_logits, word_mask)

    def predict(self, sentences, device="cuda"):
        device = torch.device(device)
        num_sentences = len(sentences)

        dataset = self.task.prepare_sentences(sentences)
        sample = dataset.collater([dataset[i] for i in range(num_sentences)])

        return self.predict_sample(sample)

    def encode(self, sentence):
        # We add a space to treat word encoding the same at the front and back of a sentence.
        if sentence[0] != " ":
            sentence = " " + sentence
        return super().encode(sentence)

    def decode(self, tokens, no_cut_space=False):
        return super().decode(tokens)[1:]

    def predict_labels(self, sentence):
        # assert task is set

        word_start_dict = get_word_beginnings(self.args, self.task.dictionary)

        tokens = self.encode(sentence)
        word_mask = torch.tensor([word_start_dict[t] for t in tokens.tolist()])
        word_mask[0] = 0
        word_mask[-1] = 0
        word_mask = word_mask.unsqueeze(0)
        tokens = tokens.unsqueeze(0)
        (cat_logits, attr_logits), _extra = self.model(tokens, features_only=True, word_mask=word_mask)

        labels = self.task.logits_to_labels(cat_logits, attr_logits, word_mask)

        return labels


@register_model_architecture("multilabel_roberta", "multilabel_roberta_base")
def multilabel_roberta_base_architecture(args):
    roberta_base_architecture(args)


@register_model_architecture("multilabel_roberta", "multilabel_roberta_large")
def multilabel_roberta_large_architecture(args):
    roberta_large_architecture(args)
