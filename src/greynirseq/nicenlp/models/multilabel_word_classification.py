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


class MultiLabelTokenClassificationHead(nn.Module):
    """Head for word-level classification tasks."""

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
        assert "word_mask_w_bos" in kwargs
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
