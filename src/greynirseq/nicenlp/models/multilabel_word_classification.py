import torch
from torch import nn

from fairseq import utils
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

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.activation_fn(x)

        cat_logits = self.cat_proj(x)
        cat_probits = torch.softmax(cat_logits, dim=-1)
        attr_logits = self.out_proj(torch.cat((cat_probits, x), -1))

        return cat_logits, attr_logits
