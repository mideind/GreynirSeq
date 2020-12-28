from typing import Optional

from torch import Tensor
import torch.nn as nn
from fairseq.modules.character_token_embedder import Highway
from fairseq.modules import LayerNorm

from greynirseq.nicenlp.modules.separable_conv import (
    SeparableConv1dBlock,
    SeparableConv2dBlock,
)


class ConvHighwayBlock(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        middle_dim: Optional[int] = None,
        num_highway: int = 2,
        dropout=0.1,
        residual_conv: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.middle_dim = middle_dim or self.output_dim
        self.residual_conv = residual_conv
        self.dropout = dropout
        half = self.output_dim // 2
        self.conv = SeparableConv1dBlock(
            input_dim,
            self.output_dim,
            [(half, 3), (half, 7), (half, 15), (half, 31)],
            pointwise_channels=self.middle_dim,
            dropout=self.dropout,
        )
        self.layer_norm1 = LayerNorm(self.output_dim)
        self.glu = Highway(self.output_dim, num_layers=num_highway)
        self.layer_norm2 = LayerNorm(self.output_dim)

    def forward(self, x: Tensor):
        # (Batch x Time x Features)
        residual = x
        x = self.conv(x)
        x = residual + x if self.residual_conv else x
        x = self.layer_norm1(x)
        residual = x
        x = self.glu(x)
        x = self.layer_norm2(residual + x)
        return x

    def reset_parameters(self):
        # from fairseq.modules.character_token_embedder
        nn.init.xavier_normal_(self.char_embeddings.weight)
        nn.init.xavier_normal_(self.symbol_embeddings)
        nn.init.xavier_uniform_(self.projection.weight)

        nn.init.constant_(
            self.char_embeddings.weight[self.char_embeddings.padding_idx], 0.0
        )
        nn.init.constant_(self.projection.bias, 0.0)
