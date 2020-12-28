from typing import Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F
import fairseq.utils


class SeparableConv1dBlock(nn.Module):
    """Inspired by [Depthwise Separable Convolutions for Neural Machine Translation](https://arxiv.org/abs/1706.03059)"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        filters: int,
        activation_fn: str = "relu",
        pointwise_channels: Optional[int] = None,
        dropout: int = 0.1,
        pre_activation: bool=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.out_channels = sum(num_kernels for (num_kernels, width) in filters)
        self.pointwise_channels = self.output_dim if pointwise_channels is None else pointwise_channels
        self.pre_activation = pre_activation
        self.dropout = dropout

        self.conv_block = nn.ModuleList([])
        for num_kernels, width in filters:
            # width must be odd for this padding to to maintain temporal resolution
            padding = width // 2
            assert width % 2 == 1
            conv = nn.Conv1d(input_dim, num_kernels, width, padding=padding)
            self.conv_block.append(conv)

        self.conv_1x1 = nn.Linear(self.out_channels, self.pointwise_channels)
        self.dense = nn.Linear(self.pointwise_channels, self.output_dim)
        self.activation_fn = fairseq.utils.get_activation_fn(activation_fn)

    def forward(self, x):
        if self.pre_activation:
            x = self.activation_fn(x)
        # (Batch x Time x Features) -> (Batch x Features x Time)
        x = x.transpose(2, 1)
        outputs = []
        for conv in self.conv_block:
            outputs.append(conv(x))
        x = torch.cat(outputs, dim=1)
        # TODO: use activation here?
        # (Batch x Features x Time) -> (Batch x Time x Features)
        x = x.transpose(1, 2)
        x = self.conv_1x1(x)
        x = self.activation_fn(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.dense(x)
        x = self.activation_fn(x)
        return x


class SeparableConv2dBlock(nn.Module):
    """Inspired by [Depthwise Separable Convolutions for Neural Machine Translation](https://arxiv.org/abs/1706.03059)"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernels: int,
        activation_fn: str = "relu",
        pointwise_channels: Optional[int] = None,
        pre_activation: bool=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.out_channels = sum(num_kernels for (num_kernels, width) in kernels)
        self.activation_fn = fairseq.utils.get_activation_fn(activation_fn)
        self.pre_activation = pre_activation
        self.pointwise_channels = self.output_dim if pointwise_channels is None else pointwise_channels

        self.conv_block = nn.ModuleList([])
        for num_kernels, kernel_width in kernels:
            # width must be odd for this padding to to maintain temporal resolution
            padding = kernel_width // 2
            assert kernel_width % 2 == 1
            conv = nn.Conv2d(
                1, num_kernels, kernel_width, padding=padding, groups=num_kernels
            )
        self.conv_1x1 = nn.Conv2d(
            self.out_channels, self.pointwise_channels, 1, padding=0
        )
        self.dense = nn.Linear(self.pointwise_channels, self.output_dim)

    def forward(self, x):
        if self.pre_activation:
            x = self.activation_fn(x)
        # (Batch x Time x Features) -> (Batch x 1 x Features x Time)
        x = x.transpose(2, 1).unsqueeze(1)
        outputs = []
        for conv in self.conv_block:
            outputs.append(conv(x))
        x = torch.cat(outputs, dim=1)
        # TODO: use activation here?
        x = self.conv_1x1(x)
        # (Batch x C_out x Features x Time) - > (Batch x C_out x Time)
        x = x.max(dim=-1)
        # (Batch x C_out x Time) - > (Batch x Time x C_out)
        x = x.transpose(1, 2)
        x = self.dense(x)
        return x
