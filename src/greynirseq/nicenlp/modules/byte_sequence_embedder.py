from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, FloatTensor, LongTensor
from torch.nn.utils.rnn import pad_sequence

from fairseq.data import Dictionary, BaseWrapperDataset, FairseqDataset
from greynirseq.nicenlp.modules.conv_highway import ConvHighwayBlock


class ByteDataset(BaseWrapperDataset):
    def __init__(self, dataset: FairseqDataset):
        self.dataset = dataset

    def __getitem__(self, index):
        item = self.dataset[index]


class ByteSequenceEmbedder(nn.Module):
    """Embed a sequence of bytes (batched), with byte masking and word masking"""

    def __init__(
        self,
        byte_embed_dim: int,
        word_embed_dim: int,
        num_layers: int = 2,
        num_highway: int = 2,
        dropout: int = 0.1,
    ):
        super().__init__()
        self.byte_embed_dim = byte_embed_dim
        self.word_embed_dim = word_embed_dim
        self.pad_idx = 0
        # 256 bytes, bos, eos, pad, byte_mask, (and unsed 4 extra)
        self.bos_idx, self.eos_idx, self.byte_mask_idx = 1, 2, 3
        self.token_embeddings = nn.Embedding(
            256 + 4 + 4, self.byte_embed_dim, padding_idx=self.pad_idx
        )
        self.word_bos_idx, self.word_eos_idx, self.word_mask_idx, self.bpe_mask_idx = 1, 2, 3, 4
        # pad, bos, eos, word_mask, bpe_mask
        self.symbol_embeddings = nn.Embedding(5, self.word_embed_dim, padding_idx=self.pad_idx)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            ConvHighwayBlock(
                self.byte_embed_dim if layer_idx == 0 else self.word_embed_dim,
                self.word_embed_dim,
                num_highway=num_highway,
                dropout=dropout,
                residual_conv=layer_idx > 0,
            ) for layer_idx in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(self.word_embed_dim, self.word_embed_dim)

    def embed_bpe_markers(self, bpe_mask):
        bpe_markers = bpe_mask.long()
        bpe_markers[bpe_mask] = self.bpe_mask_idx
        bpe_markers = self.token_embeddings(bpe_markers)
        return bpe_markers

    def embed_word_markers(self, word_mask):
        word_markers = word_mask.long()
        word_markers[word_mask] = self.word_mask_idx
        word_markers = self.token_embeddings(word_markers)
        return word_markers

    def forward(self, byte_tokens: LongTensor, nbpe, bpe_lens, bpe_mask=None, word_mask=None):
        # (Batch x Time)
        x = self.token_embeddings(byte_tokens)  # BxTxC
        x += self.embed_bpe_markers(bpe_mask)
        x += self.embed_word_markers(word_mask)
        # (Batch x Time x Features)
        for layer_idx, conv_highway_layer in enumerate(self.layers):
            x = conv_highway_layer(x)
            if layer_idx == 0:
                continue
            x = self.dropout(x)
        # (Batch x Time x Features)
        contracted = []
        for seq_idx, seq in enumerate(x.split(1, dim=0)):
            word_features = [
                bpes_in_word.max(dim=1).values
                for bpes_in_word in seq.split(bpe_lens[seq_idx][:nbpe[seq_idx]].tolist(), dim=1)
            ]
            contracted.append(torch.stack(word_features, dim=1))
        x = torch.cat(contracted, dim=0)
        x = self.projection(x)
        return x

    def reset_parameters(self):
        # from fairseq.modules.character_token_embedder
        nn.init.xavier_normal_(self.token_embeddings.weight)
        nn.init.xavier_normal_(self.symbol_embeddings)
        nn.init.xavier_uniform_(self.projection.weight)  # this is inside conv block

        nn.init.constant_(
            self.token_embeddings.weight[self.token_embeddings.padding_idx], 0.0
        )
        nn.init.constant_(self.projection.bias, 0.0)
