import itertools
import logging
from typing import Optional, Tuple, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from fairseq import utils
from fairseq.models.roberta.model import base_architecture, RobertaModel, RobertaEncoder
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models import register_model, register_model_architecture
from fairseq.modules import (
    FairseqDropout,
    LayerDropModuleList,
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    TransformerSentenceEncoder,
    TransformerSentenceEncoderLayer,
    TransformerSentenceEncoderLayer,
)
from fairseq.data import (
    RightPadDataset,
    BaseWrapperDataset,
    NestedDictionaryDataset,
    TokenBlockDataset,
    NumelDataset,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from fairseq.models.roberta.model import RobertaEncoder
from fairseq.models.fairseq_encoder import FairseqEncoder

from greynirseq.nicenlp.modules.byte_sequence_embedder import ByteSequenceEmbedder


logger = logging.getLogger(__name__)

def lengths_to_mask(lengths):
    max_len = max(lengths)
    return torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)


class ByteBertSentenceEncoder(TransformerSentenceEncoder):
    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        encoder_normalize_before: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
        num_conv_layers: int = 2,
        character_embed_dim: int = 32,
        num_highway: int = 2,
    ) -> None:
        super(TransformerSentenceEncoder, self).__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding
        self.traceable = traceable
        self.tpu = False  # whether we're on TPU
        self.character_embed_dim = character_embed_dim
        self.dropout = dropout

        self.embed_word_mask = nn.Embedding(
            2, self.embedding_dim, padding_idx=0
        )  # one for mask, one for padding

        self.embed_tokens = self.build_embedding(
            self.vocab_size, self.embedding_dim, self.padding_idx
        )
        self.embed_scale = embed_scale

        self.byte_seq_embedder = ByteSequenceEmbedder(
            self.character_embed_dim,
            self.embedding_dim,
            num_layers=2,
            num_highway=num_highway,
            dropout=self.dropout,
        )

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_transformer_sentence_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def build_embedding(self, vocab_size, embedding_dim, padding_idx):
        return nn.Embedding(vocab_size, embedding_dim, padding_idx)

    def prepare_for_tpu_(self, **kwargs):
        return NotImplemented

    def forward(
        self,
        tokens: Tensor,
        nbpe: Optional[Tensor] = None,
        bpe_lens: Optional[Tensor] = None,
        segment_labels: Tensor = None,
        last_state_only: bool = False,
        positions: Optional[Tensor] = None,
        word_mask: Optional[Tensor] = None,
        bpe_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # compute padding mask. This is needed for multi-head attention
        padding_mask: Optional[Tensor] = tokens.eq(self.padding_idx)
        if (
            not self.traceable
            and not self.tpu
            and padding_mask is not None
            and not padding_mask.any()
        ):
            padding_mask = None

        x = self.embed_tokens(tokens)

        if self.embed_scale is not None:
            x *= self.embed_scale

        x = self.byte_seq_embedder(tokens, nbpe, bpe_lens, bpe_mask=bpe_mask, word_mask=word_mask)
        word_positions = lengths_to_mask(nbpe).long()
        padding_mask = 1 - word_positions
        word_positions[word_positions.eq(1)] = self.padding_idx + 1
        word_positions[padding_mask] = self.padding_idx

        if self.embed_positions is not None:
            # x += self.embed_positions(tokens, positions=positions)
            x += self.embed_positions(word_positions, positions=positions)

        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)  # type: ignore

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(x, self_attn_padding_mask=padding_mask)
            if not last_state_only:
                inner_states.append(x)

        sentence_rep: Tensor = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), sentence_rep
        else:
            return inner_states, sentence_rep  # type: ignore


# @register_model("bytebert")
# class ByteBert(RobertaEncoder):
#     def __init__(self, args, dictionary):
#         super().__init__()
#         self.args = args
#         # self.sentence_encoder =


# # hparams from https://arxiv.org/pdf/1908.08962.pdf (Well read students learn better: [...])
# @register_model_architecture("bytebert", "bytebert_medium")
# def roberta_base_architecture(args):
#     base_architecture(args)
#     args.encoder_layers = getattr(args, "character_embed_dim", 32)
#     args.encoder_layers = getattr(args, "encoder_layers", 8)
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
#     base_architecture(args)


# @register_model_architecture("bytebert", "bytebert_small")
# def roberta_base_architecture(args):
#     base_architecture(args)
#     args.encoder_layers = getattr(args, "character_embed_dim", 32)
#     args.encoder_layers = getattr(args, "encoder_layers", 4)
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
#     base_architecture(args)


# @register_model_architecture("bytebert", "bytebert_mini")
# def roberta_base_architecture(args):
#     base_architecture(args)
#     args.encoder_layers = getattr(args, "character_embed_dim", 32)
#     args.encoder_layers = getattr(args, "encoder_layers", 4)
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
#     base_architecture(args)


# @register_model_architecture("bytebert", "bytebert_tiny")
# def roberta_base_architecture(args):
#     base_architecture(args)
#     args.encoder_layers = getattr(args, "character_embed_dim", 32)
#     args.encoder_layers = getattr(args, "encoder_layers", 16)
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
#     base_architecture(args)


# @register_model_architecture("bytebert", "bytebert_tinybert")
# def roberta_base_architecture(args):
#     base_architecture(args)
#     args.encoder_layers = getattr(args, "character_embed_dim", 32)
#     # hparams from tinybert paper
#     args.encoder_layers = getattr(args, "encoder_layers", 4)
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 312)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1200)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 12)
#     base_architecture(args)
