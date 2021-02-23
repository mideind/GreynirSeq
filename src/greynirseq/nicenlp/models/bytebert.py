import itertools
import logging
from typing import Optional, Tuple, List

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from fairseq import utils
from fairseq.models import FairseqEncoder, FairseqEncoderModel, BaseFairseqModel
from fairseq.models.roberta.model import base_architecture, RobertaModel, RobertaEncoder
import fairseq.models.roberta as roberta
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

from fairseq.models.roberta.model import RobertaEncoder, RobertaLMHead
from fairseq.models.fairseq_encoder import FairseqEncoder

from greynirseq.nicenlp.modules.byte_sequence_embedder import ByteSequenceEmbedder


logger = logging.getLogger(__name__)


def lengths_to_mask(lengths):
    max_len = max(lengths)
    return torch.arange(max_len).expand(len(lengths), max_len) < lengths.unsqueeze(1)


@register_model("bytebert")
class ByteBertModel(RobertaModel):
    @classmethod
    def hub_models(cls):
        return NotImplemented

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument(
            "--num-conv-layers",
            type=int,
            metavar="L",
            help="number of convolutional blocks",
        )
        parser.add_argument(
            "--num-highway-layers",
            type=int,
            metavar="L",
            help="number of highway layers in each convolutional block",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="L", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )
        parser.add_argument(
            "--pooler-activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use for pooler layer",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )
        parser.add_argument(
            "--dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
        )
        parser.add_argument(
            "--pooler-dropout",
            type=float,
            metavar="D",
            help="dropout probability in the masked_lm pooler layers",
        )
        parser.add_argument(
            "--max-positions", type=int, help="number of positional embeddings to learn"
        )
        parser.add_argument(
            "--load-checkpoint-heads",
            action="store_true",
            help="(re-)register and load heads when loading checkpoints",
        )
        # args for "Reducing Transformer Depth on Demand with Structured Dropout" (Fan et al., 2019)
        parser.add_argument(
            "--encoder-layerdrop",
            type=float,
            metavar="D",
            default=0,
            help="LayerDrop probability for encoder",
        )
        parser.add_argument(
            "--encoder-layers-to-keep",
            default=None,
            help="which layers to *keep* when pruning as a comma-separated list",
        )
        # args for Training with Quantization Noise for Extreme Model Compression ({Fan*, Stock*} et al., 2020)
        parser.add_argument(
            "--quant-noise-pq",
            type=float,
            metavar="D",
            default=0,
            help="iterative PQ quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-pq-block-size",
            type=int,
            metavar="D",
            default=8,
            help="block size of quantization noise at training time",
        )
        parser.add_argument(
            "--quant-noise-scalar",
            type=float,
            metavar="D",
            default=0,
            help="scalar quantization noise and scalar quantization at training time",
        )
        parser.add_argument(
            "--untie-weights-roberta",
            action="store_true",
            help="Untie weights between embeddings and classifiers in RoBERTa",
        )
        parser.add_argument(
            "--spectral-norm-classification-head",
            action="store_true",
            default=False,
            help="Apply spectral normalization on the classification head",
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        bytebert_small_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = ByteBertEncoder(args, task.source_dictionary)
        return cls(args, encoder)

    def forward(self, *args, **kwargs) -> Tuple[Tensor, Tensor]:
        x, extra = self.encoder(*args, **kwargs)

        if "classification_head_name" in kwargs is not None:
            x = self.classification_heads[kwargs["classification_head_name"]](x)
        return x, extra


class ByteBertEncoder(RobertaEncoder):
    """ByteBERT Encoder. This class encapsulates the transformer-module implementation"""

    def __init__(self, args, dictionary):
        super(RobertaEncoder, self).__init__(dictionary)  # dont use Roberta constructor
        self.args = args

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))

        self.sentence_encoder = ConvGLUSentenceEncoder(
            padding_idx=dictionary.pad(),
            vocab_size=len(dictionary),
            num_encoder_layers=args.encoder_layers,
            embedding_dim=args.encoder_embed_dim,
            ffn_embedding_dim=args.encoder_ffn_embed_dim,
            num_attention_heads=args.encoder_attention_heads,
            dropout=args.dropout,
            attention_dropout=args.attention_dropout,
            activation_dropout=args.activation_dropout,
            layerdrop=args.encoder_layerdrop,
            max_seq_len=args.max_positions,
            num_segments=0,
            encoder_normalize_before=True,
            apply_bert_init=True,
            activation_fn=args.activation_fn,
            q_noise=args.quant_noise_pq,
            qn_block_size=args.quant_noise_pq_block_size,
            num_conv_layers=args.num_conv_layers,
            num_highway_layers=args.num_highway_layers,
        )

        self.lm_head = RobertaLMHead(
            embed_dim=args.encoder_embed_dim,
            output_dim=len(dictionary),
            activation_fn=args.activation_fn,
        )

    def forward(
        self,
        src_tokens: Tensor,
        src_lengths: Tensor,
        nbpe: Optional[Tensor] = None,
        segment_labels: Tensor = None,
        last_state_only: bool = False,
        positions: Optional[Tensor] = None,
        word_mask: Optional[Tensor] = None,
        bpe_mask: Optional[Tensor] = None,
        pool_lengths: Optional[Tensor] = None,
        features_only=False,
        return_all_hiddens=None,
        masked_tokens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        x, extra = self.extract_features(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            nbpe=nbpe,
            segment_labels=segment_labels,
            last_state_only=last_state_only,
            positions=positions,
            word_mask=word_mask,
            bpe_mask=bpe_mask,
            pool_lengths=pool_lengths,
        )
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens)
        return x, extra

    def extract_features(
        self,
        src_tokens: Tensor,
        src_lengths: Tensor,
        nbpe: Optional[Tensor] = None,
        segment_labels: Tensor = None,
        last_state_only: bool = False,
        positions: Optional[Tensor] = None,
        word_mask: Optional[Tensor] = None,
        bpe_mask: Optional[Tensor] = None,
        pool_lengths: Optional[Tensor] = None,
        return_all_hiddens=None,
    ):

        inner_states, _ = self.sentence_encoder(
            src_tokens=src_tokens,
            src_lengths=src_lengths,
            nbpe=nbpe,
            segment_labels=segment_labels,
            last_state_only=last_state_only,
            positions=positions,
            word_mask=word_mask,
            bpe_mask=bpe_mask,
            pool_lengths=pool_lengths,
        )
        features = inner_states[-1].transpose(0, 1)  # T x B x C -> B x T x C
        return features, {"inner_states": inner_states if return_all_hiddens else None}


# based on fairseq.modules.transformer_sentence_encoder
class ConvGLUSentenceEncoder(TransformerSentenceEncoder):
    """Convolutional version of TransformerSentenceEncoder module from fairseq"""

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
        num_highway_layers: int = 2,
    ) -> None:
        # super().__init__()
        super().__init__(
            padding_idx=padding_idx,
            vocab_size=vocab_size,
            num_encoder_layers=num_encoder_layers,
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            layerdrop=layerdrop,
            max_seq_len=max_seq_len,
            num_segments=num_segments,
            encoder_normalize_before=encoder_normalize_before,
            apply_bert_init=apply_bert_init,
            activation_fn=activation_fn,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            use_position_embeddings=use_position_embeddings,
            offset_positions_by_padding=offset_positions_by_padding,
            learned_pos_embedding=learned_pos_embedding,
            embed_scale=embed_scale,
            freeze_embeddings=freeze_embeddings,
            n_trans_layers_to_freeze=n_trans_layers_to_freeze,
            export=export,
            traceable=traceable,
        )
        # we dont really need to remove embed_tokens from TransformerSentenceEncoder since
        # we expect (vocab_size x embedding_dim) to be negligibly small
        self.character_embed_dim = character_embed_dim

        self.byte_seq_embedder = ByteSequenceEmbedder(
            self.character_embed_dim,
            self.embedding_dim,
            num_layers=num_conv_layers,
            num_highway=num_highway_layers,
            dropout=dropout,
        )
        self.embed_word_mask = nn.Embedding(
            2, self.embedding_dim, padding_idx=0
        )  # one vector for mask, one for padding
        # TODO: freeze and quant arguments are ignored for above components

    def prepare_for_tpu_(self, **kwargs):
        return NotImplemented

    def forward(
        self,
        src_tokens: Tensor,
        src_lengths: Tensor,
        nbpe: Optional[Tensor] = None,
        segment_labels: Tensor = None,
        last_state_only: bool = False,
        positions: Optional[Tensor] = None,
        word_mask: Optional[Tensor] = None,
        bpe_mask: Optional[Tensor] = None,
        pool_lengths: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        # compute padding mask. This is needed for multi-head attention
        padding_mask: Optional[Tensor] = pool_lengths.eq(0)
        if (
            not self.traceable
            and not self.tpu
            and padding_mask is not None
            and not padding_mask.any()
        ):
            padding_mask = None

        assert bpe_mask is not None or word_mask is not None

        x = self.byte_seq_embedder(
            src_tokens, bpe_mask=bpe_mask, word_mask=word_mask, pool_lengths=pool_lengths,
        )

        if self.embed_scale is not None:
            x *= self.embed_scale

        post_max_pool_mask = src_tokens.new_zeros(pool_lengths.shape).fill_(self.padding_idx)
        post_max_pool_mask[pool_lengths.ne(0)] = self.padding_idx + 1

        if self.embed_positions is not None:
            # x += self.embed_positions(tokens, positions=positions)
            x += self.embed_positions(post_max_pool_mask, positions=positions)

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


# # hparams from https://arxiv.org/pdf/1908.08962.pdf (Well read students learn better: [...])
# @register_model_architecture("bytebert", "bytebert_medium")
# def bytebert_medium_architecture(args):
#     args.encoder_layers = getattr(args, "character_embed_dim", 32)
#     args.encoder_layers = getattr(args, "encoder_layers", 8)
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
#     base_architecture(args)


@register_model_architecture("bytebert", "bytebert_small")
def bytebert_small_architecture(args):
    args.num_conv_layers = getattr(args, "num_conv_layers", 2)
    args.num_highway_layers = getattr(args, "num_highway_layers", 2)

    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_layers = getattr(args, "encoder_layers", 4)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    roberta.model.base_architecture(args)


# @register_model_architecture("bytebert", "bytebert_mini")
# def roberta_base_architecture(args):
#     base_architecture(args)
#     args.encoder_layers = getattr(args, "character_embed_dim", 32)
#     args.encoder_layers = getattr(args, "encoder_layers", 4)
#     args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 256)
#     args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 1024)
#     args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 4)
#     base_architecture(args)


@register_model_architecture("bytebert", "bytebert_tiny")
def bytebert_tiny_architecture(args):
    args.character_embed_dim = getattr(args, "character_embed_dim", 32)
    args.encoder_layers = getattr(args, "encoder_layers", 2)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 128)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 512)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 2)
    bytebert_small_architecture(args)


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
