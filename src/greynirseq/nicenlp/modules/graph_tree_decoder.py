# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

from argparse import Namespace
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass import FairseqDataclass
from fairseq.models.roberta.model import roberta_base_architecture
from fairseq.modules import FairseqDropout, LayerNorm, LearnedPositionalEmbedding, PositionalEmbedding
from fairseq.modules.transformer_sentence_encoder import TransformerSentenceEncoderLayer
from omegaconf import II
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

from greynirseq.nicenlp.criterions.incremental_parser import (
    IncrementalParserCriterion,
    IncrementalParserCriterionConfig,
)
from greynirseq.nicenlp.data.datasets import collate_2d
from greynirseq.nicenlp.utils.constituency.greynir_utils import NonterminalNode, TerminalNode
from greynirseq.nicenlp.utils.constituency.incremental_parsing import NULL_LABEL, ROOT_LABEL
from greynirseq.nicenlp.utils.constituency.scratch_incremental import IncrementalParser

PADDING_VALUE_FOR_NON_INDEX = -100
NULL_SPAN = (PADDING_VALUE_FOR_NON_INDEX, PADDING_VALUE_FOR_NON_INDEX)
_roberta_base_args = Namespace()
roberta_base_architecture(_roberta_base_args)


@dataclass
class ConstituencyParserOutput:
    """Simple container for incremental outputs of constituency part of GraphTreeDecoder"""

    parent_logits: Union[List[Tensor], Tensor]
    preterm_logits: Union[List[Tensor], Tensor]
    parent_flag_logits: Union[List[Tensor], Tensor]
    preterm_flag_logits: Union[List[Tensor], Tensor]
    attention: Union[List[Tensor], Tensor]


@dataclass
class _TransformerSentenceEncoderLayerConfig:
    embedding_dim: int = _roberta_base_args.encoder_embed_dim  # 768
    ffn_embedding_dim: int = _roberta_base_args.encoder_ffn_embed_dim  # 3072
    num_attention_heads: int = _roberta_base_args.encoder_attention_heads  # 8
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    activation_fn: str = "relu"
    export: bool = False
    q_noise: float = 0.0
    qn_block_size: int = 8


@dataclass
class TreeGraphDecoderConfig(FairseqDataclass):
    transformer: Optional[_TransformerSentenceEncoderLayerConfig] = _TransformerSentenceEncoderLayerConfig()
    # embed_dim: int = field(default=_roberta_base_args.encoder_embed_dim, metadata={"help": "Embedding dimension"})
    # embed_dim: int = II("..transformer.embedding_dim")
    _embedding_dim: int = II("model.graph_decoder.transformer.embedding_dim")
    layers: int = field(default=2, metadata={"help": "Number of layers in the decoder"})
    max_positions: int = field(default=_roberta_base_args.max_source_positions)
    learned_pos: bool = field(default=_roberta_base_args.encoder_learned_pos)
    layernorm_embedding: bool = field(default=_roberta_base_args.layernorm_embedding)
    dropout: float = field(default=_roberta_base_args.dropout)
    factored_embeddings: bool = field(default=False)
    mlp_attn_is_sigmoid: bool = field(default=True, metadata={"help": ""})
    freeze_position_embeddings: bool = field(default=True)
    num_recursions: int = field(default=0)
    shared_embeddings: bool = field(default=True)
    chain_classifiers: bool = field(default=False)
    use_word_position: bool = field(default=False)
    add_attention_outputs: bool = field(
        default=True, metadata={"help": "Add the output from MLPAttention or concatenate it"}
    )


# @dataclass
# class _TransformerSentenceEncoderConfig:
#     padding_idx: int,
#     vocab_size: int,
#     num_encoder_layers: int = 6,
#     embedding_dim: int = 768,
#     ffn_embedding_dim: int = 3072,
#     num_attention_heads: int = 8,
#     dropout: float = 0.1,
#     attention_dropout: float = 0.1,
#     activation_dropout: float = 0.1,
#     layerdrop: float = 0.0,
#     max_seq_len: int = 256,
#     num_segments: int = 2,
#     use_position_embeddings: bool = True,
#     offset_positions_by_padding: bool = True,
#     encoder_normalize_before: bool = False,
#     app= 8,


class TreeGraphDecoder(nn.Module):
    def __init__(
        self,
        cfg: TreeGraphDecoderConfig,
        embed_positions: Optional[Tensor] = None,
        padding_idx: int = 1,
        num_labels: int = -1,
        root_label_index: int = -1,
        project_input_from: Optional[int] = None,
    ):
        # XXX: dropout is missing
        super().__init__()
        self.cfg = cfg
        assert num_labels > 0
        assert root_label_index > 0
        self.embed_dim = cfg._embedding_dim
        self.num_labels = num_labels
        self.padding_idx = padding_idx
        self.root_label_index = root_label_index
        assert cfg.learned_pos, "Currently unsupported"
        self.embed_positions = PositionalEmbedding(
            num_embeddings=cfg.max_positions,
            embedding_dim=project_input_from or self.embed_dim,
            padding_idx=padding_idx,
            learned=cfg.learned_pos,
        )

        # XXX: these are nonterminal labels, as well as attribute labels
        self.embed_labels = nn.Embedding(self.num_labels, self.embed_dim, self.padding_idx)

        self.input_projection = None
        if project_input_from is not None and project_input_from != self.embed_dim:
            self.input_projection = nn.Linear(project_input_from, self.embed_dim, bias=False)

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(**cfg.transformer)
                for _ in range(cfg.layers)
            ]
        )
        self.dropout_module = FairseqDropout(cfg.dropout, module_name=self.__class__.__name__)
        self.layernorm_embedding = None
        if self.cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(self.embed_dim)
        self.layer_norm = LayerNorm(self.embed_dim)

        # XXX: should the output heads be here or on the wrapping model?
        # XXX: the out-projection should be shared with the input label embedding matrix of the decoder!
        classifier_input_dim = self.embed_dim if self.cfg.add_attention_outputs else 2 * self.embed_dim
        self.classification_heads = nn.ModuleDict()
        out_projection = self.embed_labels.weight if cfg.shared_embeddings else None
        self.classification_heads["constit_preterm"] = ScaffoldHead(
            classifier_input_dim,
            self.num_labels,
            self.cfg.dropout,
            out_projection=out_projection,
            inner_dim=self.embed_dim,
        )
        self.classification_heads["constit_parent"] = ScaffoldHead(
            classifier_input_dim,
            self.num_labels,
            self.cfg.dropout,
            out_projection=out_projection,
            inner_dim=self.embed_dim,
        )
        self.classification_heads["constit_parent_flags"] = ScaffoldHead(
            classifier_input_dim,
            self.num_labels,
            self.cfg.dropout,
            out_projection=out_projection,
            inner_dim=self.embed_dim,
        )
        self.classification_heads["constit_preterm_flags"] = ScaffoldHead(
            classifier_input_dim,
            self.num_labels,
            self.cfg.dropout,
            out_projection=out_projection,
            inner_dim=self.embed_dim,
        )
        self.mlp_attention = SingleVectorMLPAttention(
            2 * self.embed_dim, self.embed_dim // 4, self.cfg.dropout, use_sigmoid=cfg.mlp_attn_is_sigmoid
        )

    def embed_spans(self, spans: Tensor, end_thresholds: Optional[Tensor] = None) -> Tensor:
        # spans: B x T x S
        span_starts = spans[:, :, 0].clone()
        span_ends = spans[:, :, 1].clone()  # clone because we mutate
        if end_thresholds is not None:
            for seq_idx in range(len(end_thresholds)):
                span_ends[seq_idx][end_thresholds[seq_idx] < span_ends[seq_idx]] = end_thresholds[seq_idx]

        # we need to provide the positions vector ourselves, but we inherited the position embedding from roberta
        # so we need to make some assumptions since LearnedPositionalEmbedding does not allow use of padding_idx
        # at the same time as providing positions
        assert isinstance(self.embed_positions, LearnedPositionalEmbedding), "Currently only learned_pos is supported"
        assert self.embed_positions.padding_idx is not None

        # this is how fairseq.utils.make_positions() behaves, which LearnedPositionalEmbedding uses
        span_starts = span_starts + self.padding_idx + 1
        span_ends = span_ends + self.padding_idx + 1
        span_embs = (
            F.embedding(
                span_starts,
                self.embed_positions.weight,
                self.embed_positions.padding_idx,
                self.embed_positions.max_norm,
                self.embed_positions.norm_type,
                self.embed_positions.scale_grad_by_freq,
                self.embed_positions.sparse,
            )
            + F.embedding(
                span_ends,
                self.embed_positions.weight,
                self.embed_positions.padding_idx,
                self.embed_positions.max_norm,
                self.embed_positions.norm_type,
                self.embed_positions.scale_grad_by_freq,
                self.embed_positions.sparse,
            )
        ) / 2
        if self.input_projection is not None:
            return self.input_projection(span_embs)
        return span_embs

    def embed_word_positions(self, nwords) -> Tensor:
        bsz, = nwords.shape
        start = self.padding_idx + 1
        positions = torch.arange(start, start + nwords.max()).to(nwords.device)
        pos_emb = F.embedding(
            positions,
            self.embed_positions.weight,
            self.embed_positions.padding_idx,
            self.embed_positions.max_norm,
            self.embed_positions.norm_type,
            self.embed_positions.scale_grad_by_freq,
            self.embed_positions.sparse,
        )
        if self.input_projection is not None:
            pos_emb = self.input_projection(pos_emb)
        return pos_emb.tile(bsz, 1, 1)

    def forward_nodes(self, x: Tensor, self_attn_padding_mask: Tensor) -> Tensor:
        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        niterations = self.cfg.num_recursions or 1
        for _recursive_iterations in range(niterations):
            for _idx, layer in enumerate(self.layers):
                x, _layer_attn = layer(x, self_attn_padding_mask=self_attn_padding_mask, self_attn_mask=None)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        return x

    def forward_step(
        self,
        encoder_out: Tensor,
        preorder_nts: Tensor,
        preorder_mask: Tensor,
        chain_mask: Tensor,
        preorder_spans: Tensor,
        nwords: Tensor,
        preorder_flags: Tensor,
        state: ConstituencyParserOutput,
        **kwargs,
    ) -> ConstituencyParserOutput:
        """
            preorder_nts: bsz x nodes
            preorder_mask: bsz x nsteps x nodes
            chain_mask: bsz x nsteps x nodes
            preorder_spans: bsz x nodes x 2
            nwords: bsz x nsteps
            preorder_flags: bsz x nodes x flags
        """
        if self.input_projection is not None:
            encoder_out = self.input_projection(encoder_out)
        assert nwords.gt(0).all(), "Empty sequences in decoder results in nans during training"
        # bsz x nsteps x nodes ->  nsteps x bsz x nodes
        bsz, _num_nodes = preorder_mask.shape
        preorder_embs = self.embed_labels(preorder_nts)
        # bsz x nodes x flags  ->  bsz x nodes x flags x features  ->  bsz x nodes x features
        preorder_embs += self.embed_labels(preorder_flags).sum(dim=-2)
        # simple container for incremental outputs
        max_span = nwords - 1  # newest word is not actually in the incremental tree yet, so subtract 1
        preorder_embs = preorder_embs + self.embed_spans(preorder_spans, end_thresholds=max_span)
        preorder_embs *= preorder_mask.unsqueeze(-1)

        root_spans = torch.stack([nwords.new_zeros(bsz), max_span], dim=1)
        # root_spans: B x 2  ->  B x 1 x 2  ->  B x 1 x C
        root_span_emb = self.embed_spans(root_spans.unsqueeze(1), end_thresholds=max_span)
        # root_emb: 0  ->  C  ->  B x 1 x C
        root_emb = self.embed_labels(root_spans.new_tensor(1).fill_(self.root_label_index)).tile((bsz, 1, 1))
        root_emb = root_emb + root_span_emb

        word_mask = lengths_to_padding_mask(nwords).logical_not()
        word_embs = encoder_out[:, : nwords.max(), :][word_mask]
        word_embs = pad_sequence(word_embs.split(word_mask.sum(-1).tolist()), batch_first=True, padding_value=0)
        if self.cfg.use_word_position:
            word_embs = word_embs + self.embed_word_positions(nwords)
        root_mask = preorder_mask.new_ones(bsz).unsqueeze(-1)
        input_mask = torch.cat([word_mask, root_mask, preorder_mask], dim=1)
        input_embs = torch.cat([word_embs, root_emb, preorder_embs], dim=1) * input_mask.unsqueeze(-1)

        # x shape: bsz x (nodes+1) x features  # we add one for root
        x = self.forward_nodes(
            input_embs, self_attn_padding_mask=input_mask.logical_not()
        )
        #  * input_mask.unsqueeze(-1)
        assert not x.isnan().any()

        # chain_nodes x features
        output_chain_mask = torch.cat([torch.zeros_like(word_mask), root_mask, chain_mask], dim=1)
        right_chain_outputs = x[output_chain_mask].split(output_chain_mask.sum(dim=-1).tolist())
        # bsz x maxchainlen x features
        right_chain_outputs = pad_sequence(right_chain_outputs, batch_first=True, padding_value=0)
        # bsz x features  ->  bsz x 1 x features
        attending_words = x[torch.arange(bsz), nwords - 1, :].unsqueeze(1)
        attn_padding_mask = lengths_to_padding_mask(output_chain_mask.sum(-1))
        attn_output_features, attn = self.mlp_attention(
            right_chain_outputs, attending_words, attn_padding_mask=attn_padding_mask
        )
        assert attending_words.shape == attn_output_features.shape

        if self.cfg.add_attention_outputs:
            clsf_features = attending_words + attn_output_features
        else:
            clsf_features = torch.cat([attending_words, attn_output_features], dim=-1)

        # pp [(v.isnan().any().item(),k) for k,v in locals().items() if hasattr(v, "isnan")]
        # bsz x 1 x features  ->  bsz x features
        state.parent_logits.append(self.classification_heads["constit_parent"](clsf_features).squeeze(1))
        state.preterm_logits.append(self.classification_heads["constit_preterm"](clsf_features).squeeze(1))
        state.parent_flag_logits.append(self.classification_heads["constit_parent_flags"](clsf_features).squeeze(1))
        state.preterm_flag_logits.append(self.classification_heads["constit_preterm_flags"](clsf_features).squeeze(1))
        state.attention.append(attn.squeeze(2))
        return state

    def forward(
        self,
        encoder_out: Tensor,
        preorder_nts: Tensor,
        preorder_mask: Tensor,
        chain_mask: Tensor,
        preorder_spans: Tensor,
        nwords_per_step: Tensor,
        preorder_flags: Tensor,
        **kwargs,
    ) -> ConstituencyParserOutput:
        """
            preorder_nts: bsz x nodes
            preorder_mask: bsz x nsteps x nodes
            chain_mask: bsz x nsteps x nodes
            preorder_spans: bsz x nodes x 2
            nwords_per_step: bsz x nsteps
            preorder_flags: bsz x nodes x flags
        """

        # simple container for incremental outputs
        state = ConstituencyParserOutput([], [], [], [], [])
        _bsz, nsteps = nwords_per_step.shape

        # for curr_step in
        for curr_step in range(nsteps):
            is_alive = nwords_per_step[:, curr_step] > 0
            self.forward_step(
                encoder_out=encoder_out[is_alive],
                preorder_nts=preorder_nts[is_alive],
                preorder_flags=preorder_flags[is_alive],
                preorder_spans=preorder_spans[is_alive],
                preorder_mask=preorder_mask[is_alive, curr_step],
                chain_mask=chain_mask[is_alive, curr_step],
                nwords=nwords_per_step[is_alive, curr_step],
                state=state,
            )

        state.parent_logits = torch.cat(state.parent_logits, dim=0)
        state.preterm_logits = torch.cat(state.preterm_logits, dim=0)
        state.parent_flag_logits = torch.cat(state.parent_flag_logits, dim=0)
        state.preterm_flag_logits = torch.cat(state.preterm_flag_logits, dim=0)
        return state


class ScaffoldHead(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim,
        dropout,
        inner_dim: Optional[Tensor] = None,
        out_projection: Optional[Tensor] = None,
        bias=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.inner_dim = input_dim if inner_dim is None else inner_dim
        self.output_dim = output_dim

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = LayerNorm(self.input_dim)
        self.activation_fn = utils.get_activation_fn("relu")

        self.dense = nn.Linear(self.input_dim, self.input_dim)
        self.dense2 = nn.Linear(self.input_dim, self.inner_dim)
        # NOTE: the linear part may be shared but not the bias is separate
        self.out_projection = nn.Linear(self.inner_dim, self.output_dim, bias=True)
        if out_projection is not None:
            self.out_projection.weight = out_projection

    def forward(self, x, **kwargs) -> Tensor:
        x = self.dropout(x)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.layer_norm(x)
        x = self.dense2(x)
        x = self.activation_fn(x)
        return self.out_projection(x)


class SingleVectorMLPAttention(nn.Module):
    """This is an implementation of the attention module described in the paper
       See for more details: https://arxiv.org/abs/2010.14568 """

    def __init__(self, input_dim, inner_dim, dropout: float, use_sigmoid: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.inner_dim = inner_dim
        self.use_sigmoid = use_sigmoid

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = LayerNorm(self.inner_dim)
        self.activation_fn = utils.get_activation_fn("relu")

        self.dense = nn.Linear(self.input_dim, self.inner_dim)
        self.dense2 = nn.Linear(self.inner_dim, self.inner_dim)
        self.out_projection = nn.Linear(self.inner_dim, 1, bias=False)

    def forward(self, right_chain_nodes, newest_word, attn_padding_mask) -> Tuple[Tensor, Tensor]:
        """The vector v[i] attends to each vector in right_chain_nodes[i, :] (in batched manner)

           newest_word:       v has shape (Batch,    1, Features)
           right_chain_nodes: x has shape (Batch, Time, Features)
        Returns a version of x where the features of a single v (respectively for each batch dimension)
        have been concatenated to all vectors in the time dimension """
        _, right_chain_len, _ = right_chain_nodes.shape
        # B x 1 x C  ->  B x RCHAIN x C
        tiled_newest_word = torch.tile(newest_word, dims=(1, right_chain_len, 1))
        # concatenate along feature dimension
        x = torch.cat([right_chain_nodes, tiled_newest_word], dim=-1)

        x = self.dropout(x)
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.activation_fn(x)
        x = self.dense2(x)
        x = self.activation_fn(x)
        # attn: B x T x 1
        attn = self.out_projection(x)
        attn[attn_padding_mask] = float("-inf")

        # add the contribution of each node in the right-chain to the output, weighted by its attention
        # (B x T x C) * (B x T x 1)
        # NOTE: this does not use a Value matrix like regular attention, maybe we should try it
        if self.use_sigmoid:
            output_features = (right_chain_nodes * attn.sigmoid()).mean(dim=1)
        else:
            output_features = (right_chain_nodes * attn.softmax(dim=1)).mean(dim=1)

        # B x C  ->  B x 1 x C
        return output_features.unsqueeze(dim=1), attn


def make_example_tree_01():
    #    ADVP
    #     |
    #     x
    #     |
    #    ekki
    ekki = NonterminalNode("ADVP", [TerminalNode("ekki", "x")])
    return ekki


def make_example_tree_02():
    #        VP
    #     ___|___
    #    ADVP   VP
    #     |     |
    #     x     x
    #     |     |
    #    ekki renna
    ekki = NonterminalNode("ADVP", [TerminalNode("ekki", "x")])
    renna = NonterminalNode("VP", [TerminalNode("renna", "x")])
    sentence = NonterminalNode("VP", [ekki, renna]).collapse_unary()
    return sentence


def make_example_tree_03():
    #            S0
    #     ________|______
    #     NP      VP   ADVP
    #     |       |     |
    #     x       x     x
    #     |       |     |
    #   bíllinn  rann  ekki
    billinn = NonterminalNode("NP", [TerminalNode("bíllinn", "x")])
    rann = NonterminalNode("VP", [TerminalNode("rann", "x")])
    ekki = NonterminalNode("ADVP", [TerminalNode("ekki", "x")])
    sentence = NonterminalNode("S0", [billinn, rann, ekki]).collapse_unary()
    return sentence


def make_example_tree_04():
    #           S0
    #     ______|_____
    #     NP         VP
    #     |       ___|___
    #     x       VP   ADVP
    #     |       |     |
    #   bíllinn   x     x
    #             |     |
    #            rann  ekki
    billinn = NonterminalNode("NP", [TerminalNode("bíllinn", "x")])
    rann = NonterminalNode("VP", [TerminalNode("rann", "x")])
    ekki = NonterminalNode("ADVP", [TerminalNode("ekki", "x")])
    rann_ekki = NonterminalNode("VP", [rann, ekki])
    sentence = NonterminalNode("S0", [billinn, rann_ekki]).collapse_unary()
    return sentence


def make_example_tree_05():
    #                S0
    #         _______|_____
    #         NP          VP
    #    _____|_____      |
    #    NP        NP     x
    #    |         |      |
    #    x         x     rann
    #    |         |
    #   bíllinn  minn
    billinn = NonterminalNode("NP", [TerminalNode("bíllinn", "x")])
    minn = NonterminalNode("ADVP", [TerminalNode("minn", "x")])
    billinn_minn = NonterminalNode("NP", [billinn, minn])
    rann = NonterminalNode("VP", [TerminalNode("rann", "x")])
    sentence = NonterminalNode("S0", [billinn_minn, rann]).collapse_unary()
    return sentence


def make_example_tree_06():
    #                S0
    #         _______|______
    #         NP           VP
    #    _____|_____    ___|___
    #    NP        NP   VP   ADVP
    #    |         |    |     |
    #    x         x    x     x
    #    |         |    |     |
    #   bíllinn  minn  rann  ekki
    billinn = NonterminalNode("NP", [TerminalNode("bíllinn", "x")])
    minn = NonterminalNode("NP", [TerminalNode("minn", "x")])
    billinn_minn = NonterminalNode("NP", [billinn, minn])
    rann = NonterminalNode("VP", [TerminalNode("rann", "x")])
    ekki = NonterminalNode("ADVP", [TerminalNode("ekki", "x")])
    rann_ekki = NonterminalNode("VP", [rann, ekki])
    sentence = NonterminalNode("S0", [billinn_minn, rann_ekki]).collapse_unary()
    return sentence


def make_example_tree_07():
    #                S0
    #         _______|__________
    #         NP                VP
    #    _____|___________    ___|___
    #    NP        NP    NP   VP   ADVP
    #    |         |     |    |     |
    #    x         x     x    x     x
    #    |         |     |    |     |
    #   bíllinn  hans  hans  rann  ekki
    billinn = NonterminalNode("NP", [TerminalNode("bíllinn", "x")])
    hans = NonterminalNode("NP", [TerminalNode("hans", "x")])
    jons = NonterminalNode("NP", [TerminalNode("jóns", "x")])
    billinn_hans_jons = NonterminalNode("NP", [billinn, hans, jons])
    rann = NonterminalNode("VP", [TerminalNode("rann", "x")])
    ekki = NonterminalNode("ADVP", [TerminalNode("ekki", "x")])
    rann_ekki = NonterminalNode("VP", [rann, ekki])
    sentence = NonterminalNode("S0", [billinn_hans_jons, rann_ekki]).collapse_unary()
    return sentence


def make_example_tree_08():
    #                S0
    #         _______|__________
    #         NP                VP
    #    _____|________       ___|___
    #    NP           NP      VP   ADVP
    #    |         ___|___    |     |
    #    x         NP    NP   x     x
    #    |         |     |    |     |
    #   bíllinn    x     x   rann  ekki
    #              |     |
    #            hans  jóns
    # 0        1     2     3     4     5  # span fences
    billinn = NonterminalNode("NP", [TerminalNode("bíllinn", "x")])
    hans = NonterminalNode("NP", [TerminalNode("hans", "x")])
    jons = NonterminalNode("NP", [TerminalNode("jóns", "x")])
    rann = NonterminalNode("VP", [TerminalNode("rann", "x")])
    ekki = NonterminalNode("ADVP", [TerminalNode("ekki", "x")])
    hans_jons = NonterminalNode("NP", [hans, jons])
    billinn_hans_jons = NonterminalNode("NP", [billinn, hans_jons])
    rann_ekki = NonterminalNode("VP", [rann, ekki])
    sentence = NonterminalNode("S0", [billinn_hans_jons, rann_ekki]).collapse_unary()
    return sentence


def make_example_tree():
    #            S0-TOP>S0>IP
    #      ___________|_______
    #     NP3-SUBJ           VP
    #      |            _____|___________
    #     NP2      VP-AUX-SEQ   ADVP    VP
    #      |            |        |      |
    #     NP1           x        x      x
    #      |            |        |      |
    #      x          hafði     ekki  runnið
    #      |
    #   bíllinn
    #
    billinn = NonterminalNode("NP1", [TerminalNode("bíllinn", "x")])
    billinn = NonterminalNode("NP2", [billinn])
    billinn = NonterminalNode("NP3-SUBJ", [billinn])
    hafdi = NonterminalNode("VP-AUX-SEQ", [TerminalNode("hafði", "x")])
    ekki = NonterminalNode("ADVP", [TerminalNode("ekki", "x")])
    runnid = NonterminalNode("VP", [TerminalNode("runnið", "x")])
    hafdi_ekki_runnid = NonterminalNode("VP", [hafdi, ekki, runnid])
    ip = NonterminalNode("IP", [billinn, hafdi_ekki_runnid])
    sentence = NonterminalNode("S0", [ip]).collapse_unary()
    sentence = NonterminalNode("S0-TOP", [sentence]).collapse_unary()
    return sentence


def test_example():
    sentence = make_example_tree()
    from greynirseq.nicenlp.utils.constituency.incremental_parsing import get_incremental_parse_actions

    actions = get_incremental_parse_actions(sentence)
    tokens = [leaf.text for leaf in sentence.leaves]
    parser = IncrementalParser(tokens)
    for action in actions:
        parser.add(action, strict=True)
    return sentence


def test_forward():
    from greynirseq.nicenlp.utils.constituency.incremental_parsing import get_incremental_parse_actions
    from pprint import pprint
    from fairseq.data import Dictionary
    from icecream import ic

    ic.enable()

    sents = [
        make_example_tree_01(),
        make_example_tree_02(),
        make_example_tree_03(),
        make_example_tree_04(),
        make_example_tree_05(),
        make_example_tree_06(),
        make_example_tree_07(),
        make_example_tree_08(),
        make_example_tree(),
    ]
    sents = [tree.uncollapse_unary() for tree in sents]

    acts, preorder_lists = zip(*[get_incremental_parse_actions(sent, collapse=False) for sent in sents])

    for tree, act in zip(sents, acts):
        print()
        tree.pretty_print()
        ic(act)

    # ic(node.label_without_flags, node.label_head)

    # this is just scaffolding, we need a label_dictionary to develop other stuff
    all_labels = {node.label_head for preorder_list in preorder_lists for node in preorder_list}
    all_flags = {f for preorder_list in preorder_lists for node in preorder_list for f in node.label_flags}
    ldict = Dictionary()
    ldict.add_symbol(ROOT_LABEL)
    ldict.add_symbol(NULL_LABEL)
    for label in all_labels.union(all_flags):
        ldict.add_symbol(label)
    ic(ldict.symbols)

    # this is just scaffolding, we need a label_dictionary to develop other stuff
    src_dict = Dictionary()
    src_tokens = pad_sequence(
        [src_dict.encode_line(line=tree.text, add_if_not_exist=True) for tree in sents],
        batch_first=True,
        padding_value=src_dict.pad(),
    )
    ic(src_dict.symbols, src_tokens)

    padded_preorder_nts = pad_sequence(
        [
            torch.tensor([ldict.index(node.label_head) for node in preorder_list], dtype=torch.long)
            for preorder_list in preorder_lists
        ],
        batch_first=True,
        padding_value=ldict.pad(),
    )
    preorder_spans = [torch.tensor([node.span for node in preorder_list]) for preorder_list in preorder_lists]

    def encode_flags_batched(flags, ldict):
        encoded_flags = [
            pad_sequence(
                [
                    torch.tensor([ldict.index(f) for f in node_flags], dtype=torch.long)
                    if node_flags
                    else torch.tensor([ldict.pad()], dtype=torch.long)
                    for node_flags in seq_flags
                ],
                batch_first=True,
                padding_value=ldict.pad(),
            )
            for seq_flags in flags
        ]
        return collate_2d(encoded_flags, pad_idx=ldict.pad())

    padded_preorder_flags = encode_flags_batched(
        [[node.label_flags for node in preorder_list] for preorder_list in preorder_lists], ldict
    )

    padded_tgt_parent_flags = encode_flags_batched(
        [[action.parent.label_flags for action in seq_acts] for seq_acts in acts], ldict
    )

    padded_tgt_preterm_flags = encode_flags_batched(
        [[action.preterminal.label_flags for action in seq_acts] for seq_acts in acts], ldict
    )

    # ic(padded_tgt_parent_flags, padded_tgt_preterm_flags)

    # bsz x seq x flags
    ic(padded_preorder_flags)
    padded_preorder_spans = pad_sequence(preorder_spans, batch_first=True, padding_value=0)
    padded_input_masks, padded_chain_masks = [], []
    max_acts = max(len(seq_acts) for seq_acts in acts)
    nwords_per_step = torch.zeros(len(acts), max_acts, dtype=torch.long)
    for step in range(max_acts):
        padded_inputs_step = []
        padded_chain_step = []
        for idx, seq_acts in enumerate(acts):
            act_inputs = torch.zeros(len(preorder_lists[idx]), dtype=torch.bool)
            act_chain = torch.zeros(len(preorder_lists[idx]), dtype=torch.bool)
            if step < len(seq_acts):
                act_inputs[seq_acts[step].preorder_indices] = 1
                act_chain[seq_acts[step].right_chain_indices] = 1
                nwords_per_step[idx, step] = seq_acts[step].nwords
            padded_inputs_step.append(act_inputs)
            padded_chain_step.append(act_chain)
        padded_input_masks.append(pad_sequence(padded_inputs_step, batch_first=True, padding_value=0))
        padded_chain_masks.append(pad_sequence(padded_chain_step, batch_first=True, padding_value=0))
    padded_input_masks = pad_sequence(padded_input_masks, batch_first=True, padding_value=0).transpose(0, 1)
    padded_chain_masks = pad_sequence(padded_chain_masks, batch_first=True, padding_value=0).transpose(0, 1)

    ic([t.long() for t in padded_input_masks])
    ic([t.long() for t in padded_chain_masks])
    ic(padded_preorder_nts)

    ic(acts)

    bsz = len(sents)
    seq_len = max(len(sent.leaves) for sent in sents)

    # tgt_depths: bsz x nsteps  ->  nsteps x bsz
    tgt_depths = pad_sequence(
        [torch.tensor([act.depth for act in seq_acts]) for seq_acts in acts],
        batch_first=True,
        padding_value=PADDING_VALUE_FOR_NON_INDEX,
    ).transpose(0, 1)
    tgt_parents = pad_sequence(
        [torch.tensor([ldict.index(act.parent.label_without_flags) for act in seq_acts]) for seq_acts in acts],
        batch_first=True,
        padding_value=ldict.pad(),
    )
    tgt_preterms = pad_sequence(
        [torch.tensor([ldict.index(act.preterminal.label_without_flags) for act in seq_acts]) for seq_acts in acts],
        batch_first=True,
        padding_value=ldict.pad(),
    )
    tgt_padding_mask = tgt_depths.eq(PADDING_VALUE_FOR_NON_INDEX).T

    ic.enable()
    ic(tgt_depths)
    ic(tgt_parents)
    ic(tgt_preterms)
    ic(ldict.string(tgt_preterms[0]))
    ic(tgt_padding_mask.long())
    ic(padded_preorder_nts)

    sample = {
        "net_input": {
            "src_tokens": src_tokens,
            "preorder_nts": padded_preorder_nts,
            "preorder_mask": padded_input_masks,
            "chain_mask": padded_chain_masks,
            "preorder_spans": padded_preorder_spans,
            "preorder_flags": padded_preorder_flags,
            "nwords_per_step": nwords_per_step,
        },
        "nwords": torch.tensor([len(sent.leaves) for sent in sents], dtype=torch.long),
        "target_depths": tgt_depths,
        "target_padding_mask": tgt_padding_mask,
        "target_parents": tgt_parents,
        "target_preterminals": tgt_preterms,
        "target_parent_flags": padded_tgt_parent_flags,
        "target_preterm_flags": padded_tgt_preterm_flags,
        "preorder_nts": padded_preorder_nts,
        "preorder_mask": padded_input_masks,
        "chain_mask": padded_chain_masks,
        "preorder_spans": padded_preorder_spans,
        "preorder_flags": padded_preorder_flags,
        "nwords_per_step": nwords_per_step,
        "nsentences": bsz,
    }

    from omegaconf import OmegaConf

    cfg = OmegaConf.create()
    cfg.model = OmegaConf.create()
    cfg.model.graph_decoder = OmegaConf.create()
    cfg.model.graph_decoder = OmegaConf.merge(cfg, TreeGraphDecoderConfig())
    cfg.criterion = OmegaConf.create()
    cfg.criterion = OmegaConf.merge(cfg, IncrementalParserCriterionConfig())
    OmegaConf.set_struct(cfg, True)
    dec_cfg = cfg.model.graph_decoder

    # B x T
    word_padding_mask = torch.zeros(bsz, seq_len).bool()
    # B x T x C
    encoder_out = torch.rand((bsz, seq_len, dec_cfg.embedding_dim))

    # first example is deliberately one shorter than the other example, so replace the corresponding slot with padding
    word_padding_mask[0, -1] = 1
    encoder_out[word_padding_mask] = 0

    ic(sample)
    dec = TreeGraphDecoder(dec_cfg, root_label_index=ldict.index(ROOT_LABEL), padding_idx=ldict.pad(), num_labels=20)
    incr_parser_criterion = IncrementalParserCriterion(cfg=cfg.criterion, task=None)
    _ = incr_parser_criterion(model=None, sample=sample, encoder_out=encoder_out, dec=dec)
