# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

from omegaconf import II

import torch
from torch import Tensor
from torch import nn
from torch.nn.utils.rnn import pad_sequence

from fairseq import utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.modules.transformer_sentence_encoder import TransformerSentenceEncoderLayer
from fairseq.modules import PositionalEmbedding, FairseqDropout, LayerNorm
from fairseq.dataclass import FairseqDataclass

from greynirseq.nicenlp.utils.constituency.greynir_utils import NonterminalNode, TerminalNode, Node
from greynirseq.nicenlp.utils.constituency.incremental_parsing import (
    NULL,
    ROOT,
    ParseAction,
    get_incremental_parse_actions,
    get_right_chain,
    parse_by_actions,
    mark_preterminal_and_parent,
    get_preorder_index_of_right_chain,
)
from greynirseq.nicenlp.utils.constituency.scratch_incremental import IncrementalParser
from greynirseq.nicenlp.models.simple_parser import ChartParserHead
from greynirseq.nicenlp.criterions.incremental_parser import IncrementalParserCriterion

from icecream import ic


@dataclass
class TreeGraphDecoderConfig:
    embed_dim: int = field(default=64, metadata={"help": "Embedding dimension"})
    layers: int = field(default=2, metadata={"help": "Number of layers"})
    max_positions: int = 512  # II("max_source_positions")
    learned_positions: bool = True  # II("encoder.learned_pos")
    factored_embeddings: bool = False
    layernorm_embedding: bool = True
    dropout: float = 0.1
    mlp_attn_is_sigmoid: bool = True


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
#     apply_bert_init: bool = False,
#     activation_fn: str = "relu",
#     learned_pos_embedding: bool = True,
#     embed_scale: float = None,
#     freeze_embeddings: bool = False,
#     n_trans_layers_to_freeze: int = 0,
#     export: bool = False,
#     traceable: bool = False,
#     q_noise: float = 0.0,
#     qn_block_size: int = 8,


@dataclass
class _TransformerSentenceEncoderLayerConfig:
    embedding_dim: int = 64  # 768
    ffn_embedding_dim: int = 128  # 3072
    num_attention_heads: int = 2  # 8
    dropout: float = 0.1
    attention_dropout: float = 0.1
    activation_dropout: float = 0.1
    activation_fn: str = "relu"
    export: bool = False
    q_noise: float = 0.0
    qn_block_size: int = 8


class TreeGraphDecoder(nn.Module):
    def __init__(
        self,
        cfg: TreeGraphDecoderConfig,
        embed_positions: Any = None,
        padding_idx: int = 0,
        num_labels: int = 20,
        root_label_index: int = 0,
    ):
        # XXX: dropout is missing
        super().__init__()
        self.cfg = cfg
        self.embed_dim = cfg.embed_dim
        self.num_labels = num_labels
        self.embed_positions = embed_positions
        self.padding_idx = padding_idx
        self.root_label_index = root_label_index
        if embed_positions is None:
            self.embed_positions = PositionalEmbedding(
                # cfg.max_positions, self.embed_dim, self.padding_idx, learned=cfg.learned_positions
                cfg.max_positions,
                self.embed_dim,
                None,
                learned=cfg.learned_positions,
            )

        # XXX: these are nonterminal labels, as well as attribute labels
        self.embed_labels = nn.Embedding(self.num_labels, self.embed_dim, self.padding_idx)

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(**asdict(_TransformerSentenceEncoderLayerConfig()))
                for _ in range(cfg.layers)
            ]
        )
        self.dropout_module = FairseqDropout(cfg.dropout, module_name=self.__class__.__name__)
        self.layernorm_embedding = None
        if self.cfg.layernorm_embedding:
            self.layernorm_embedding = LayerNorm(self.embed_dim)
        self.layer_norm = LayerNorm(self.embed_dim)

        self.task_head1 = ScaffholdHead(self.embed_dim, self.num_labels, self.cfg.dropout)
        # self.heads = nn.ModuleDict()
        # self.heads[""]
        self.task_head2 = ScaffholdHead(self.embed_dim, self.num_labels, self.cfg.dropout)
        self.mlp_attention = SingleVectorMLPAttention(
            2 * self.embed_dim, self.embed_dim // 4, self.cfg.dropout, use_sigmoid=cfg.mlp_attn_is_sigmoid
        )

    def forward_increment(self, encoder_out: Tensor):
        pass

    def embed_span_position(self, spans: Tensor, max_end: int = None):
        # one_if_include = 1 if include_current_pos else 0
        # spans: B x T x S
        # XXX: in LearnedPositionalEmbedding: "If positions is pre-computed then padding_idx should not be set."
        #      we inherit our positionalembedding from the encoder, so this will need to be adjusted

        # span_starts = spans[:, : (curr_step + one_if_include), 0].clone()
        # span_starts[span_starts < 0] = 0

        span_starts = spans[:, :, 0]

        span_ends = spans[:, :, 1].clone()  # clone because we might mutate
        # span_ends[span_ends < 0] = 0
        # span_ends = spans[:, : (curr_step + one_if_include), 1].clone()
        # span_ends[curr_step <= span_ends] = curr_step  # with 1 word, ROOT will be (start=0, end=1)

        if max_end is not None:
            span_ends = span_ends[max_end < span_ends] = max_end

        span_embs = (
            self.embed_positions(None, positions=span_starts) + self.embed_positions(None, positions=span_ends)
        ) / 2

        return span_embs

    def forward_nodes(self, x: Tensor, self_attn_padding_mask: Tensor):
        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)

        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        for idx, layer in enumerate(self.layers):
            x, _layer_attn = layer(x, self_attn_padding_mask=self_attn_padding_mask, self_attn_mask=None)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        return x

    def forward(
        self,
        # prev_output_tokens,
        encoder_out: Tensor,
        sample: Dict[str, Any],
        # incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
    ):
        ic.enable()
        nwords = sample["nwords"]
        word_padding_mask = lengths_to_padding_mask(nwords)
        tgt_depths = sample["target_depths"]
        tgt_padding_mask = sample["target_padding_mask"]

        tgt_parents = sample["target_parents"]
        parent_mask = sample["target_parent_mask"]
        tgt_parent_spans = sample["target_parent_spans"]
        tgt_preterms = sample["target_preterminals"]
        preterm_mask = sample["target_preterminal_mask"]
        tgt_preterm_spans = sample["target_preterminal_spans"]

        bsz, max_seq_len, _encoder_embed_dim = encoder_out.shape

        # assurance that max_seq_len is actually max sequence length
        assert (parent_mask + preterm_mask).long().sum(dim=-1).max() == max_seq_len, "Unexpected unnecessary padding"

        # number of nodes in _current_ incremental tree that is about to be embedded
        num_preterms = preterm_mask.cumsum(dim=-1).roll(1)
        num_preterms[:, 0] = 0
        num_parents = parent_mask.cumsum(dim=-1).roll(1)
        num_parents[:, 0] = 0
        # ic(parent_mask.long(), preterm_mask.long())
        # ic(num_preterms.long(), num_parents.long(), incremental_seq_lens)

        # XXX: We mask Null label for embedding nodes in self_attention_mask
        # XXX: But it is still a target prediction for the classifier!

        parent_logits = []
        preterm_logits = []
        depth_logits = []
        attentions = []

        narrow_to_step = None
        narrow_to_step = 4
        for curr_step in range(max_seq_len):
            if narrow_to_step is not None and narrow_to_step != curr_step:
                continue
            step_preterm_mask = preterm_mask[:, :curr_step]
            step_parent_mask = parent_mask[:, :curr_step]
            step_word_padding_mask = word_padding_mask[:, : (curr_step + 1)]
            step_parent_spans = tgt_parent_spans[:, :curr_step]
            step_preterm_spans = tgt_preterm_spans[:, :curr_step]

            # ic(curr_step, num_preterms[:, curr_step], num_parents[:, curr_step])
            root_spans = torch.zeros((bsz, 1, 2), dtype=torch.long)
            root_spans[:, 0, 1] = (
                curr_step + num_parents[:, curr_step] + num_preterms[:, curr_step]
            )  # number of nodes in tree
            root_span_emb = self.embed_span_position(root_spans)
            root_emb = self.embed_labels(torch.tensor(self.root_label_index)).tile((bsz, 1, 1))  # B x 1 x C
            # ic(root_span_emb.shape, root_emb.shape)
            root_emb = root_emb + root_span_emb

            word_positions = torch.arange(curr_step + 1).tile((bsz, 1))
            word_embs = encoder_out[:, : (curr_step + 1), :]
            if self.cfg.factored_embeddings:
                assert False
            else:
                word_embs = word_embs + self.embed_positions(None, positions=word_positions)

            if step_parent_mask.sum() > 0:
                parent_embs = self.embed_labels(tgt_parents[:, :curr_step])
                parent_span_emb = self.embed_span_position(
                    step_parent_spans * step_parent_mask.unsqueeze(dim=-1)
                )  # set span of NULL/padding to (0, 0)
                parent_embs = parent_embs + parent_span_emb
            else:
                parent_embs = word_embs.new_zeros(bsz, 0, _encoder_embed_dim)

            if step_preterm_mask.sum() > 0:
                preterm_embs = self.embed_labels(tgt_preterms[:, :curr_step])
                preterm_span_emb = self.embed_span_position(
                    step_preterm_spans * step_preterm_mask.unsqueeze(dim=-1)
                )  # set span of NULL/padding to (0, 0)
                preterm_embs = preterm_embs + preterm_span_emb
            else:
                preterm_embs = word_embs.new_zeros(bsz, 0, _encoder_embed_dim)

            empty_mask = step_preterm_mask.new_zeros(bsz, 0)
            self_attn_padding_mask = torch.cat(
                [
                    preterm_mask.new_ones(bsz, 1),
                    step_preterm_mask if step_preterm_mask.sum() > 0 else empty_mask,
                    step_parent_mask if step_parent_mask.sum() > 0 else empty_mask,
                    step_word_padding_mask.logical_not(),
                ],
                dim=1,
            ).logical_not()
            # ic(root_emb.shape, preterm_embs.shape, parent_embs.shape, word_embs.shape)
            x = torch.cat([root_emb, preterm_embs, parent_embs, word_embs], dim=1)
            # ic(embs.shape, self_attn_padding_mask.shape)
            ic(curr_step, self_attn_padding_mask.long())
            assert x.shape[1] == self_attn_padding_mask.shape[1]
            x = self.forward_nodes(x, self_attn_padding_mask=self_attn_padding_mask)

            # for right chain:
            #   select rightmost preterm for every seq (there is always either 0 or 1 preterm on the right chain)
            #   always select root
            #   to select parents:
            #       find max value of right fence for all parent spans
            #       all spans that end in the same place are on the right chains

            # this should equal to nwords in curr_step, except for sequences shorter than max_seq_len
            parent_right_chain_ends = step_parent_spans[:, :, 1:].max(dim=1).values  # select end points of all spans
            parent_right_chain_mask = (parent_right_chain_ends == step_parent_spans[:, :, 1]) * step_parent_mask
            parent_right_chain_mask = (parent_right_chain_ends == step_parent_spans[:, :, 1]) * step_parent_mask

            preterm_right_chain_ends = step_preterm_spans[:, :, 1].max(dim=1).indices
            preterm_right_chain_embs = preterm_embs[torch.arange(bsz), preterm_right_chain_ends].unsqueeze(1)
            # after unsqueeze: B x 1 x C


            # XXX: the above is wrong, we need to select these (with masks or something) from x (ie output from layer stack)
            # right_chain_parents_emb = parent_embs * parent_right_chain_mask.unsqueeze(-1).type_as(parent_embs)
            # x = torch.cat([root_emb, preterm_embs, parent_embs, word_embs], dim=1)

            ic(parent_right_chain_mask.long())
            ic(parent_embs.shape, parent_right_chain_mask.shape)
            breakpoint()

            # this is just a dummy set up to complete a forward pass
            right_chain_nodes = x[:, :-1, :]  # this is just a dummy variable
            newest_word = x[:, -1:, :]
            # XXX: we have yet to extract the right chain!
            attn_output_features, attn  = self.mlp_attention(right_chain_nodes, newest_word)

            # merge attention output and newest word, this could probably just as well be an add
            # clsf_features = torch.cat([newest_word, attn_output_features], dim=-1)
            assert newest_word.shape == attn_output_features.shape
            clsf_features = newest_word + attn_output_features

            step_parent_logits = self.task_head1(clsf_features)
            step_preterm_logits = self.task_head2(clsf_features)

            parent_logits.append(step_parent_logits)
            preterm_logits.append(step_preterm_logits)
            attentions.append(attn)

        # T * (B x 1 x C)  ->  B x T x C
        parent_logits = torch.cat(parent_logits, dim=1)
        preterm_logits = torch.cat(preterm_logits, dim=1)

        breakpoint()
        IncrementalParserCriterion.compute_whole(
            tgt_padding_mask=tgt_padding_mask,
            tgt_parents=tgt_parents,
            tgt_preterms=tgt_preterms,
            tgt_depths=tgt_depths,
            parent_logits=parent_logits,
            preterm_logits=preterm_logits,
            attention=attentions,
        )

        ic(tgt_parents.shape, step_parent_logits.shape, parent_logits.shape)
        breakpoint()


class ScaffholdHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        self.input_dim = input_dim
        self.inner_dim = input_dim
        self.output_dim = output_dim

        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = LayerNorm(self.inner_dim)
        self.activation_fn = utils.get_activation_fn("relu")

        self.dense = nn.Linear(self.input_dim, self.inner_dim)
        self.dense2 = nn.Linear(self.inner_dim, self.inner_dim)
        self.out_proj = nn.Linear(self.inner_dim, self.output_dim)

    def forward(self, x, **kwargs):
        x = self.dropout(x)
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.activation_fn(x)
        x = self.dense2(x)
        x = self.activation_fn(x)
        logits = self.out_proj(x)
        return logits


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
        self.out_proj = nn.Linear(self.inner_dim, 1)

    def forward(self, right_chain_nodes, newest_word):
        """The vectors in v attend to the set of vectors in x (in batched manner)
           v has shape (Batch,    1, Features)
           x has shape (Batch, Time, Features)
        Returns a version of x where the features of a single v (respectively for each batch dimension)
        have been concatenated to all vectors in the time dimension """
        _, right_chain_len, _ = right_chain_nodes.shape
        # B x 1 x C  ->  B x RCHAIN x C
        tiled_newest_word = torch.tile(newest_word, dims=(1, right_chain_len, 1))
        # concatenate along feature dimension
        x = torch.cat([right_chain_nodes, tiled_newest_word], dim=-1)
        # ic(right_chain_nodes.shape, newest_word.shape, tiled_newest_word.shape, x.shape)

        x = self.dropout(x)
        x = self.dense(x)
        x = self.layer_norm(x)
        x = self.activation_fn(x)
        x = self.dense2(x)
        x = self.activation_fn(x)
        # attn: B x T x 1
        attn = self.out_proj(x)
        # ic(attn.shape)

        # add the contribution of each node in the right-chain to the output, weighted by its attention
        # (B x T x C) * (B x T x 1)
        if self.use_sigmoid:
            output_features = (right_chain_nodes * attn.sigmoid()).sum(dim=1)
        else:
            output_features = (right_chain_nodes * attn.softmax(dim=1)).sum(dim=1)

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
    #                  S0>IP
    #       _____________|____
    #      |                  VP
    #      |       ___________|_____
    #      NP   VP-AUX       ADVP   VP
    #      |      |           |     |
    #      x      x           x     x
    #      |      |           |     |
    #   bíllinn hafði        ekki runnið
    billinn = NonterminalNode("NP", [TerminalNode("bíllinn", "x")])
    hafdi = NonterminalNode("VP-AUX", [TerminalNode("hafði", "x")])
    ekki = NonterminalNode("ADVP", [TerminalNode("ekki", "x")])
    runnid = NonterminalNode("VP", [TerminalNode("runnið", "x")])
    hafdi_ekki_runnid = NonterminalNode("VP", [hafdi, ekki, runnid])
    ip = NonterminalNode("IP", [billinn, hafdi_ekki_runnid])
    sentence = NonterminalNode("S0", [ip]).collapse_unary()
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


@dataclass
class ParseNodeInfo:
    # this is just temporary for debugging purposes (delete me)
    preorder_index: int
    nonterminal: str
    is_preterminal: bool
    span: Any
    is_right_chain: bool
    depth: int


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
    ]

    # for tree in sents:
    #     tree.pretty_print()

    acts = [get_incremental_parse_actions(sent) for sent in sents]
    sent01, sent02, *_ = sents
    act01, act02, *_ = acts
    mytree = make_example_tree_08()
    mark_preterminal_and_parent(mytree)
    _ = mytree.span
    # rod = mark_preorder(mytree)
    rod = mytree.preorder_list()
    rchain = tuple(get_preorder_index_of_right_chain(mytree))

    get_incremental_parse_actions(mytree, verbose=True)
    ### exampel of using tree traversal to generate input data for every increment step in parser
    ic.enable()
    ic(rod)
    ic([node.preorder_index for node in rod])
    # ic([(node.preorder_index, node.nonterminal, node.is_preterminal, node.span, node.leaves) for node in rod])
    ic()

    preorder_nodeinfo = [ParseNodeInfo(node.preorder_index, node.nonterminal, node.is_preterminal, node.span, node.preorder_index in rchain, node.depth) for node in rod]
    ic(preorder_nodeinfo)
    ic([preorder_nodeinfo[i] for i in rchain])
    ic(rchain)
    ic(acts[-1])
    mytree.pretty_print()
    breakpoint()
    # sent01.pretty_print()
    # pprint(act01)

    ldict = Dictionary()
    SYMBOL_ROOT, SYMBOL_NULL = "ROOT", "NULL"
    ldict.add_symbol(SYMBOL_ROOT)
    for seq_acts in acts:
        for act in seq_acts:
            ldict.add_symbol(act.parent.label)
            ldict.add_symbol(act.preterminal.label)
    print()
    pprint(acts)
    print()

    print("ldict.symbols", ldict.symbols, sep="\n", end="\n\n")

    bsz = len(sents)
    seq_len = max(len(sent.leaves) for sent in sents)

    padding_value_for_non_index = -100
    tgt_depths = pad_sequence(
        [torch.tensor([act.depth for act in seq_acts]) for seq_acts in acts],
        batch_first=True,
        padding_value=padding_value_for_non_index,
    )
    tgt_parents = pad_sequence(
        [torch.tensor([ldict.index(act.parent.label) for act in seq_acts]) for seq_acts in acts],
        batch_first=True,
        padding_value=ldict.pad(),
    )
    tgt_preterms = pad_sequence(
        [torch.tensor([ldict.index(act.preterminal.label) for act in seq_acts]) for seq_acts in acts],
        batch_first=True,
        padding_value=ldict.pad(),
    )
    tgt_padding_mask = tgt_depths.eq(padding_value_for_non_index)

    print()
    print("tgt_depths", tgt_depths, sep="\n", end="\n\n")
    print("tgt_parents", tgt_parents, sep="\n", end="\n\n")
    print("tgt_preterms", tgt_preterms, sep="\n", end="\n\n")

    print("tgt_preterm_labels", ldict.string(tgt_preterms[0]), ldict.string(tgt_preterms[1]), sep="\n", end="\n\n")

    print("tgt_padding_mask", tgt_padding_mask.long(), sep="\n", end="\n\n")

    tgt_parent_mask = (tgt_parents.eq(ldict.pad()) | tgt_parents.eq(ldict.index(SYMBOL_NULL))).logical_not()
    tgt_preterm_mask = (tgt_preterms.eq(ldict.pad()) | tgt_preterms.eq(ldict.index(SYMBOL_NULL))).logical_not()
    print("tgt_parent_mask", tgt_parent_mask.long(), sep="\n", end="\n\n")
    print("tgt_preterm_mask", tgt_preterm_mask.long(), sep="\n", end="\n\n")

    NULL_SPAN = (padding_value_for_non_index, padding_value_for_non_index)
    tgt_parent_spans = []
    for seq_acts in acts:
        flat_seq_acts = []
        for act in seq_acts:
            if act.parent.span is not None:
                flat_seq_acts.extend(act.parent.span)
            else:
                flat_seq_acts.extend(NULL_SPAN)
        tgt_parent_spans.append(torch.tensor(flat_seq_acts, dtype=torch.long))
    tgt_parent_spans = pad_sequence(
        tgt_parent_spans, batch_first=True, padding_value=padding_value_for_non_index
    ).reshape(bsz, -1, 2)
    print("tgt_parent_spans", tgt_parent_spans, sep="\n", end="\n\n")

    tgt_preterm_spans = []
    for seq_acts in acts:
        flat_seq_acts = []
        for act in seq_acts:
            if act.preterminal.span is not None:
                flat_seq_acts.extend(act.preterminal.span)
            else:
                flat_seq_acts.extend(NULL_SPAN)
        tgt_preterm_spans.append(torch.tensor(flat_seq_acts, dtype=torch.long))
    tgt_preterm_spans = pad_sequence(
        tgt_preterm_spans, batch_first=True, padding_value=padding_value_for_non_index
    ).reshape(bsz, -1, 2)
    print("tgt_preterm_spans", tgt_preterm_spans, sep="\n", end="\n\n")

    sample = {
        # "net_input": {  # this would normally be here when training, this input for bert encoder
        #     "src_tokens": [1,2,3]
        # }
        "nwords": torch.tensor([len(sent.leaves) for sent in sents], dtype=torch.long),
        "target_depths": tgt_depths,
        "target_padding_mask": tgt_padding_mask,
        "target_parents": tgt_parents,
        "target_parent_mask": tgt_parent_mask,
        "target_parent_spans": tgt_parent_spans,
        "target_preterminals": tgt_preterms,
        "target_preterminal_mask": tgt_preterm_mask,
        "target_preterminal_spans": tgt_preterm_spans,
    }

    # B x T
    word_padding_mask = torch.zeros(bsz, seq_len).bool()
    # B x T x C
    encoder_out = torch.rand((bsz, seq_len, TreeGraphDecoderConfig.embed_dim))

    # first example is deliberately one shorter than the other example, so replace the corresponding slot with padding
    word_padding_mask[0, -1] = 1
    encoder_out[word_padding_mask] = 0

    pprint(sample)
    dec = TreeGraphDecoder(TreeGraphDecoderConfig, root_label_index=ldict.index(ROOT))
    dec(encoder_out=encoder_out, sample=sample)
