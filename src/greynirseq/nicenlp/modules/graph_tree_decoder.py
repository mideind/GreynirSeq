# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, NamedTuple, Union

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
from greynirseq.nicenlp.criterions.incremental_parser import (
    IncrementalParserCriterion,
    IncrementalParserCriterionConfig,
)
from greynirseq.nicenlp.data.datasets import collate_2d

from icecream import ic


SYMBOL_ROOT, SYMBOL_NULL = "ROOT", "NULL"
PADDING_VALUE_FOR_NON_INDEX = -100
NULL_SPAN = (PADDING_VALUE_FOR_NON_INDEX, PADDING_VALUE_FOR_NON_INDEX)
TensorListOrTensor = Union[List[Tensor], Tensor]


@dataclass
class ConstituencyParserOutput:
    """Simple container for incremental outputs of constituency part of GraphTreeDecoder"""

    # simple container for incremental outputs
    parent_logits: TensorListOrTensor
    preterm_logits: TensorListOrTensor
    parent_flag_logits: TensorListOrTensor
    preterm_flag_logits: TensorListOrTensor
    attention: TensorListOrTensor


GraphDecoderOutput = NamedTuple("GraphDecoderOutput", [("constituency", ConstituencyParserOutput)])


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

        self.classification_heads = nn.ModuleDict()
        self.classification_heads["constit_parent"] = ScaffholdHead(self.embed_dim, self.num_labels, self.cfg.dropout)
        self.classification_heads["constit_preterm"] = ScaffholdHead(self.embed_dim, self.num_labels, self.cfg.dropout)
        self.classification_heads["constit_parent_flags"] = ScaffholdHead(
            self.embed_dim, self.num_labels, self.cfg.dropout
        )
        self.classification_heads["constit_preterm_flags"] = ScaffholdHead(
            self.embed_dim, self.num_labels, self.cfg.dropout
        )
        self.mlp_attention = SingleVectorMLPAttention(
            2 * self.embed_dim, self.embed_dim // 4, self.cfg.dropout, use_sigmoid=cfg.mlp_attn_is_sigmoid
        )

    def embed_spans(self, spans: Tensor, end_thresholds: Tensor = None):
        # spans: B x T x S
        # XXX: in LearnedPositionalEmbedding: "If positions is pre-computed then padding_idx should not be set."
        #      we inherit our positionalembedding from the encoder, so this will need to be adjusted
        span_starts = spans[:, :, 0]
        span_ends = spans[:, :, 1].clone()  # clone because we might mutate
        if end_thresholds is not None:
            for seq_idx in range(len(end_thresholds)):
                span_ends[seq_idx][end_thresholds[seq_idx] < span_ends[seq_idx]] = end_thresholds[seq_idx]

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
        _nwords = sample["nwords"]
        preorder_nts = sample["preorder_nts"]  # bsz x nodes
        preorder_mask = sample["preorder_mask"]  # nsteps x bsz x nodes
        chain_mask = sample["chain_mask"]  # nsteps x bsz x num_nodes
        preorder_spans = sample["preorder_spans"]  # bsz x num_nodes x 2
        nwords_per_step = sample["nwords_per_step"]  # nwords x bsz
        preorder_flags = sample["preorder_flags"]  # bsz x nodes x flags

        # bsz, max_seq_len, _encoder_embed_dim = encoder_out.shape
        nsteps, bsz, _num_nodes = preorder_mask.shape

        emb_preorder_nts = self.embed_labels(preorder_nts)
        # bsz x nodes x flags  ->  bsz x nodes x flags x features  ->  bsz x nodes x features
        emb_preorder_nts += self.embed_labels(preorder_flags).sum(dim=-2)

        # simple container for incremental outputs
        state = ConstituencyParserOutput([], [], [], [], [])
        for curr_step in range(nsteps):
            num_words_in_tree = (nwords_per_step[curr_step] - 1).clamp(0)
            root_spans = torch.stack([torch.zeros(bsz).long(), num_words_in_tree], dim=1)
            # root_spans: B x 2  ->  B x 1 x 2  ->  B x 1 x C
            root_span_emb = self.embed_spans(root_spans.unsqueeze(1), end_thresholds=num_words_in_tree)

            # root_emb: 0  ->  C  ->  B x 1 x C
            root_emb = self.embed_labels(torch.tensor(self.root_label_index)).tile((bsz, 1, 1))
            root_emb = root_emb + root_span_emb

            word_mask = lengths_to_padding_mask(nwords_per_step[curr_step]).logical_not()
            word_embs = encoder_out[:, : nwords_per_step[curr_step].max(), :][word_mask]
            word_embs = pad_sequence(word_embs.split(word_mask.sum(-1).tolist()), batch_first=True, padding_value=0)
            preorder_step_embs = emb_preorder_nts + self.embed_spans(preorder_spans, end_thresholds=num_words_in_tree)
            preorder_step_embs *= preorder_mask[curr_step].unsqueeze(-1)
            # sequences that are finished have 0 words this step
            root_mask = (nwords_per_step[curr_step] > 0).unsqueeze(-1)
            # print()
            # ic(word_mask.shape, root_mask.shape, preorder_mask[curr_step].shape)
            # ic(word_embs.shape, root_emb.shape, preorder_step_embs.shape)
            assert (
                word_mask.shape[1] + root_mask.shape[1] + preorder_mask[curr_step].shape[1]
                == word_embs.shape[1] + root_emb.shape[1] + preorder_step_embs.shape[1]
            )
            input_mask = torch.cat([word_mask, root_mask, preorder_mask[curr_step]], dim=1)
            input_embs = torch.cat([word_embs, root_emb, preorder_step_embs], dim=1) * input_mask.unsqueeze(-1)

            # bsz x (num_nodes+1) x features
            x = self.forward_nodes(input_embs, self_attn_padding_mask=input_mask)

            # chain_nodes x features
            output_chain_mask = torch.cat([torch.zeros_like(word_mask), root_mask, chain_mask[curr_step]], dim=1)
            right_chain_outputs = x[output_chain_mask].split(output_chain_mask.sum(dim=-1).tolist())
            # bsz x maxchainlen x features
            right_chain_outputs = pad_sequence(right_chain_outputs, batch_first=True, padding_value=0)

            word_output_idxs = (nwords_per_step[curr_step].clone() - 1).clamp(min=0)
            is_alive = nwords_per_step[curr_step] > 0
            # bsz x features  ->  bsz x 1 x features
            attending_words = (x[torch.arange(bsz), word_output_idxs, :] * is_alive.unsqueeze(-1)).unsqueeze(1)

            attn_output_features, attn = self.mlp_attention(right_chain_outputs, attending_words)
            assert attending_words.shape == attn_output_features.shape
            # merge attention output and newest word, this should eithger be an add or cat
            clsf_features = attending_words + attn_output_features

            state.parent_logits.append(self.classification_heads["constit_parent"](clsf_features))
            state.preterm_logits.append(self.classification_heads["constit_preterm"](clsf_features))
            state.parent_flag_logits.append(self.classification_heads["constit_parent_flags"](clsf_features))
            state.preterm_flag_logits.append(self.classification_heads["constit_preterm_flags"](clsf_features))
            state.attention.append(attn.squeeze(2))

        state.parent_logits = torch.cat(state.parent_logits, dim=1)
        state.preterm_logits = torch.cat(state.preterm_logits, dim=1)
        state.parent_flag_logits = torch.cat(state.parent_flag_logits, dim=1)
        state.preterm_flag_logits = torch.cat(state.preterm_flag_logits, dim=1)

        return state


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
    ldict.add_symbol(SYMBOL_ROOT)
    ldict.add_symbol(SYMBOL_NULL)
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
    nwords_per_step = torch.zeros(max_acts, len(acts), dtype=torch.long)
    for step in range(max_acts):
        padded_inputs_step = []
        padded_chain_step = []
        for idx, seq_acts in enumerate(acts):
            act_inputs = torch.zeros(len(preorder_lists[idx]), dtype=torch.bool)
            act_chain = torch.zeros(len(preorder_lists[idx]), dtype=torch.bool)
            if step < len(seq_acts):
                act_inputs[seq_acts[step].preorder_indices] = 1
                act_chain[seq_acts[step].right_chain_indices] = 1
                nwords_per_step[step, idx] = seq_acts[step].nwords
            padded_inputs_step.append(act_inputs)
            padded_chain_step.append(act_chain)
        padded_input_masks.append(pad_sequence(padded_inputs_step, batch_first=True, padding_value=0))
        padded_chain_masks.append(pad_sequence(padded_chain_step, batch_first=True, padding_value=0))
    padded_input_masks = pad_sequence(padded_input_masks, batch_first=True, padding_value=0)
    padded_chain_masks = pad_sequence(padded_chain_masks, batch_first=True, padding_value=0)

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
        "net_input": {  # this would normally be here when training, this input for bert encoder
            "src_tokens": src_tokens
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

    # B x T
    word_padding_mask = torch.zeros(bsz, seq_len).bool()
    # B x T x C
    encoder_out = torch.rand((bsz, seq_len, TreeGraphDecoderConfig.embed_dim))

    # first example is deliberately one shorter than the other example, so replace the corresponding slot with padding
    word_padding_mask[0, -1] = 1
    encoder_out[word_padding_mask] = 0

    ic(sample)
    dec = TreeGraphDecoder(TreeGraphDecoderConfig, root_label_index=ldict.index(ROOT), padding_idx=ldict.pad())
    # dec(encoder_out=encoder_out, sample=sample)
    incr_parser_criterion = IncrementalParserCriterion(cfg=IncrementalParserCriterionConfig, task=None)
    _ = incr_parser_criterion(model=None, sample=sample, encoder_out=encoder_out, dec=dec)
