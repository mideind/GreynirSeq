# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import logging
from typing import List, Any, Dict
from dataclasses import dataclass, field, asdict

from omegaconf import II

import torch
from fairseq import utils
from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models.roberta.model import RobertaModel, roberta_base_architecture
from fairseq.modules import LayerNorm
from fairseq.utils import safe_hasattr, safe_getattr
from fairseq.models.roberta.model import roberta_base_architecture

from torch import nn, Tensor
from torch.nn.utils.rnn import pad_sequence

from fairseq.dataclass import FairseqDataclass

from greynirseq.nicenlp.modules.graph_tree_decoder import TreeGraphDecoder, TreeGraphDecoderConfig

logger = logging.getLogger(__name__)


@dataclass
class GraphTreeParserConfig(FairseqDataclass):
    graph_decoder: Any = TreeGraphDecoderConfig
    encoder: Any = None


@register_model("graph_tree_parser", dataclass=GraphTreeParserConfig)
class GraphTreeParserModel(RobertaModel):
    """Graph-tree constituency parser model, builds on a pre-trained base model such as BERT
    and adds a few transformer-layers for graph-representations and decoding. """

    def __init__(self, cfg, encoder, decoder, task):
        super().__init__(cfg, encoder)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        sentence_encoder = self.encoder.sentence_encoder
        self.decoder = decoder

        # Not freezing embeddings degrades result according to multiple papers (e.g. Kitaev)
        freeze_module_params(sentence_encoder.embed_tokens)
        freeze_module_params(sentence_encoder.embed_positions)
        freeze_module_params(sentence_encoder.layernorm_embedding)

        _num_embeddings, _embed_dim = sentence_encoder.embed_tokens.weight.shape
        self.task = task

    @classmethod
    def build_model(cls, cfg, task):
        """Build a new model instance."""

        from omegaconf import OmegaConf

        assert cfg.encoder is None, "Currently we dont support encoder overrides"
        if OmegaConf.is_config(cfg):
            OmegaConf.set_struct(cfg, False)

        # make sure all arguments for roberta encoder are present
        roberta_base_architecture(cfg)

        if not safe_hasattr(cfg, "max_positions"):
            if not safe_hasattr(cfg, "tokens_per_sample"):
                cfg.tokens_per_sample = task.max_positions()
            cfg.max_positions = cfg.tokens_per_sample

        encoder = RobertaModel.build_model(cfg, task).encoder
        decoder = TreeGraphDecoder(
            cfg.graph_decoder,
            root_label_index=task.root_label_index,
            padding_idx=task.label_dictionary.pad(),
            embed_positions=encoder.sentence_encoder.embed_positions,
            num_labels=task.num_labels,
        )
        decoder.embed_spans(torch.arange(24).reshape(3,4,2))

        return cls(cfg, encoder, decoder, task)

    def forward(
        self,
        src_tokens: Tensor,
        preorder_nts: Tensor,
        preorder_mask: Tensor,
        chain_mask: Tensor,
        preorder_spans: Tensor,
        nwords_per_step: Tensor,
        preorder_flags: Tensor,
        **kwargs
    ):
        """
            preorder_nts: bsz x nodes
            preorder_mask: nsteps x bsz x nodes
            chain_mask: nsteps x bsz x num_nodes
            preorder_spans: bsz x num_nodes x 2
            nwords_per_step: nwords x bsz
            preorder_flags: bsz x nodes x flags
        """
        encoder_out, _extra = self.encoder(src_tokens, features_only=True, return_all_hiddens=False, **kwargs)
        decoder_out = self.decoder(
            encoder_out=encoder_out,
            preorder_nts=preorder_nts,
            preorder_mask=preorder_mask,
            chain_mask=chain_mask,
            preorder_spans=preorder_spans,
            nwords_per_step=nwords_per_step,
            preorder_flags=preorder_flags,
        )
        return decoder_out
