# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import logging
from typing import List, Any, Dict, Optional
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
    graph_decoder: TreeGraphDecoderConfig = TreeGraphDecoderConfig()
    encoder: Any = field(default=None)
    freeze_encoder_embeddings: bool = field(default=True)
    freeze_encoder_position_embeddings: bool = field(default=True)
    freeze_encoder_layers: bool = field(default=True)


@register_model("graph_tree_parser", dataclass=GraphTreeParserConfig)
class GraphTreeParserModel(RobertaModel):
    """Graph-tree constituency parser model, builds on a pre-trained base model such as BERT
    and adds a few transformer-layers for graph-representations and decoding. """

    def __init__(self, cfg: GraphTreeParserConfig, encoder, decoder, task):
        super().__init__(cfg, encoder)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        sentence_encoder = self.encoder.sentence_encoder
        self.graph_decoder = decoder

        # Not freezing embeddings degrades result according to multiple papers (e.g. Kitaev)
        if cfg.freeze_encoder_embeddings:
            freeze_module_params(sentence_encoder.embed_tokens)
            freeze_module_params(sentence_encoder.layernorm_embedding)
        if cfg.freeze_encoder_position_embeddings:
            freeze_module_params(sentence_encoder.embed_positions)

        if cfg.freeze_encoder_layers:
            for layer in range(len(sentence_encoder.layers)):
                freeze_module_params(sentence_encoder.layers[layer])

        _num_embeddings, _embed_dim = sentence_encoder.embed_tokens.weight.shape
        self.task = task

    def upgrade_state_dict_named(self, state_dict, name):
        old_keys = state_dict.keys()
        # older version of fairseq used decoder name for the roberta encoder
        # assert any(key.startswith("encoder") for key in old_keys)

        # rename emb_layer_norm -> layernorm_embedding
        for k in list(state_dict.keys()):
            if ".emb_layer_norm." in k:
                new_k = k.replace(".emb_layer_norm.", ".layernorm_embedding.")
                state_dict[new_k] = state_dict[k]
                del state_dict[k]

        # we are restoring from an icebert checkpoint and not a parser checkpoint
        if not any(key.startswith("graph_decoder") for key in old_keys):
            new_state_dict = self.state_dict()
            for new_key, value in new_state_dict.items():
                if new_key == "encoder.sentence_encoder.version" and new_key not in old_keys:
                    state_dict[new_key] = new_state_dict[new_key]
                elif not new_key.startswith("graph_decoder."):
                    continue
                state_dict[new_key] = value
            # inherit position encoding from bert model into decoder
            state_dict["graph_decoder.embed_positions.weight"] = state_dict["encoder.sentence_encoder.embed_positions.weight"]

        return super().upgrade_state_dict_named(state_dict, name)

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
        decoder_out = self.graph_decoder(
            encoder_out=encoder_out,
            preorder_nts=preorder_nts,
            preorder_mask=preorder_mask,
            chain_mask=chain_mask,
            preorder_spans=preorder_spans,
            nwords_per_step=nwords_per_step,
            preorder_flags=preorder_flags,
        )
        return decoder_out

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        checkpoint_file="model.pt",
        data_name_or_path=".",
        bpe="gpt2",
        **kwargs,
    ):
        from fairseq import hub_utils

        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return GraphTreeParserHubInterface(x["args"], x["task"], x["models"][0])


class GraphTreeParserHubInterface(nn.Module):
    def __init__(self, cfg, task, model):
        super().__init__()
        self.cfg = cfg
        self.task = task
        self.model = model

        # self.bpe = encoders.build_bpe(cfg.bpe)
    def predict_sample(self, sample):
        ...
    def predict(self, sentences, device="cuda"):
        ...
    def prepare_sentences(self, sentences: List[str]):
        ...
