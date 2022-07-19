# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import logging
from dataclasses import dataclass, field
from typing import Any, List

import torch
from fairseq.dataclass import FairseqDataclass
from fairseq.models import register_model
from fairseq.models.roberta.model import RobertaModel, roberta_base_architecture
from fairseq.utils import safe_hasattr
from torch import Tensor, nn
from torch.nn.utils.rnn import pad_sequence

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
    and adds a few transformer-layers for graph-representations and decoding."""

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
            state_dict["graph_decoder.embed_positions.weight"] = state_dict[
                "encoder.sentence_encoder.embed_positions.weight"
            ].clone()
            if "graph_decoder.embed_word_positions.weight" in state_dict:
                state_dict["graph_decoder.embed_word_positions.weight"] = state_dict[
                    "encoder.sentence_encoder.embed_positions.weight"
                ].clone()

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
            num_labels=task.num_labels,
            project_input_from=encoder.args.encoder_embed_dim,
        )
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
        word_mask: Tensor,
        preorder_depths: Tensor,
        **kwargs,
    ):
        """Arguments:
        preorder_nts: bsz x nodes
        preorder_mask: nsteps x bsz x nodes
        chain_mask: nsteps x bsz x num_nodes
        preorder_spans: bsz x num_nodes x 2
        nwords_per_step: bsz x nsteps
        preorder_flags: bsz x nodes x flags
        word_mask_w_bos: bsz x ntokens
        """
        encoder_out, _extra = self.encoder(src_tokens, features_only=True, return_all_hiddens=False, **kwargs)
        _, _, enc_embed_dim = encoder_out.shape
        if word_mask.sum(dim=-1).max() != nwords_per_step.max():
            breakpoint()
            print()
        if word_mask.shape != encoder_out.shape[:2]:
            print(word_mask.shape, encoder_out.shape[:2], sep="\n")
            breakpoint()
            print()
        words = encoder_out.masked_select(word_mask.unsqueeze(-1).bool()).reshape(-1, enc_embed_dim)
        words_padded = pad_sequence(words.split(word_mask.sum(dim=-1).tolist()), padding_value=0, batch_first=True)
        decoder_out = self.graph_decoder(
            encoder_out=words_padded,
            preorder_nts=preorder_nts,
            preorder_mask=preorder_mask,
            chain_mask=chain_mask,
            preorder_spans=preorder_spans,
            nwords_per_step=nwords_per_step,
            preorder_flags=preorder_flags,
            preorder_depths=preorder_depths,
            **kwargs,
        )
        return decoder_out

    @classmethod
    def from_pretrained(
        cls, model_name_or_path, checkpoint_file="model.pt", data_name_or_path=".", bpe="gpt2", **kwargs
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

        from fairseq.data import encoders

        self.bpe = encoders.build_bpe(cfg.bpe)

        # this is useful for determining the device
        self.register_buffer("_float_tensor", torch.tensor([0], dtype=torch.float))

    def predict_sample(self, sample):
        ...

    def predict(self, sentences, device="cuda"):
        ...

    def prepare_sentences(self, sentences: List[str]):
        return self.task.prepare_sentences(sentences)
