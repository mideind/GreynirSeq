import itertools
import logging

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from fairseq import utils
from fairseq.models.roberta.model import base_architecture, RobertaModel
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models.roberta.model import RobertaEncoder
from fairseq.models import register_model, register_model_architecture

from greynirseq.nicenlp.utils.label_schema.label_schema import (
    make_vec_idx_to_dict_idx,
    make_mapped_group_masks,
    make_group_name_to_mapped_group_idxs,
)

from greynirseq.nicenlp.utils.constituency import token_utils
from greynirseq.nicenlp.models.multilabel_word_classification import (
    MultiLabelTokenClassificationHead
)


logger = logging.getLogger(__name__)


@register_model("icebert_pos")
class IceBERTPOS(RobertaModel):
    def __init__(self, args, encoder, task):
        super().__init__(args, encoder)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        sentence_encoder = self.decoder.sentence_encoder
        if args.freeze_embeddings:
            freeze_module_params(sentence_encoder.embed_tokens)
            freeze_module_params(sentence_encoder.segment_embeddings)
            freeze_module_params(sentence_encoder.embed_positions)
            freeze_module_params(sentence_encoder.emb_layer_norm)

        for layer in range(args.n_trans_layers_to_freeze):
            freeze_module_params(sentence_encoder.layers[layer])

        self.task = task
        self.pos_head = MultiLabelTokenClassificationHead(
            in_features=sentence_encoder.embedding_dim,
            out_features=self.task.num_labels,
            num_cats=self.task.num_cats,
            pooler_dropout=self.args.pooler_dropout,
        )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(IceBERTPOS, IceBERTPOS).add_args(parser)
        # fmt: off
        parser.add_argument('--freeze-embeddings', default=False,
                            help='Freeze transformer embeddings during fine-tuning')
        parser.add_argument('--n-trans-layers-to-freeze', default=0, type=int,
                            help='Number of transformer layers to freeze during fine-tuning')
        parser.add_argument('--tag-layer', default=12, type=int,
                            help='Which layer-activations to use to compute POS tags (note: 1-indexed)')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = RobertaEncoder(args, task.source_dictionary)
        return cls(args, encoder, task)

    def forward(
        self, src_tokens, features_only=False, return_all_hiddens=False, **kwargs
    ):
        _x, _extra = self.decoder(
            src_tokens, features_only, return_all_hiddens=True, **kwargs
        )

        x_tag = _extra["inner_states"][self.args.tag_layer - 1]
        x_tag = x_tag.transpose(0, 1)
        x = x_tag

        _, _, inner_dim = x.shape
        word_mask = kwargs["word_mask"]

        words = x.masked_select(word_mask.unsqueeze(-1).bool()).reshape(-1, inner_dim)
        nwords_w_bos = word_mask.sum(-1)

        (cat_logits, attr_logits) = self.pos_head(words)

        # (Batch * Time) x Depth -> Batch x Time x Depth
        cat_logits = pad_sequence(
            cat_logits.split((nwords_w_bos).tolist()), padding_value=0, batch_first=True
        )
        attr_logits = pad_sequence(
            attr_logits.split((nwords_w_bos).tolist()),
            padding_value=0,
            batch_first=True,
        )

        return (cat_logits, attr_logits), _extra

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
        return IceBERTPOSHubInterface(x["args"], x["task"], x["models"][0])

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + "." if name != "" else ""

        for key, value in self.pos_head.state_dict().items():
            path = prefix + "pos_head." + key
            logger.info("Initializing pos_head." + key)
            if path not in state_dict:
                state_dict[path] = value


class IceBERTPOSHubInterface(RobertaHubInterface):
    def predict_sample(self, sample):
        net_input = sample["net_input"]
        tokens = net_input["src_tokens"]

        word_mask = sample["net_input"]["word_mask"]
        nwords = sample["net_input"]["word_mask"].sum(-1)

        (cat_logits, attr_logits), _extra = self.model(tokens, word_mask=word_mask)

        return self.task.logits_to_labels(cat_logits, attr_logits, word_mask)

    def predict(self, sentences, device="cuda"):
        device = torch.device(device)
        num_sentences = len(sentences)

        dataset = self.prepare_sentences(sentences)
        sample = dataset.collater([dataset[i] for i in range(num_sentences)])

        return self.predict_sample(sample)


@register_model_architecture("icebert_pos", "icebert_base_pos")
def roberta_base_architecture(args):
    base_architecture(args)