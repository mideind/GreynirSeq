# flake8: noqa

import itertools
import logging
from collections import namedtuple
from typing import List

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.data import BaseWrapperDataset, NestedDictionaryDataset, NumelDataset, RightPadDataset
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import register_model, register_model_architecture
from fairseq.models.roberta.hub_interface import RobertaHubInterface
from fairseq.models.roberta.model import RobertaEncoder, RobertaModel, base_architecture
from fairseq.modules import LayerNorm
from torch import nn
from torch.nn.utils.rnn import pad_sequence

import greynirseq.nicenlp.utils.constituency.chart_parser as chart_parser  # pylint: disable=no-name-in-module
import greynirseq.nicenlp.utils.constituency.greynir_utils as greynir_utils
from greynirseq.nicenlp.data.datasets import (
    LabelledSpanDataset,
    NestedDictionaryDatasetFix,
    NestedDictionaryDatasetFix2,
    NumSpanDataset,
    ProductSpanDataset,
    WordEndMaskDataset,
)
from greynirseq.nicenlp.models.multilabel import MultiLabelTokenClassificationHead
from greynirseq.nicenlp.models.simple_parser import ChartParserHead
from greynirseq.nicenlp.utils.constituency import token_utils
from greynirseq.nicenlp.utils.constituency.greynir_utils import Node
from greynirseq.nicenlp.utils.label_schema.label_schema import make_vec_idx_to_dict_idx

logger = logging.getLogger(__name__)


MultiParserOutput = namedtuple("MultiParserOutput", ["greynir_parser", "greynir_pos"])


@register_model("icebert_multiparser")
class MultiParserModel(RobertaModel):
    """Simple chart parser model based on Roberta. Simple in the sense that it does not
    add additional self-attentive layers or feature-engineering from Kitaev and Klein 2018."""

    def __init__(self, args, encoder, task):
        super().__init__(args, encoder)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        sentence_encoder = self.encoder.sentence_encoder

        layer = sentence_encoder.build_encoder_layer(args)
        if args.parser_layers > 0:
            self.parser_layers = nn.ModuleList(
                [sentence_encoder.build_encoder_layer(args) for _ in range(args.parser_layers)]
            )
        else:
            self.parser_layers = None

        # self.pre_parser_layer = sentence_encoder.build_encoder_layer(args)
        # self.pre_pos_layer = sentence_encoder.build_encoder_layer(args)

        # Not freezing embeddings degrades result according to multiple papers (e.g. Kitaev)
        freeze_module_params(sentence_encoder.embed_tokens)
        freeze_module_params(sentence_encoder.embed_positions)
        freeze_module_params(sentence_encoder.layernorm_embedding)

        for layer in range(args.n_trans_layers_to_freeze):
            freeze_module_params(sentence_encoder.layers[layer])

        _num_embeddings, embed_dim = sentence_encoder.embed_tokens.weight.shape
        self.task = task

        greynir_parser_head = ChartParserHead(
            embed_dim,
            self.task.num_nterm_cats,
            self.args.pooler_dropout,
        )

        greynir_pos_head = MultiLabelTokenClassificationHead(
            in_features=embed_dim,
            out_features=self.task.num_term_labels,
            num_cats=self.task.num_term_cats,
            pooler_dropout=self.args.pooler_dropout,
        )

        # self.task_head = greynir_parser_head
        self.task_heads = nn.ModuleDict(
            {
                "greynir_parser": greynir_parser_head,
                "greynir_pos": greynir_pos_head,
            }
        )

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        super(MultiParserModel, MultiParserModel).add_args(parser)
        # fmt: off
        parser.add_argument('--freeze-embeddings', default=False,
                            help='Freeze transformer embeddings during fine-tuning')
        parser.add_argument('--n-trans-layers-to-freeze', default=0, type=int,
                            help='Number of transformer layers to freeze during fine-tuning')
        parser.add_argument('--parser-layers', default=2, type=int,
                            help='Number of transformer layers to stack on-top of original encoder for parsing')
        # fmt: on

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        roberta = RobertaModel.build_model(args, task)
        encoder = roberta.encoder
        return cls(args, encoder, task)

    def forward(self, src_tokens, features_only=False, return_all_hiddens=False, **kwargs):
        x, _extra = self.encoder(src_tokens, features_only, return_all_hiddens=return_all_hiddens, **kwargs)
        _, _, nchannels = x.shape
        mask = kwargs["word_mask_w_bos"]
        mask_no_bos = mask.clone()
        mask_no_bos[:, 0] = 0  # exclude bos in mask
        words_w_bos = x.masked_select(mask.unsqueeze(-1).bool()).reshape(-1, nchannels)
        words_without_bos = x.masked_select(mask_no_bos.unsqueeze(-1).bool()).reshape(-1, nchannels)
        nwords_w_bos = kwargs["word_mask_w_bos"].sum(-1)
        words_w_bos_padded = pad_sequence(
            words_w_bos.split((nwords_w_bos).tolist()),
            padding_value=0,
            batch_first=True,
        )
        word_padding_mask = lengths_to_padding_mask(nwords_w_bos)
        wpm = word_padding_mask

        # B x T x C -> T x B x C
        x = words_w_bos_padded.transpose(0, 1)
        if self.parser_layers:
            for layer in self.parser_layers:
                x = layer(x, word_padding_mask)
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        words_without_bos_padded = pad_sequence(
            words_without_bos.split((nwords_w_bos - 1).tolist()),
            padding_value=0,
            batch_first=True,
        )

        # greynir_span_features = self.task_heads["greynir_parser"](x)
        # greynir_pos_features = self.task_heads["greynir_pos"](words_without_bos_padded)

        greynir_span_features = self.task_heads["greynir_parser"](
            self.parser_layers[0](words_w_bos_padded.transpose(0, 1), lengths_to_padding_mask(nwords_w_bos)).transpose(
                0, 1
            )
        )
        greynir_pos_features = self.task_heads["greynir_pos"](
            self.parser_layers[1](
                words_without_bos_padded.transpose(0, 1), lengths_to_padding_mask(nwords_w_bos - 1)
            ).transpose(0, 1)
        )

        output = MultiParserOutput(
            greynir_parser=greynir_span_features,
            greynir_pos=greynir_pos_features,
        )
        return output

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)

        prefix = name + "." if name != "" else ""

        for key, value in self.task_heads.state_dict().items():
            path = prefix + "task_heads." + key
            logger.info("Initializing task_heads." + key)
            if path not in state_dict:
                state_dict[path] = value

        if self.parser_layers:
            for key, value in self.parser_layers.state_dict().items():
                path = prefix + "parser_layers." + key
                logger.info("Initializing self-attentive parser: parser_layers." + key)
                if path not in state_dict:
                    state_dict[path] = value

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
        return MultiParserHubInterface(x["args"], x["task"], x["models"][0])


class MultiParserHubInterface(RobertaHubInterface):
    def predict_sample(self, sample):
        nwords = sample["nwords"]
        net_input = sample["net_input"]
        tokens = net_input["src_tokens"]

        span_features = self.model(**sample["net_input"], features_only=True)

        _tree_scores, lspans, _mask, _lmask = chart_parser.parse_many(span_features.cpu().detach(), nwords.cpu())

        cat_vec_idx_to_dict_idx = make_vec_idx_to_dict_idx(self.task.nterm_dictionary, self.task.nterm_schema.labels)
        span_labels = [
            self.task.nterm_dictionary.string(cat_vec_idx_to_dict_idx[seq_lspans[:, 2].long()]).split(" ")
            for seq_lspans in lspans
        ]
        spans = [seq_lspans[:, :2] for seq_lspans in lspans]
        src_tokens_unpadded = tokens[tokens != 1].split(sample["net_input"]["nsrc_tokens"].tolist())
        sentences_in_tokens = [self.decode(seq_tokens).strip().split(" ") for seq_tokens in src_tokens_unpadded]
        pred_trees = [
            Node.from_labelled_spans(seq_spans, seq_span_labels, sentence_tokens).debinarize()
            for seq_spans, seq_span_labels, sentence_tokens in zip(spans, span_labels, sentences_in_tokens)
        ]

        return pred_trees, (lspans, span_labels, _lmask)

    def predict(self, sentences, device="cuda"):
        device = torch.device(device)
        num_sentences = len(sentences)

        dataset = self.prepare_sentences(sentences)
        sample = dataset.collater([dataset[i] for i in range(num_sentences)])

        return self.predict_sample(sample)

    def prepare_sentences(self, sentences: List[str]):
        tokens = [
            self.encode(token_utils.tokenize_to_string(sentence, add_prefix_space=True)) for sentence in sentences
        ]
        return self.task.prepare_tokens(tokens)


@register_model_architecture("icebert_multiparser", "icebert_multiparser")
def roberta_base_architecture(args):
    args.parser_layers = getattr(args, "parser_layers", 2)
    base_architecture(args)
