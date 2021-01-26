# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path

import numpy as np
from fairseq import utils
from fairseq.data import (
    Dictionary,
    IdDataset,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    PrependTokenDataset,
    RightPadDataset,
    SortDataset,
    TokenBlockDataset,
    data_utils,
)
from fairseq.data.shorten_dataset import maybe_shorten_dataset
from fairseq.tasks import register_task, FairseqTask

from greynirseq.nicenlp.data.bpe_encoder_dataset import BPEEncoderDataset
from greynirseq.nicenlp.data.token_offsets_dataset import TokenOffsetsDataset
from greynirseq.nicenlp.data.map_dataset import MapDataset
from greynirseq.nicenlp.data.byte_noising import ByteNoising
from greynirseq.nicenlp.data.icelandic_character_noising import IcelandicCharacterNoising
from greynirseq.nicenlp.data.masked_byte_sequence import MaskedByteSequenceDataset
from greynirseq.nicenlp.data.mmapped_text import MmappedTextDataset

logger = logging.getLogger(__name__)


# based on MaskedLMTask from  fairseq
@register_task("byte_masked_lm")
class ByteMaskedLMTask(FairseqTask):
    """Task for training byte-level masked language models (i.e. ByteBERT)."""

    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "data",
        )
        # parser.add_argument(
        #     "--sample-break-mode",
        #     default="complete",
        #     choices=["none", "complete", "complete_doc", "eos"],
        #     help='If omitted or "none", fills each sample with tokens-per-sample '
        #     'tokens. If set to "complete", splits samples only at the end '
        #     "of sentence, but may include multiple sentences per sample. "
        #     '"complete_doc" is similar but respects doc boundaries. '
        #     'If set to "eos", includes only one sentence per sample.',
        # )
        parser.add_argument(
            "--tokens-per-sample",
            default=512,
            type=int,
            help="max number of total tokens over all segments "
            "per sample for ByteBERT dataset",
        )
        # parser.add_argument(
        #     "--shorten-method",
        #     default="none",
        #     choices=["none", "truncate", "random_crop"],
        #     help="if not none, shorten sequences that exceed --tokens-per-sample",
        # )
        # parser.add_argument(
        #     "--shorten-data-split-list",
        #     default="",
        #     help="comma-separated list of dataset splits to apply shortening to, "
        #     'e.g., "train,valid" (default: all dataset splits)',
        # )
        parser.add_argument(
            "--bpe-dropout",
            default=0.1,
            type=float,
            help="word segmentation sampling parameter (higher means more fragmentation)",
        )
        parser.add_argument(
            "--byte-mask-prob",
            default=0.02,
            type=float,
            help="probability of replacing a byte with mask",
        )
        parser.add_argument(
            "--delete-byte-prob",
            default=0.02,
            type=float,
            help="probability of deleting a given byte",
        )
        parser.add_argument(
            "--insert-byte-prob",
            default=0.02,
            type=float,
            help="probability that a random byte is inserted at a given position",
        )
        parser.add_argument(
            "--random-byte-prob",
            default=0.02,
            type=float,
            help="probability of replacing a byte with a random byte",
        )
        parser.add_argument(
            "--transpose-bytes-prob",
            default=0.02,
            type=float,
            help="probability that a given byte has been transposed with a neighbour",
        )
        parser.add_argument(
            "--case-switch-prob",
            default=0.02,
            type=float,
            help="probability that a given character is swapped between upper/lower case",
        )
        parser.add_argument(
            "--mask-prob",
            default=0.15,
            type=float,
            help="probability that a given word/bpe token is masked",
        )
        parser.add_argument(
            "--use-word-level",
            action="store_true",
            help="use targets at the word level instead of bpe level (requires precomputed dict_words.txt)",
        )

    def __init__(self, args: argparse.Namespace, byte_dictionary: ByteDictionary, bpe_dictionary: Optional[Dictionary]=None, word_dictionary: Optional[Dictionary]=None):
        super().__init__(args)
        if (bpe_mask_prob is None or bpe_mask_prob <= 0) and (word_mask_prob is None or word_mask_prob <= 0):
            raise ValueError("bpe_dictionary and word_dictionary cannot both be None")
        self.byte_dictionary = byte_dictionary
        self.bpe_dictionary = bpe_dictionary
        self.word_dictionary = word_dictionary
        self.seed = args.seed
        self.bpe_dropout = args.bpe_dropout

    @classmethod
    def setup_task(cls, args, **kwargs):
        logger.info("byte_dictionary: {} types".format(len(self.byte_dictionary)))
        logger.info("byte_masked_lm: using targets at {} level".format("word" if self.args.use_word_level else "bpe"))
        if self.args.use_word_level:
            """
            we want to load dict_words.txt and look up each token (according to offsets)
                in the dictionary.
            words not in output embeddings (target-vocab) will therefore become unk by default
                denoting that they cannot be targets
            """
            raise NotImplementedError("use_word_level unimplemented")
        logger.info("Using tokens_per_sample: {args.tokens_per_sample}")
        logger.info("Using shortening method: {args.shorten_method}")

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        inputs_path = Path(self.args.data) / f"{split}"
        if offsets_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {split} ({inputs_path})")

        src_text = MmappedTextDataset(
            str(inputs_path),
        )

        offsets_path = Path(self.args.data) / f"{split}.offsets.txt"
        if offsets_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {split} ({offsets_path})")
        offsets = TokenOffsetsDataset(offsets_path)

        dataset = BPEEncoderDataset(
            self.args,
            src_tokens,
            offsets,
            self.dictionary
            seed=self.seed,
            dropout=self.bpe_dropout,
        )  # appends eos, but not bos

        # TODO: should probably implement maybe_shorten_dataset for ByteSequence (see masked_lm)
        # TODO: token_block_dataset of tokens
        # logger.info("loaded {} blocks from: {}".format(len(dataset), split_path))

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        dataset = MapDataset(
            src_tokens,
            fn=lambda x: x.add_prefix(self.byte_dictionary.bos_idx),
        )

        dataset = IcelandicCharacterNoising(
            dataset,
            case_prob=self.args.case_switch_prob,
        )

        dataset = ByteNoising(
            dataset,
            mask_prob = self.args.mask_prob,
            delete_prob = self.args.delete_prob,
            insert_prob = self.args.insert_prob,
            replace_prob = self.args.replace_prob,
            transpose_prob = self.args.transpose_prob,
        )

        dataset = MaskedByteSequenceDataset(
            dataset,
            bpe_dictionary=self.bpe_dictionary,
            word_dictionary=self.word_dictionary,
            bpe_mask_prob=self.args.bpe_mask_prob,
            word_mask_prob=self.args.word_mask_prob,
        )

        with data_utils.numpy_seed(self.seed + epoch):
            shuffle = np.random.permutation(len(dataset))

        src_tokens = MapDataset(dataset, fn=lambda x: x.byte_seq)
        targets = MapDataset(dataset, fn=lambda x: x.targets)
        target_mask = MapDataset(dataset, fn=lambda x: x.target_mask)

        contraction_lengths = MapDataset(dataset, fn=lambda x: x.bpe_mask)
        if self.args.use_word_level:
            contraction_lengths = MapDataset(dataset, fn=lambda x: x.word_mask)

        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    "id": IdDataset(),
                    "net_input": {
                        "src_tokens": RightPadDataset(
                            src_tokens,
                            pad_idx=self.byte_dictionary.pad(),
                        ),
                        "src_lengths": NumelDataset(byte_seq, reduce=False),
                        "contraction_lengths": RightPadDataset(
                            contraction_lengths,
                            pad_idx=0,
                        ),
                    },
                    "target": RightPadDataset(
                        tgt_dataset,
                        pad_idx=self.source_dictionary.pad(),
                    ),
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(src_dataset, reduce=True),
                },
                sizes=[src_dataset.sizes],
            ),
            sort_order=[
                shuffle,
                src_dataset.sizes,
            ],
        )
        if sort:
            src_dataset = SortDataset(src_dataset, sort_order=[src_lengths])
        return src_dataset

    # def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
    #     pass

    @property
    def bpe_dictionary(self):
        return self._bpe_dictionary
        pass

    @property
    def word_dictionary(self):
        return self._word_dictionary

    @property
    def source_dictionary(self):
        # relevance/applicability?
        # return self.dictionary
        pass

    @property
    def target_dictionary(self):
        # relevance/applicability?
        # return self.dictionary
        pass
