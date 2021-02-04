from typing import List, Tuple, Optional, Callable

import argparse
import logging
import os
from pathlib import Path

import numpy as np

import torch

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
from greynirseq.nicenlp.data.byte_dictionary import ByteDictionary
from greynirseq.nicenlp.data.icelandic_character_noising import (
    IcelandicCharacterNoising
)
from greynirseq.nicenlp.data.masked_byte_sequence import MaskedByteSequenceDataset
from greynirseq.nicenlp.data.mmapped_text import MmappedTextDataset

logger = logging.getLogger(__name__)


# based on MaskedLMTask from  fairseq
@register_task("byte_masked_lm")
class ByteMaskedLMTask(FairseqTask):
    """Task for training byte-level masked language models (i.e. ByteBERT)."""

    @staticmethod
    def add_args(parser):
        parser.add_argument("data")
        parser.add_argument(
            "--sample-break-mode",
            default="complete",
            choices=["none", "complete", "complete_doc", "eos"],
            help='If omitted or "none", fills each sample with tokens-per-sample '
            'tokens. If set to "complete", splits samples only at the end '
            "of sentence, but may include multiple sentences per sample. "
            '"complete_doc" is similar but respects doc boundaries. '
            'If set to "eos", includes only one sentence per sample.',
        )
        parser.add_argument(
            "--tokens-per-sample",
            default=512,
            type=int,
            help="max number of total tokens over all segments "
            "per sample for ByteBERT dataset",
        )
        parser.add_argument(
            "--shorten-method",
            default="none",
            choices=["none", "truncate", "random_crop"],
            help="if not none, shorten sequences that exceed --tokens-per-sample",
        )
        parser.add_argument(
            "--shorten-data-split-list",
            default="",
            help="comma-separated list of dataset splits to apply shortening to, "
            'e.g., "train,valid" (default: all dataset splits)',
        )
        parser.add_argument(
            "--bpe-dropout",
            default=0.1,
            type=float,
            help="word segmentation sampling parameter (higher means more fragmentation)",
        )
        parser.add_argument(
            "--mask-byte-prob",
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

    def __init__(
        self,
        args: argparse.Namespace,
        byte_dictionary: ByteDictionary,
        bpe_dictionary: Optional[Dictionary] = None,
        word_dictionary: Optional[Dictionary] = None,
        use_word_level: bool = False,
    ):
        super().__init__(args)
        # if bpe_dictionary is None and bpe_dictionary is None:
        #     raise ValueError("bpe_dictionary and word_dictionary cannot both be None")
        self.byte_dictionary = byte_dictionary
        self._bpe_dictionary = bpe_dictionary
        self._word_dictionary = word_dictionary
        self.seed = args.seed
        self.bpe_dropout = args.bpe_dropout
        self.use_word_level = use_word_level

    @classmethod
    def setup_task(cls, args, **kwargs):
        byte_dictionary = ByteDictionary()
        logger.info(f"byte_dictionary: {len(byte_dictionary)} types")
        logger.info(
            "byte_masked_lm: using targets at {} level".format(
                "word" if args.use_word_level else "bpe"
            )
        )
        if args.use_word_level:
            """
            we want to load dict_words.txt and look up each token (according to offsets)
                in the dictionary.
            words not in output embeddings (target-vocab) will therefore become unk by default
                denoting that they cannot be targets
            """
            raise NotImplementedError("use_word_level unimplemented")
        logger.info(f"Using tokens_per_sample: {args.tokens_per_sample}")
        logger.info(f"Using shortening method: {args.shorten_method}")

        bpe_path = Path(args.data) / f"bpe_dict.txt"
        if not bpe_path.is_file():
            raise FileNotFoundError(f"bpe_dict.txt file not found: {bpe_path}")
        # add mask or not?
        bpe_dictionary = Dictionary.load(str(bpe_path))
        assert bpe_dictionary is not None

        # TODO: word_dictionary (for ablation)
        return cls(args, byte_dictionary, bpe_dictionary)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        inputs_path = Path(self.args.data) / f"{split}.txt"
        if not inputs_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {split} ({inputs_path})")

        src_text = MmappedTextDataset(str(inputs_path))

        offsets_path = Path(self.args.data) / f"{split}.offsets.byte"
        if not offsets_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {split} ({offsets_path})")
        offsets = TokenOffsetsDataset(offsets_path)

        seq_dataset = BPEEncoderDataset(
            self.args,
            src_text,
            offsets,
            self.bpe_dictionary,
            seed=self.seed,
            dropout=self.bpe_dropout,
        )  # appends eos, but not bos

        # TODO: should probably implement maybe_shorten_dataset for ByteSequence (see masked_lm)
        # XXX: token_block_dataset of tokens

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        bos_tensor = torch.tensor([self.byte_dictionary.bos()], dtype=torch.long)
        seq_dataset = MapDataset(seq_dataset, fn=lambda x: x.add_byte_prefix(bos_tensor))

        seq_dataset = IcelandicCharacterNoising(
            seq_dataset, case_prob=self.args.case_switch_prob
        )

        seq_dataset = ByteNoising(
            seq_dataset,
            mask_prob=self.args.mask_byte_prob,
            delete_prob=self.args.delete_byte_prob,
            insert_prob=self.args.insert_byte_prob,
            replace_prob=self.args.random_byte_prob,
            transpose_prob=self.args.transpose_bytes_prob,
        )

        seq_dataset = MaskedByteSequenceDataset(
            seq_dataset,
            bpe_dictionary=self.bpe_dictionary,
            word_dictionary=self.word_dictionary,
            mask_prob=self.args.mask_prob,
            byte_mask_index=self.byte_dictionary.byte_mask(),
        )

        with data_utils.numpy_seed(self.seed + epoch):
            shuffle = np.random.permutation(len(seq_dataset))

        byteseq_dataset = MapDataset(seq_dataset, fn=lambda x: x.byte_seq)
        target_dataset = MapDataset(seq_dataset, fn=lambda x: x.targets)
        target_mask_dataset = MapDataset(seq_dataset, fn=lambda x: x.target_mask)
        bpe_mask_dataset = MapDataset(seq_dataset, fn=lambda x: x.bpe_mask)
        word_mask_dataset = MapDataset(seq_dataset, fn=lambda x: x.word_mask)

        pool_lengths = MapDataset(seq_dataset, fn=lambda x: x.bpe_lens)
        if self.args.use_word_level:
            pool_lengths = MapDataset(seq_dataset, fn=lambda x: x.word_lens)

        net_input = {
            "src_tokens": RightPadDataset(
                byteseq_dataset, pad_idx=self.byte_dictionary.pad()
            ),
            "src_lengths": NumelDataset(byteseq_dataset, reduce=False),
            "pool_lengths": RightPadDataset(
                pool_lengths, pad_idx=0
            ),
        }

        # temporary hack
        if self.args.use_word_level:
            net_input["word_mask"] =  RightPadDataset(word_mask_dataset, pad_idx=0)
        else:
            net_input["bpe_mask"] = RightPadDataset(bpe_mask_dataset, pad_idx=0)

        if self.bpe_dictionary.index("<mask>") == self.bpe_dictionary.unk():
            logger.error("BPE dictionary is missing <mask>")


        self.datasets[split] = SortDataset(
            NestedDictionaryDataset(
                {
                    "id": IdDataset(),
                    "net_input": net_input,
                    "target": RightPadDataset(
                        target_dataset, pad_idx=self.target_dictionary.pad()
                    ),
                    "masked_tokens": RightPadDataset(target_mask_dataset, pad_idx=0),
                    "nsentences": NumSamplesDataset(),
                    "ntokens": NumelDataset(byteseq_dataset, reduce=True),
                },
                sizes=[byteseq_dataset.sizes],
            ),
            sort_order=[shuffle, byteseq_dataset.sizes],
        )

    def build_dataset_for_inference(self, src_tokens, src_lengths, sort=True):
        raise NotImplementedError

    @property
    def bpe_dictionary(self):
        return self._bpe_dictionary

    @property
    def word_dictionary(self):
        return self._word_dictionary

    @property
    def source_dictionary(self):
        # relevance/applicability?
        if self.bpe_dictionary is None:
            return self.word_dictionary
        return self.bpe_dictionary

    @property
    def target_dictionary(self):
        return self.source_dictionary
