from typing import List, Tuple, Optional, Callable

import argparse
import logging
import os
from pathlib import Path

import numpy as np

import torch

from torch.utils.data.dataset import Dataset

from fairseq import utils
from fairseq.data import (
    Dictionary,
    IdDataset,
    ListDataset,
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
from greynirseq.nicenlp.data.byte_token_blocks import ByteTokenBlockDataset
from greynirseq.nicenlp.byte_sequence import ByteSequence

from greynirseq.nicenlp.data import NestedDictionaryDatasetFix
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
            byte_dictionary=self.byte_dictionary,
        )  # appends eos, but not bos

        # TODO: should probably implement maybe_shorten_dataset for ByteSequence (see masked_lm)
        # XXX: token_block_dataset of tokens

        # prepend beginning-of-sentence token (<s>, equiv. to [CLS] in BERT)
        byte_bos_tensor = torch.tensor([self.byte_dictionary.bos()], dtype=torch.long)
        bpe_bos_tensor = torch.tensor([self.bpe_dictionary.bos()], dtype=torch.long)

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
            seed=self.seed,
        )

        logger.info(f"Using sample-break-mode: {self.args.sample_break_mode}")
        seq_dataset = ByteTokenBlockDataset(
            seq_dataset,
            seq_dataset.sizes,
            self.args.tokens_per_sample,
            self.byte_dictionary.pad(),
            self.byte_dictionary.eos(),
            self.args.sample_break_mode,
        )

        seq_dataset = MapDataset(seq_dataset, fn=lambda x: x.add_byte_prefix(byte_bos_tensor, bpe_bos_tensor))

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
    def prepare_sentence_for_fill_masks(self, sentence):
        parts = sentence.split("<mask>")
        assert len(parts) > 1

        empty_parts = [idx for idx, part in enumerate(parts) if not part]
        text_dataset = ListDataset(
            parts,
            sizes=torch.tensor([len(part.encode()) for part in parts]),
        )
        offsets = [calculate_byte_offsets(part) for part in parts]
        offsets = ListDataset(
            [torch.tensor(part) for part in offsets],
            sizes=torch.tensor([len(part) for part in offsets])
        )
        dataset = BPEEncoderDataset(
            self.args,
            text_dataset,
            offsets,
            self.bpe_dictionary,
            self.byte_dictionary,
            seed=self.seed,
            # dropout=0.0,
            dropout=0.15,
            append_eos=False,
        )

        mask_byte_seq = ByteSequence(
            "<mask>",
            torch.tensor([self.byte_dictionary.byte_mask()]).long(),
            bpe_ids=torch.tensor([self.bpe_dictionary.index("<mask>")]).long(),
            bpe_mask=torch.tensor([True]).bool(),
            bpe_lens=torch.tensor([1]).long(),
            word_mask=torch.tensor([True]).bool(),
            word_lens=torch.tensor([1]).long(),
        )
        bos_byte_seq = ByteSequence(
            "<bos>",
            torch.tensor([self.byte_dictionary.bos()]).long(),
            bpe_ids=torch.tensor([self.bpe_dictionary.bos()]).long(),
            bpe_mask=torch.tensor([True]).bool(),
            bpe_lens=torch.tensor([1]).long(),
            word_mask=torch.tensor([True]).bool(),
            word_lens=torch.tensor([1]).long(),
        )
        eos_byte_seq = ByteSequence(
            "<eos>",
            torch.tensor([self.byte_dictionary.eos()]).long(),
            bpe_ids=torch.tensor([self.bpe_dictionary.eos()]).long(),
            bpe_mask=torch.tensor([True]).bool(),
            bpe_lens=torch.tensor([1]).long(),
            word_mask=torch.tensor([True]).bool(),
            word_lens=torch.tensor([1]).long(),
        )
        part_seqs = [bos_byte_seq]
        new_parts = []
        # This reconstructs the pattern of the input before the split function is called, masks will replace
        # the removed part.
        # Only the prefix and suffix have the correct amount of placeholders for masks, see:
        #     ab.split(a) is [-,b]
        #     aab.split(a) is [-,-,b]
        #     bab.split(a) is [b,b]
        #     baab.split(a) is [b,-,b]
        #     ba.split(a) is [b,-]
        #     baa.split(a) is [b,-,-]
        # we therefore need insert missing placeholders (ex. 3 and 4)
        # when a dash appears after b's there is an omitted 'a'
        last_was_dash = False  # or part of prefix/suffix of a's
        for idx in range(len(parts)):
            if parts[idx]:
                if not last_was_dash and idx > 0:
                    # ex: second 'b' in bab, [b,b]
                    new_parts.append(mask_byte_seq.clone())
                # ex: second 'b' in baab, [b,-,b]
                new_parts.append(dataset[idx])
                last_was_dash = False
            elif last_was_dash and (1 + idx < len(parts)):
                # ex: second 'a' in aab or any dash in [-,-,b]
                new_parts.append(mask_byte_seq.clone())
                last_was_dash = True
            elif 1 + idx < len(parts):
                # ex: first two 'a' in baaab, or first dash in [b,-,-,b]
                # this is the first of a sequence, counts double
                new_parts.append(mask_byte_seq.clone())
                new_parts.append(mask_byte_seq.clone())
                last_was_dash = True
            else:
                # ex: first a in ba, [b,-]
                # we are at last_place in sequence and it does not have
                # continuation
                new_parts.append(mask_byte_seq.clone())
        part_seqs.extend(new_parts)
        merged = ByteSequence.cat(part_seqs, str_sep="")
        merged_dataset = ListDataset(
            [merged],
            sizes=[len(merged.byte_seq)]
        )
        return self.prepare_tokens(merged_dataset)

    def prepare_sentences(self, sentences: List[str], offsets: List[int], dropout: float=0.0):
        text_dataset = ListDataset(
            sentences,
            sizes=torch.tensor([len(sentence.encode()) for sentence in sentences]),
        )
        offsets = ListDataset(
            [torch.tensor(sent_offsets) for sent_offsets in offsets],
            sizes=torch.tensor([len(sent) for sent in offsets])
        )
        byte_sequences = BPEEncoderDataset(
            self.args,
            text_dataset,
            offsets,
            self.bpe_dictionary,
            self.byte_dictionary,
            seed=self.seed,
            dropout=dropout,
            append_eos=True,
        )

        byte_bos_tensor = torch.tensor([self.byte_dictionary.bos()], dtype=torch.long)
        bpe_bos_tensor = torch.tensor([self.bpe_dictionary.bos()], dtype=torch.long)
        byte_sequences = MapDataset(byte_sequences, fn=lambda x: x.add_byte_prefix(byte_bos_tensor, bpe_bos_tensor))
        return self.prepare_tokens(byte_sequences)
    def decode(self, targets, remove_fairseq_special=False):
        # TODO: refactor?
        if not hasattr(self, "hf_tokenizer"):
            import tokenizers

            self.hf_tokenizer = tokenizers.ByteLevelBPETokenizer(
                self.args.gpt2_encoder_json,
                self.args.gpt2_vocab_bpe,
                add_prefix_space=True,
            )

        if remove_fairseq_special:
            # since we are using gpt2tokenizer approach there are no spaces in tokens in bpe_dictionary
            parts = self.bpe_dictionary.string(targets).split(" ")
        else:
            parts = [self.bpe_dictionary[t] for t in targets]
        new_parts = [self.hf_tokenizer.decode([int(t)]) if t.isnumeric() else t for t in parts]
        return "".join(new_parts)

    def prepare_tokens(self, seq_dataset: Dataset):
        byteseq_dataset = MapDataset(seq_dataset, fn=lambda x: x.byte_seq)
        target_dataset = MapDataset(seq_dataset, fn=lambda x: x.bpe_ids)
        # target_mask_dataset = MapDataset(seq_dataset, fn=lambda x: x.target_mask)
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

        dataset = NestedDictionaryDatasetFix(
            {
                "id": IdDataset(),
                "net_input": net_input,
                "target": RightPadDataset(
                    target_dataset, pad_idx=self.target_dictionary.pad()
                ),
                # "masked_tokens": RightPadDataset(target_mask_dataset, pad_idx=0),
                "nsentences": NumSamplesDataset(),
                "ntokens": NumelDataset(byteseq_dataset, reduce=True),
            },
            sizes=[byteseq_dataset.sizes],
        )

        return dataset


def byte_len(string):
    return len(bytes(string, encoding="utf-8"))


def calculate_byte_offsets(line: str, inclusive_end: bool=True) -> List[int]:
    """ Calculate token offsets for a single line """
    import tokenizer
    if not line.strip("\n"):
        return []
    elif not line.strip():
        return []  # what to do with " " as input?
    byte_offsets = [0]
    last_end = 0
    tokens_ = list(tokenizer.parse_tokens(line.strip("\n")))
    tokens = []
    byte_lens = []
    for idx, token in enumerate(tokens_):
        if token.txt is None:
            assert token._original is None
            assert token._origin_spans is None
            assert token.kind == 12001
            continue
        offset = line.find(token._original)
        tok_byte_len = byte_len(token._original)
        if idx == 0:
            # leading space is not part of first token from tokenizer.parse_tokens
            len_leading = byte_len(line[:offset])
            tok_byte_len += len_leading
        last_end += tok_byte_len
        byte_offsets.append(last_end)
        byte_lens.append(byte_len(token._original))
    if inclusive_end:
        return byte_offsets[:-1]
    return byte_offsets
