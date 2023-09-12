# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import datasets as hf_datasets
import numpy as np
import pyarrow as pa
import torch
from datasets import Dataset as HFDataset
from datasets import Sequence, Value
from fairseq.data import Dictionary, LanguagePairDataset, data_utils

from greynirseq.nicenlp.data.encoders import Encoder

logger = logging.getLogger(__name__)

_POISSON_MEAN = 1.5
_DOCUMENT_JSONL_FEATURE_DICT = {
    "document": Sequence(
        feature=Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
        length=-1,
        id=None,
    ),
    "domains": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
    "lang": Value(dtype="string", id=None),
    "title": Value(dtype="string", id=None),
    "uuid": Value(dtype="string", id=None),
}

# Types for alignment data
SegmentAlignment = Tuple[int, int]  # pg_idx, sent_idx
SrcAlignment = List[SegmentAlignment]
TgtAlignment = List[SegmentAlignment]
PairAlignment = Tuple[SrcAlignment, TgtAlignment]
DocumentAlignment = List[PairAlignment]
# An alignment json object
_ALIGNMENTS_JSONL_FEATURE_DICT = {
    "langs": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
    "alignments": Sequence(
        feature=Sequence(
            feature=Sequence(
                feature=Sequence(feature=Value(dtype="int64", id=None), length=-1, id=None),
                length=-1,
                id=None,
            ),
            length=-1,
            id=None,
        ),
        length=-1,
        id=None,
    ),
    "uuid": Value(dtype="string", id=None),
}

TMP_DICT = {
    "document.sentence_weights": Sequence(
        feature=Sequence(feature=Value(dtype="int64", id=None), length=-1, id=None),
        length=-1,
        id=None,
    ),
}


def lengths_to_offsets(lengths: np.ndarray) -> np.ndarray:
    """Convert a 1D array of lengths to offsets."""
    return np.cumsum(lengths) - lengths


class KEYS:
    DOCUMENT = "document"
    SENTENCE_WEIGHTS = "document.sentence_weights"
    DOCUMENT_IDS = "document.ids"
    DOCUMENT_WEIGHT = "document.weight"
    ALIGNMENTS = "alignments"
    DOCUMENT_INDEX = "document_index"
    UUID = "uuid"
    LANG1 = "1"
    LANG2 = "2"
    LANGS = "langs"
    LANG = "lang"
    PARAGRAPH_INDEX = "paragraph_index"
    SEGMENT_INDEX = "segment_index"
    SEGMENT_WEIGHT = "segment_weight"
    SEGMENT = "segment"
    LENGTH = "length"
    INDICES = "indices"
    NUM_SEGMENTS = "num_segments"
    DOCUMENT_SEGMENT_OFFSETS = "document_segment_offsets"
    SOURCE_INDICES = "source_indices"
    TARGET_INDICES = "target_indices"
    WEIGHT = "weight"
    SKIP = "skip"
    SOURCE_OFFSETS = "source_offsets"
    TARGET_OFFSETS = "target_offsets"
    SOURCE_WEIGHTS = "source_weights"
    TARGET_WEIGHTS = "target_weights"
    EXACT_ALIGNMENT = "exact_alignment"


@dataclass
class IndexedParallelFingerprints:
    source: str
    target: str
    align: str

    @classmethod
    def make_fingerprints(cls, src_path: str, tgt_path: str, align_path: Optional[str], version):
        from datasets.fingerprint import Hasher

        # there is no randomness that goes into flattening so we don't need seed here
        prefix = [src_path, tgt_path, align_path, version]
        align_hash = Hasher.hash(prefix + ["flat_align"])
        src_hash = Hasher.hash(prefix + ["flat_src"])
        tgt_hash = Hasher.hash(prefix + ["flat_tgt"])
        return cls(source=src_hash, target=tgt_hash, align=align_hash)


class IndexedParallelDocumentsDataset(LanguagePairDataset):
    version = "3.0"

    def __init__(
        self,
        name: str,
        is_bt: bool,
        flat_align: HFDataset,
        flat_src: HFDataset,
        flat_tgt: HFDataset,
        dictionary: Dictionary,
        encoder: Encoder,
        fingerprints: IndexedParallelFingerprints,
        max_seq_len: int,
        seed: int = 1,
        max_merges: int = 10,
    ):
        super().__init__(None, 0, dictionary)
        self.name = name
        self.is_bt = is_bt
        self.flat_align = flat_align
        self.flat_src = flat_src
        self.flat_tgt = flat_tgt
        self.max_seq_len = max_seq_len
        self.max_merges = max_merges
        self.index_dataset = None  # gets set beginning of each bepoch
        self.dictionary = dictionary
        self.seed = seed
        self.encoder = encoder
        # this is compatibility with LanguagePairDataset collater and its teacher forcing adjustments
        self.src_dict = self.dictionary
        self.left_pad_source = False  # expected by bart model
        self.left_pad_target = False  # expected by bart model
        self.src_lang_id = (
            None  # fairseq 0.10.2 accesses these in LanguagePairDataset.collater (so this attribute must exist)
        )
        self.tgt_lang_id = None

        self._interleave_epoch_index = None
        self.src_sizes = self.sizes
        self.tgt_sizes = None
        self._sorted_indices = None
        self._sorted_lengths = None
        self._bpe = None
        self.epoch = 1
        self.fingerprints = fingerprints
        # we count the skipped examples here
        self.num_skipped = sum(self.flat_align[KEYS.SKIP])

    def __len__(self):
        return len(self.index_dataset)

    def size(self, index):
        return self.index_dataset[int(index)][KEYS.LENGTH]

    def __getitem__(self, index):
        item = self.index_dataset[int(index)]
        maybe_noised_encode_fn = self.encoder.encode_noisy if self.is_bt else self.encoder.encode
        src_segments: List[str] = [self.flat_src[int(i)]["segment"] for i in item[KEYS.SOURCE_INDICES]]
        tgt_segments: List[str] = [self.flat_tgt[int(i)]["segment"] for i in item[KEYS.TARGET_INDICES]]
        src_langs: List[str] = [self.flat_src[int(i)]["lang"] for i in item[KEYS.SOURCE_INDICES]]
        tgt_langs: List[str] = [self.flat_tgt[int(i)]["lang"] for i in item[KEYS.TARGET_INDICES]]
        assert len(set(src_langs)) == 1, "source segments must be from the same language"
        assert len(set(tgt_langs)) == 1, "target segments must be from the same language"

        with data_utils.numpy_seed(self.seed, self.epoch, index):
            insert_sep = np.random.randint(2, dtype=bool)

            assert KEYS.EXACT_ALIGNMENT in item or not insert_sep  # insert_sep implies exact_alignment
            if insert_sep and len(src_segments) > 1 and np.all(item[KEYS.EXACT_ALIGNMENT]):
                # only insert separator when alignment is *exact*
                bos = torch.tensor([self.dictionary.bos()])
                src_out = [bos] * (len(src_segments) * 2 - 1)
                src_out[0::2] = [maybe_noised_encode_fn(seg) for seg in src_segments]
                tgt_out = [bos] * (len(tgt_segments) * 2 - 1)
                tgt_out[0::2] = [self.encoder.encode(seg) for seg in tgt_segments]
            else:
                src_out = [maybe_noised_encode_fn(seg) for seg in src_segments]
                tgt_out = [self.encoder.encode(seg) for seg in tgt_segments]

        # This language code handling is like the mBart-50 model and nllb-200
        src_out = torch.cat(
            [torch.tensor([self.dictionary.index(src_langs[0])])] + src_out + [torch.tensor([self.dictionary.eos()])]
        )
        tgt_out = torch.cat(
            [torch.tensor([self.dictionary.index(tgt_langs[0])])] + tgt_out + [torch.tensor([self.dictionary.eos()])]
        )

        if len(src_out) > 1020 or len(tgt_out) > 1020:
            print(f"Source: {self.encoder.bpe.decode(self.src_dict.string(src_out))}")
            print(f"Target: {self.encoder.bpe.decode(self.src_dict.string(tgt_out))}")
            assert False

        return {"id": index, "source": src_out, "target": tgt_out}

    def _decode_seq(self, seq):
        return self.encoder.bpe.decode(self.src_dict.string(seq))

    def cache_to_disk(self):
        if self.fingerprints is None:
            raise ValueError("Cannot save without fingerprints")
        cache_dir = hf_datasets.config.HF_DATASETS_CACHE  # type: ignore
        self.flat_src.save_to_disk(f"{cache_dir}/{self.fingerprints.source}")
        self.flat_tgt.save_to_disk(f"{cache_dir}/{self.fingerprints.target}")
        self.flat_align.save_to_disk(f"{cache_dir}/{self.fingerprints.align}")

    @classmethod
    def load_from_cache(
        cls,
        name: str,
        is_bt: bool,
        src_path: str,
        tgt_path: str,
        dictionary: Dictionary,
        encoder: Encoder,
        max_seq_len: int,
        max_merges: int = 10,
        align_path: Optional[str] = None,
        seed: int = 1,
    ) -> Optional["IndexedParallelDocumentsDataset"]:
        cache_dir = hf_datasets.config.HF_DATASETS_CACHE  # type: ignore

        fp = IndexedParallelFingerprints.make_fingerprints(src_path, tgt_path, align_path, cls.version)
        if not all(Path(f"{cache_dir}/{val}").exists() for val in [fp.source, fp.target, fp.align]):
            return None
        for val in [fp.source, fp.target, fp.align]:
            logger.info(f"Loading {cache_dir}/{val}")
        flat_src = hf_datasets.load_from_disk(f"{cache_dir}/{fp.source}")
        flat_tgt = hf_datasets.load_from_disk(f"{cache_dir}/{fp.target}")
        flat_align = hf_datasets.load_from_disk(f"{cache_dir}/{fp.align}")
        assert isinstance(flat_src, hf_datasets.arrow_dataset.Dataset)
        assert isinstance(flat_tgt, hf_datasets.arrow_dataset.Dataset)
        assert isinstance(flat_align, hf_datasets.arrow_dataset.Dataset)

        return cls(
            name,
            is_bt,
            flat_align,
            flat_src,
            flat_tgt,
            dictionary,
            encoder=encoder,
            seed=seed,
            max_seq_len=max_seq_len,
            max_merges=max_merges,
            fingerprints=fp,
        )

    @classmethod
    def from_parallel_jsonl(
        cls,
        name: str,
        is_bt: bool,
        src_path: str,
        tgt_path: str,
        bpe_encoder,
        dictionary: Dictionary,
        encoder: Encoder,
        max_seq_len: int,
        data_language_mapper: Dict[str, str],
        max_merges: int = 10,
        load_from_cache_file: bool = True,
        num_proc: int = 8,
        align_path: Optional[str] = None,
        seed: int = 1,
    ):
        if load_from_cache_file:
            cached_dataset = cls.load_from_cache(
                name,
                is_bt,
                src_path,
                tgt_path,
                dictionary,
                encoder=encoder,
                max_seq_len=max_seq_len,
                max_merges=max_merges,
                align_path=align_path,
                seed=seed,
            )
            if cached_dataset is not None:
                logger.info(f"Found matching cached dataset for {src_path} and {tgt_path}")
                return cached_dataset

        features = hf_datasets.Features(_DOCUMENT_JSONL_FEATURE_DICT)
        logger.info(f"Loading src_dataset: {src_path}")
        src_dataset = hf_datasets.Dataset.from_json(src_path, split="train", chunksize=40 << 20, features=features)
        logger.info(f"Loading tgt_dataset: {tgt_path}")
        tgt_dataset = hf_datasets.Dataset.from_json(tgt_path, split="train", chunksize=40 << 20, features=features)
        assert isinstance(src_dataset, hf_datasets.Dataset), f"src_dataset is {type(src_dataset)}"
        assert isinstance(tgt_dataset, hf_datasets.Dataset), f"tgt_dataset is {type(tgt_dataset)}"

        src_lang = src_dataset[0][KEYS.LANG]
        tgt_lang = tgt_dataset[0][KEYS.LANG]

        if not any("valid" in p for p in src_path):
            pass

        flip_alignment = False
        align_dataset = None
        if align_path is not None:
            logger.info(f"Loading alignments: {align_path}")
            align_dataset = hf_datasets.Dataset.from_json(
                align_path,
                split="train",  # type: ignore
                chunksize=40 << 20,
                features=hf_datasets.Features(_ALIGNMENTS_JSONL_FEATURE_DICT),
            )
            assert isinstance(align_dataset, hf_datasets.Dataset), f"align_dataset is {type(align_dataset)}"
            assert set(align_dataset[0][KEYS.LANGS]) == set([src_lang, tgt_lang])
            if align_dataset[0][KEYS.LANGS] != [src_lang, tgt_lang]:
                flip_alignment = True
                logger.info(f"Flipping alignments: {align_path}")
        else:
            align_dataset = make_align_dataset_default_alignments(src_dataset, tgt_dataset)

        flat_align = compute_offsets_and_flatten_alignments(
            src_dataset,
            tgt_dataset,
            align_dataset,
            bpe_encoder,
            dictionary,
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
            flip_alignment=flip_alignment,
            max_sequence_length=max_seq_len,
        )
        flat_src = flatten_document_dataset(
            src_dataset,
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
        )
        flat_tgt = flatten_document_dataset(
            tgt_dataset,
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
        )

        flat_src = flat_src.remove_columns([i for i in flat_src.column_names if i != KEYS.SEGMENT])
        flat_tgt = flat_tgt.remove_columns([i for i in flat_tgt.column_names if i != KEYS.SEGMENT])
        fingerprints = IndexedParallelFingerprints.make_fingerprints(src_path, tgt_path, align_path, cls.version)
        # Add the language information by adding a new column to flat_src and flat_tgt
        flat_src = flat_src.add_column(KEYS.LANG, [data_language_mapper[src_lang]] * len(flat_src))  # type: ignore
        flat_tgt = flat_tgt.add_column(KEYS.LANG, [data_language_mapper[tgt_lang]] * len(flat_tgt))  # type: ignore

        obj = cls(
            name,
            is_bt,
            flat_align,
            flat_src,
            flat_tgt,
            dictionary,
            encoder=encoder,
            seed=seed,
            max_seq_len=max_seq_len,
            max_merges=max_merges,
            fingerprints=fingerprints,
        )

        # we do this so that HuggingFace's InMemoryTable/ ConcatenationTable is memorymapped,
        #    that is, it becomes a MemoryMappedTable
        obj.cache_to_disk()
        memorymapped_obj = cls.load_from_cache(
            name,
            is_bt,
            src_path,
            tgt_path,
            dictionary,
            encoder=encoder,
            max_seq_len=max_seq_len,
            max_merges=max_merges,
            align_path=align_path,
            seed=seed,
        )
        assert memorymapped_obj is not None
        return memorymapped_obj

    def ordered_sizes(self):
        return self._sorted_lengths

    def ordered_indices(self):
        return self._sorted_indices

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        logger.info(f"Preparing epoch {epoch}")
        self.interleave_indices()
        logger.info(str(self))
        logger.info(f"Done preparing epoch {epoch}")

    def interleave_indices(self):
        # XXX: implement differing behavior based on interleave strategy -- for valid vs train
        #      self.interleave_strategy -- should work well by setting max_merges to 1 (or 0?)
        self.index_dataset = merge_adjacent_sentences(
            self.flat_align,
            max_seq_len=self.max_seq_len,
            max_merges=self.max_merges,
            seed=(self.seed, self.epoch),
        )
        if self.epoch != self._interleave_epoch_index:
            logger.info("Merging adjacent segments in source and target")
            self.index_dataset = merge_adjacent_sentences(
                self.flat_align,
                max_seq_len=self.max_seq_len,
                max_merges=self.max_merges,
                seed=(self.seed, self.epoch),
            )
            self._interleave_epoch_index = self.epoch
            lengths = np.array(self.index_dataset[KEYS.WEIGHT])
            logger.info("Sorting index dataset on lengths")
            self._sorted_indices = lengths.argsort()
            self._sorted_lengths = lengths[self._sorted_indices]
            self._sizes = self._sorted_lengths

    def __str__(self) -> str:
        return f"ParallelDataset(name={self.name},\
is_bt={self.is_bt},num_alignment_pairs={len(self.flat_align)},\
num_training_pairs={len(self.index_dataset)}, skipped={self.num_skipped})"


def compute_doc_offsets(
    dataset: HFDataset,
    load_from_cache_file=True,
    num_proc: int = 4,
    fingerprint: Optional[str] = None,
):
    """Compute the segment offsets of each document in the dataset, over the whole dataset.
    This returns a dataset with a single column, NUM_SEGMENTS, which is a list of integers, one integer per document
    and it is the number of segments in previous documents."""

    def _get_num_segments(batched_docs):
        """Compute the number of segments in each document"""
        return {KEYS.NUM_SEGMENTS: [sum(len(pg) for pg in doc) for doc in batched_docs]}

    dataset = dataset.map(
        _get_num_segments,
        batched=True,
        batch_size=250,
        input_columns=[KEYS.DOCUMENT],
        remove_columns=dataset.column_names,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )
    dataset.set_format(type="numpy", columns=[KEYS.NUM_SEGMENTS], output_all_columns=False)

    # compute offsets, i.e. the start index of each in terms of segment indieces
    # this is "just" the cumulative sum of the number of segments over the dataset
    dataset = hf_datasets.Dataset(
        pa.Table.from_arrays(
            [lengths_to_offsets(dataset[KEYS.NUM_SEGMENTS])],  # type: ignore
            names=[KEYS.NUM_SEGMENTS],
        ),
        fingerprint=str(fingerprint) + f".{KEYS.DOCUMENT_SEGMENT_OFFSETS}",
    )
    return dataset


def flatten_document_dataset(
    dataset: HFDataset,
    load_from_cache_file=True,
    num_proc: int = 4,
):
    def _inner_flatten_document_dataset(
        ex_batched,
        document_idxs: List[int],
    ):
        # this should be applied with batched=True
        ex_out = {
            KEYS.DOCUMENT_INDEX: [],
            KEYS.PARAGRAPH_INDEX: [],
            KEYS.SEGMENT: [],
        }
        for doc_idx, doc in zip(document_idxs, ex_batched[KEYS.DOCUMENT]):
            doc = cast(List[List[str]], doc)
            doc_segments, doc_pg_idxs, doc_doc_idxs = [], [], []
            for pg_idx, pg in enumerate(doc):
                if any((not s) for s in pg for pg in doc):
                    # most of EEA documents have empty sentences
                    # logger.error(f"Found empty sentence in {doc_idx} pg {pg_idx}... skipping document")
                    # doc_segments, doc_pg_idxs, doc_doc_idxs = [], [], []
                    # break
                    pass
                doc_segments.extend(pg)
                doc_pg_idxs.extend(len(pg) * [pg_idx])
                doc_doc_idxs.extend(len(pg) * [doc_idx])
            ex_out[KEYS.SEGMENT].extend(doc_segments)
            ex_out[KEYS.PARAGRAPH_INDEX].extend(doc_pg_idxs)
            ex_out[KEYS.DOCUMENT_INDEX].extend(doc_doc_idxs)
        return ex_out

    dataset = dataset.map(
        _inner_flatten_document_dataset,
        batched=True,
        batch_size=250,
        with_indices=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )
    # the column_names dataset._format_columns can become stale if saved to disk
    # this is to avoid that
    dataset.set_format("numpy", KEYS.SEGMENT)  # type: ignore
    return dataset


def make_align_dataset_from_monolingual_document_dataset(
    dataset: HFDataset,
    *,
    langs: List[str],
    load_from_cache_file=True,
    num_proc: int = 4,
):
    def document_to_one_to_one_alignment(example):
        idxs = []
        for pg_idx, paragraph in enumerate(example[KEYS.DOCUMENT]):
            idxs.extend([[[[pg_idx, s_idx]], [[pg_idx, s_idx]]] for s_idx in range(len(paragraph))])
        return {
            KEYS.UUID: example[KEYS.UUID],
            KEYS.ALIGNMENTS: idxs,
        }

    keep_columns = [KEYS.UUID]
    remove_columns = [k for k in dataset.column_names if k not in keep_columns]
    dataset = dataset.map(
        document_to_one_to_one_alignment,
        remove_columns=remove_columns,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )
    dataset = dataset.add_column(KEYS.LANGS, len(dataset) * [langs])  # type: ignore
    return dataset


def make_align_dataset_default_alignments(
    src_dataset: HFDataset,
    tgt_dataset: HFDataset,
    *,
    load_from_cache_file=True,
    num_proc: int = 4,
) -> HFDataset:
    """Creates an alignment dataset which assumes 1-1 alignment between source and target sentences."""

    def _inner(example, doc_idx):
        idxs = []
        for pg_idx, (src_pg, tgt_pg) in enumerate(zip(example[KEYS.DOCUMENT], example[KEYS.DOCUMENT + ".tgt"])):
            pg_len = len(src_pg)
            new = [[[[pg_idx, s_idx]], [[pg_idx, s_idx]]] for s_idx in range(pg_len)]
            if len(example[KEYS.DOCUMENT][pg_idx]) != len(example[KEYS.DOCUMENT + ".tgt"][pg_idx]):
                logger.error(f"Alignment mismatch in default alignment in doc {doc_idx} pg {pg_idx}... skipping")
                breakpoint()
                continue
            idxs.extend(new)
        return {
            KEYS.UUID: example[KEYS.UUID],
            KEYS.ALIGNMENTS: idxs,
        }

    dataset = hf_datasets.concatenate_datasets(
        [
            src_dataset,
            tgt_dataset.rename_columns({name: f"{name}.tgt" for name in tgt_dataset.column_names}),
        ],
        axis=1,
    )
    dataset = dataset.map(
        _inner,
        remove_columns=dataset.column_names,
        with_indices=True,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )
    langs = [src_dataset[0][KEYS.LANG], tgt_dataset[0][KEYS.LANG]]
    dataset = dataset.add_column(KEYS.LANGS, len(dataset) * [langs])  # type: ignore
    return dataset


def compute_offsets_and_flatten_alignments(
    src_dataset,
    tgt_dataset,
    align_dataset,
    bpe_encoder,
    dictionary: Dictionary,
    load_from_cache_file=True,
    num_proc: int = 4,
    flip_alignment: bool = False,
    max_sequence_length: int = 1024,
):
    """Flattens the alignments so that each element of the dataset refers to a translation pair.
    Additionally computes "weights" (number of BPEs) of src and tgt and "skip", i.e. whether a pair should be skipped.

    Returns a Dataset with the following columns:
    - document_index: int, the index of the document in the original dataset
    - paragraph_index: int, the index of the paragraph in the document
    - weight: int, the max of src and tgt number of BPEs in the segment pair
    - source_indices: list of ints, the indices of the source segments in the pair
    - target_indices: list of ints, the indices of the target segments in the pair
    - skip: bool, whether to skip this pair
    """
    num_proc = 1 if len(src_dataset) < 11_000 else num_proc
    logger.info("Computing sentence lengths of src documents")
    # src_num_bpes has a single column: sentence_weights and a single example (document) is a
    # list of lists of segment lengths:
    # [
    #   [56, 31, 622, 1239, 4983], # first paragraph has 5 segments of these lengths
    #   [954, 33, 3, 4, 5, 6, 7],
    #   ...
    # ]
    src_num_bpes = get_mono_document_sentence_lengths_dataset(
        src_dataset,
        bpe_encoder,
        dictionary,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    ).rename_column(KEYS.SENTENCE_WEIGHTS, KEYS.SOURCE_WEIGHTS, new_fingerprint=None)
    logger.info("Computing sentence lengths of tgt documents")
    tgt_num_bpes = get_mono_document_sentence_lengths_dataset(
        tgt_dataset,
        bpe_encoder,
        dictionary,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    ).rename_column(KEYS.SENTENCE_WEIGHTS, KEYS.TARGET_WEIGHTS, new_fingerprint=None)

    logger.info("Computing offsets")
    # the offset is a single column which is a list of integers, which is the cumulative sum of the
    # number of segments in each document, over all the documents.
    # For example, if the first document has 5 segments and the second document has 7 segments, then
    # the offset will start with: [0, 5, 12, ...]
    src_doc_offsets = compute_doc_offsets(
        src_dataset,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        fingerprint="src",
    ).rename_column(KEYS.NUM_SEGMENTS, KEYS.SOURCE_OFFSETS, new_fingerprint="src")
    tgt_doc_offsets = compute_doc_offsets(
        tgt_dataset,
        num_proc=num_proc,
        load_from_cache_file=load_from_cache_file,
        fingerprint="tgt",
    ).rename_column(KEYS.NUM_SEGMENTS, KEYS.TARGET_OFFSETS, new_fingerprint="tgt")

    logger.info("Flattening alignments")
    # align_w_info is a dataset with the following columns:
    # - alignments as per the original alignment dataset
    # - langs as per the original alignment dataset
    # - uuid as per the original alignment dataset
    # - src_doc: the source document
    # - tgt_doc: the target document
    # - src_offsets: the cumulative sum of the number of segments in each document, over all the documents.
    # - tgt_offsets: the cumulative sum of the number of segments in each document, over all the documents.
    # - src_weights: the number of BPEs in each segment of the source document
    # - tgt_weights: the number of BPEs in each segment of the target document
    align_w_info = hf_datasets.concatenate_datasets(
        [
            align_dataset,
            src_num_bpes,
            tgt_num_bpes,
            src_doc_offsets,
            tgt_doc_offsets,
            src_dataset.remove_columns([c for c in src_dataset.column_names if c != KEYS.DOCUMENT]).rename_column(
                KEYS.DOCUMENT, "src_doc"
            ),
            tgt_dataset.remove_columns([c for c in tgt_dataset.column_names if c != KEYS.DOCUMENT]).rename_column(
                KEYS.DOCUMENT, "tgt_doc"
            ),
        ],
        axis=1,
    )

    flat = flatten_alignments(
        align_w_info,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
        flip_alignment=flip_alignment,
        max_sequence_length=max_sequence_length,
    )
    return flat


def get_mono_document_sentence_lengths_dataset(
    dataset: HFDataset,
    bpe_encoder,
    dictionary: Dictionary,
    load_from_cache_file=True,
    num_proc: int = 4,
):
    """Computes the number of BPEs in each sentence, for each paragraph for each document in the dataset.
    Returns a Dataset with a single column which is a list of lists of ints, where the ints are the number of BPEs."""

    def _mono_document_to_sentence_lengths2(example, bpe_encoder, dictionary: Dictionary):
        sent_lens_per_pg = [
            [
                np.array(
                    [
                        len(
                            dictionary.encode_line(
                                bpe_encoder.encode(segment.strip()),
                                append_eos=False,
                                add_if_not_exist=False,
                            )
                        )
                        for segment in paragraph
                    ],
                    dtype=np.int64,
                )
                for paragraph in doc
            ]
            for doc in example[KEYS.DOCUMENT]
        ]
        return {KEYS.SENTENCE_WEIGHTS: sent_lens_per_pg}

    dataset = dataset.map(
        _mono_document_to_sentence_lengths2,
        fn_kwargs={"bpe_encoder": bpe_encoder, "dictionary": dictionary},
        batch_size=250,
        batched=True,
        remove_columns=dataset.column_names,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )
    # this fixes internal pyarrow block length mismatch which occurs when concatting this table with offsets table
    return hf_datasets.Dataset.from_pandas(dataset.to_pandas()).with_format("numpy", KEYS.SENTENCE_WEIGHTS)


def flatten_alignments(
    align_dataset: HFDataset,
    *,
    load_from_cache_file=True,
    num_proc: int = 4,
    flip_alignment: bool = False,
    max_sequence_length: int = 1024,
):
    """Flattens the alignments in the dataset, so that each row is a single alignment pair.
    Returns a Dataset with the following columns:
    - document_index: int, the index of the document in the original dataset
    - paragraph_index: int, the index of the paragraph in the document
    - weight: int, the max of src and tgt number of BPEs in the segment pair
    - source_indices: list of ints, the indices of the source segments in the pair
    - target_indices: list of ints, the indices of the target segments in the pair
    - skip: bool, whether to skip this pair
    """

    def _inner(
        ex_batched,
        indices: List[int],
        *,
        flip_alignment: bool = False,
        min_long_weight: int = 4,
        max_relative_diff: float = 1.4,
        max_sequence_length: int = 1024,
    ):
        # this should be applied with batched=True and with_indices=True
        ex_out = {
            KEYS.DOCUMENT_INDEX: [],
            KEYS.PARAGRAPH_INDEX: [],
            KEYS.WEIGHT: [],
            KEYS.SOURCE_INDICES: [],
            KEYS.TARGET_INDICES: [],
            KEYS.SKIP: [],
        }
        for rel_doc_idx, (abs_idx, doc_alignments) in enumerate(zip(indices, ex_batched[KEYS.ALIGNMENTS])):
            doc_alignments = cast(DocumentAlignment, doc_alignments)
            # semantics of each index:
            # doc_alignments[pair_idx][source or target][segments in pair][pg_idx or seg_idx]
            # src_doc_offset and tgt_doc_offset are the segment offsets over the whole dataset
            # i.e. the number of segments in all the documents before this one
            src_doc_offset = ex_batched[KEYS.SOURCE_OFFSETS][rel_doc_idx]
            assert isinstance(src_doc_offset, int)
            tgt_doc_offset = ex_batched[KEYS.TARGET_OFFSETS][rel_doc_idx]
            assert isinstance(tgt_doc_offset, int)
            # scr_pg_offsets and tgt_pg_offsets are the segment offsets for each paragraph in the document
            # i.e. the number of segments in all the paragraphs of this document, starting from this document
            # example: given a document with 3 paragraphs, with 2, 3, and 1 segments respectively, the offsets are
            # [0, 2, 5]
            src_pg_offsets = lengths_to_offsets([len(pg) for pg in ex_batched[KEYS.SOURCE_WEIGHTS][rel_doc_idx]])
            tgt_pg_offsets = lengths_to_offsets([len(pg) for pg in ex_batched[KEYS.TARGET_WEIGHTS][rel_doc_idx]])

            skip_pairs = np.repeat(False, len(doc_alignments))
            pair_weights = np.repeat(0, len(doc_alignments))
            ex_out[KEYS.DOCUMENT_INDEX].extend(len(doc_alignments) * [abs_idx])
            for rel_pair_idx, (src_parts, tgt_parts) in enumerate(doc_alignments):
                if flip_alignment:
                    src_parts, tgt_parts = tgt_parts, src_parts
                src_pg_indices, rel_src_seg_indices = zip(*src_parts)
                tgt_pg_indices, rel_tgt_seg_indices = zip(*tgt_parts)
                pg_idx = src_pg_indices[0]
                assert all(pg_idx == idx for idx in src_pg_indices) and all(
                    pg_idx == idx for idx in tgt_pg_indices
                ), "all segments in a pair must be from the same paragraph"
                # the total "weight" (number of BPE fragments) of the source and target segments
                src_weight = sum(
                    ex_batched[KEYS.SOURCE_WEIGHTS][rel_doc_idx][pg_idx][seg_idx] for seg_idx in rel_src_seg_indices
                )
                tgt_weight = sum(
                    ex_batched[KEYS.TARGET_WEIGHTS][rel_doc_idx][pg_idx][seg_idx] for seg_idx in rel_tgt_seg_indices
                )
                pair_weights[rel_pair_idx] = max(src_weight, tgt_weight)

                # skip if any part is of length 0 or maximum length is exceeded
                if min(src_weight, tgt_weight) == 0:
                    skip_pairs[rel_pair_idx] = True
                elif max(src_weight, tgt_weight) >= max_sequence_length:
                    skip_pairs[rel_pair_idx] = True

                ex_out[KEYS.PARAGRAPH_INDEX].append(pg_idx)
                ex_out[KEYS.SOURCE_INDICES].append(
                    [src_doc_offset + src_pg_offsets[pg_idx] + seg_offset for seg_offset in rel_src_seg_indices]
                )
                ex_out[KEYS.TARGET_INDICES].append(
                    [tgt_doc_offset + tgt_pg_offsets[pg_idx] + seg_offset for seg_offset in rel_tgt_seg_indices]
                )

            ex_out[KEYS.SKIP].extend(skip_pairs.tolist())
            ex_out[KEYS.WEIGHT].extend(pair_weights.tolist())
        return ex_out

    dataset = align_dataset.map(
        _inner,
        batched=True,
        batch_size=250,
        with_indices=True,
        fn_kwargs={"flip_alignment": flip_alignment, "max_sequence_length": max_sequence_length},
        remove_columns=align_dataset.column_names,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )

    return dataset


def merge_adjacent_sentences(
    flat_align: HFDataset,
    num_proc: int = 4,
    max_seq_len: int = 1024,
    max_merges: int = 10,
    no_merge_prob: float = 0.1,
    seed: Tuple[int, ...] = (1,),
):
    def _inner(
        ex_batched,
        abs_align_indices,
        *,
        max_seq_len: int,
        max_merges: int,
        no_merge_prob: float,
        seed: Tuple[int, ...],
    ) -> Dict[str, List]:
        # The keys of the flat_align dataset - see the docstring of the compute_offsets_and_flatten_alignments function
        keys = [
            KEYS.DOCUMENT_INDEX,
            KEYS.PARAGRAPH_INDEX,
            KEYS.WEIGHT,
            KEYS.SOURCE_INDICES,
            KEYS.TARGET_INDICES,
            KEYS.SKIP,
        ]
        doc_idxs, pg_idxs, weights, all_src_idxs, all_tgt_idxs, skip = [ex_batched[k] for k in keys]

        # SOURCE_OFFSETS and TARGET_OFFSETS in a IndexedParallelBTDocumentsDataset
        # where many flat_src and flat_tgt are concatenated together, this would skew the source/target_indices
        # so we need to offset them by the document offsets
        if KEYS.SOURCE_OFFSETS in ex_batched:
            src_doc_offsets = ex_batched[KEYS.SOURCE_OFFSETS]
            all_src_idxs = [
                [idx + document_offset for idx in idxs]
                for (idxs, document_offset) in zip(all_src_idxs, src_doc_offsets)
            ]
            doc_idxs = [idx + document_offset for (idx, document_offset) in zip(doc_idxs, src_doc_offsets)]
        if KEYS.TARGET_OFFSETS in ex_batched:
            tgt_doc_offsets = ex_batched[KEYS.TARGET_OFFSETS]
            all_tgt_idxs = [
                [idx + document_offset for idx in idxs]
                for (idxs, document_offset) in zip(all_tgt_idxs, tgt_doc_offsets)
            ]

        # cast for readability
        all_src_idxs = cast(List[List[int]], all_src_idxs)
        all_tgt_idxs = cast(List[List[int]], all_tgt_idxs)
        doc_idxs = cast(List[int], doc_idxs)
        pg_idxs = cast(List[int], pg_idxs)
        weights = cast(List[int], weights)
        skip = cast(List[bool], skip)
        abs_align_indices = cast(List[int], abs_align_indices)

        ex_out = {
            KEYS.SOURCE_INDICES: [],
            KEYS.TARGET_INDICES: [],
            KEYS.WEIGHT: [],
            KEYS.EXACT_ALIGNMENT: [],
        }
        # set up reproducible rng state that depends implicitly on batch_size but is invariant to num_proc
        rng = np.random.default_rng((seed, abs_align_indices))
        # these are the sequence lengths of the bins we will use to merge sentences
        extra_padding = max_merges + 1 + 1 + 1  # SENT_SEP*max_merges + Start + End + Maybe BT info
        maximum_lengths = np.array([50, 100, 150, 350, max_seq_len - extra_padding], dtype=np.int64)

        # fetch maximum number of rolls to minimize fn calls
        bin_idxs = np.clip(rng.poisson(_POISSON_MEAN, size=len(doc_idxs)), 0, len(maximum_lengths) - 1)
        # examples to pass through without merging
        ok_to_merge = rng.random(len(doc_idxs)) > no_merge_prob

        # Initial conditions
        last_context_idxs: Optional[Tuple[int, int]] = None
        accum_src, accum_tgt, accum_weight = (
            [],
            [],
            0,
        )
        num_outputs, curr_num_merges = 0, 0
        for _loop_idx, (
            doc_idx,
            pg_idx,
            weight,
            src_idxs,
            tgt_idxs,
            skip_example,
        ) in enumerate(zip(doc_idxs, pg_idxs, weights, all_src_idxs, all_tgt_idxs, skip)):
            # This loop is a bit tricky to understand, here is a high level overview:
            # We iterate over the examples in the batch, and for each example we roll a dice to decide whether to
            # merge it with the next example or not. If we decide to merge, we check whether the accumulated weight
            # is within the maximum sequence length, if it is, we merge the current example with the accumulator
            # and continue. If it is not, we store the accumulator and reset it to the current example.
            # An implicit assumption here is that the examples have not been shuffled,
            # so that nearby examples "are in context"

            # examples are skipped if they are structurally invalid: too long, too short (no fragments) or src and tgt
            # have very different relative lengths
            if skip_example:
                continue

            # only happens on the first iteration
            if last_context_idxs is None:
                is_boundary = False
            else:
                # if we are at a boundary - we cannot merge.
                is_boundary = last_context_idxs != (doc_idx, pg_idx) or (curr_num_merges == max_merges)
            # if we are not ok with merging - we cannot merge
            is_not_ok_to_merge = not ok_to_merge[num_outputs]
            # if we do not have a budget to merge the current example with the accumulator - we cannot merge
            has_no_budget = accum_weight + weight > maximum_lengths[bin_idxs[num_outputs]]
            # if we are at any of these conditions - we cannot merge the current example to our accumulator
            if is_boundary or is_not_ok_to_merge or has_no_budget:
                if accum_src and accum_tgt:
                    ex_out[KEYS.SOURCE_INDICES].append(accum_src)
                    ex_out[KEYS.TARGET_INDICES].append(accum_tgt)
                    ex_out[KEYS.WEIGHT].append(accum_weight)
                    ex_out[KEYS.EXACT_ALIGNMENT].append(len(accum_src) == len(accum_tgt))
                    num_outputs += 1
                    curr_num_merges = 0
                # store the current example in the accumulator
                accum_src, accum_tgt, accum_weight = (
                    src_idxs,
                    tgt_idxs,
                    weight,
                )
            else:
                # merge the current example to the accumulator
                accum_src.extend(src_idxs)
                accum_tgt.extend(tgt_idxs)
                accum_weight += weight
                curr_num_merges += 1

            # update the context
            last_context_idxs = (doc_idx, pg_idx)

        # if we have something left in the accumulator - add it
        if accum_src and accum_tgt:
            ex_out[KEYS.SOURCE_INDICES].append(accum_src)
            ex_out[KEYS.TARGET_INDICES].append(accum_tgt)
            ex_out[KEYS.WEIGHT].append(accum_weight)
            ex_out[KEYS.EXACT_ALIGNMENT].append(len(accum_src) == len(accum_tgt))
            num_outputs += 1

        return ex_out

    assert max_seq_len is not None
    dataset = flat_align.map(
        _inner,
        batched=True,
        with_indices=True,
        batch_size=5_000,
        remove_columns=flat_align.column_names,
        fn_kwargs={
            "max_seq_len": max_seq_len,
            "seed": seed,
            "max_merges": max_merges,
            "no_merge_prob": no_merge_prob,
        },
        # we never want to load this from cache, since it should be fresh per epoch
        load_from_cache_file=False,
        num_proc=num_proc,
    )
    dataset.set_format("numpy", [KEYS.SOURCE_INDICES, KEYS.TARGET_INDICES, KEYS.WEIGHT, KEYS.EXACT_ALIGNMENT])
    return dataset
