from dataclasses import dataclass
import logging
from typing import List, Optional, Tuple
from pathlib import Path

import pyarrow as pa
import datasets as hf_datasets
import numpy as np
import torch
from datasets import Dataset as HFDataset
from datasets import Sequence, Value
from fairseq.data import Dictionary, LanguagePairDataset, data_utils

from fairseq_user_dir.encoders import Encoder

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

_ALIGNMENTS_JSONL_FEATURE_DICT = {
    "langs": Sequence(feature=Value(dtype="string", id=None), length=-1, id=None),
    "alignments": Sequence(
        feature=Sequence(
            feature=Sequence(
                feature=Sequence(
                    feature=Value(dtype="int64", id=None), length=-1, id=None
                ),
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


def lengths_to_offsets(lengths):
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


@dataclass
class IndexedParallelFingerprints:
    source: str
    target: str
    align: str

    @classmethod
    def make_fingerprints(cls, src_paths, tgt_paths, align_paths, version):
        from datasets.fingerprint import Hasher

        # there is no randomness that goes into flattening so we don't need seed here
        prefix = [src_paths, tgt_paths, align_paths, version]
        align_hash = Hasher.hash(prefix + ["flat_align"])
        src_hash = Hasher.hash(prefix + ["flat_src"])
        tgt_hash = Hasher.hash(prefix + ["flat_tgt"])
        return cls(source=src_hash, target=tgt_hash, align=align_hash)


class IndexedParallelDocumentsDataset(LanguagePairDataset):
    version = "1.7"

    def __init__(
        self,
        flat_align,
        flat_src,
        flat_tgt,
        dictionary,
        encoder: Encoder,
        append_source_id=None,
        append_target_id=None,
        seed: int = 1,
        max_seq_len=None,
        max_merges=10,
        fingerprints: IndexedParallelFingerprints = None,
        paths=None,
    ):
        super().__init__(None, 0, dictionary)
        self.flat_align = flat_align
        self.flat_src = flat_src
        self.flat_tgt = flat_tgt
        self.max_seq_len = max_seq_len
        self.max_merges = max_merges
        self.index_dataset = None  # gets set beginning of each bepoch
        self.dictionary = dictionary
        self.seed = seed
        self.paths = paths
        # ConcatDataset expects a numpy array or list
        # self.sizes = np.array(self.index_dataset[KEYS.LENGTH])
        self.append_source_id = append_source_id
        self.append_target_id = append_target_id
        self.encoder = encoder
        # this is compatibility with LanguagePairDataset collater and its teacher forcing adjustments
        self.src_dict = self.dictionary
        self.left_pad_source = False  # expected by bart model
        self.left_pad_target = False  # expected by bart model
        self.src_lang_id = None  # fairseq 0.10.2 accesses these in LanguagePairDataset.collater (so this attribute must exist)
        self.tgt_lang_id = None

        self._interleave_epoch_index = None
        self.src_sizes = self.sizes
        self.tgt_sizes = None
        self._dataset_ntokens = None
        self._sorted_indices = None
        self._sorted_lengths = None
        self._bpe = None
        self.epoch = 1
        self.fingerprints = fingerprints

    def __len__(self):
        return len(self.index_dataset)

    def size(self, index):
        return self.index_dataset[int(index)][KEYS.LENGTH]

    @property
    def dataset_ntokens(self):
        if self._dataset_ntokens is None:
            self._dataset_ntokens = sum(
                self.document_dataset[f"{KEYS.DOCUMENT_WEIGHT}.{KEYS.LANG1}"]
            )
        return self._dataset_ntokens

    def __getitem__(self, index):
        item = self.index_dataset[int(index)]
        src_segments = [self.flat_src[int(i)]["segment"] for i in item[KEYS.SOURCE_INDICES]]
        tgt_segments = [self.flat_tgt[int(i)]["segment"] for i in item[KEYS.TARGET_INDICES]]

        with data_utils.numpy_seed(self.seed, self.epoch, index):
            insert_sep = np.random.randint(2, dtype=np.bool)

        if insert_sep and len(src_segments) > 1:
            bos = torch.tensor([self.dictionary.bos()])
            src_out = [bos] * (len(src_segments) * 2 - 1)
            src_out[0::2] = [self.encoder.encode(seg) for seg in src_segments]
            tgt_out = [bos] * (len(tgt_segments) * 2 - 1)
            tgt_out[0::2] = [self.encoder.encode(seg) for seg in tgt_segments]
        else:
            src_out = [self.encoder.encode(seg) for seg in src_segments]
            tgt_out = [self.encoder.encode(seg) for seg in tgt_segments]

        src_affix = (
            [self.dictionary.eos()]
            if self.append_source_id is None
            else [self.dictionary.eos(), self.append_source_id]
        )
        tgt_affix = (
            [self.dictionary.eos()]
            if self.append_target_id is None
            else [self.dictionary.eos(), self.append_target_id]
        )
        src_out = torch.cat(src_out + [torch.tensor(src_affix)])
        tgt_out = torch.cat(tgt_out + [torch.tensor(tgt_affix)])

        return {"id": index, "source": src_out, "target": tgt_out}

    def cache_to_disk(self):
        if self.fingerprints is None:
            raise ValueError("Cannot save without fingerprints")
        cache_dir = hf_datasets.config.HF_DATASETS_CACHE
        self.flat_src.save_to_disk(f"{cache_dir}/{self.fingerprints.source}")
        self.flat_tgt.save_to_disk(f"{cache_dir}/{self.fingerprints.target}")
        self.flat_align.save_to_disk(f"{cache_dir}/{self.fingerprints.align}")

    @classmethod
    def load_from_cache(
        cls,
        src_paths: List[str],
        tgt_paths: List[str],
        bpe_encoder,
        dictionary: Dictionary,
        encoder: Encoder,
        max_seq_len: int = None,
        append_source_id: int = None,
        append_target_id: int = None,
        max_merges: Optional[int] = 10,
        align_paths: Optional[str] = None,
        seed: int = 1,
    ):
        cache_dir = hf_datasets.config.HF_DATASETS_CACHE

        fp = IndexedParallelFingerprints.make_fingerprints(
            src_paths, tgt_paths, align_paths, cls.version
        )
        if not all(
            Path(f"{cache_dir}/{val}").exists()
            for val in [fp.source, fp.target, fp.align]
        ):
            return None
        flat_src = hf_datasets.load_from_disk(f"{cache_dir}/{fp.source}")
        flat_tgt = hf_datasets.load_from_disk(f"{cache_dir}/{fp.target}")
        flat_align = hf_datasets.load_from_disk(f"{cache_dir}/{fp.align}")

        return cls(
            flat_align,
            flat_src,
            flat_tgt,
            dictionary,
            encoder=encoder,
            append_source_id=append_source_id,
            append_target_id=append_target_id,
            seed=seed,
            max_seq_len=max_seq_len,
            max_merges=max_merges,
            fingerprints=fp,
            paths=[src_paths, tgt_paths],
        )

    @classmethod
    def from_parallel_jsonl_many(
        cls,
        src_paths: List[str],
        tgt_paths: List[str],
        bpe_encoder,
        dictionary: Dictionary,
        encoder: Encoder,
        max_seq_len: int = None,
        append_source_id: int = None,
        append_target_id: int = None,
        max_merges: int = 10,
        load_from_cache_file: bool = True,
        num_proc: int = 8,
        align_paths: Optional[str] = None,
        seed: int = 1,
    ):
        # check if we it is already precomputed
        #if load_from_cache_file and False:
        if load_from_cache_file:
            cached_dataset = cls.load_from_cache(
                src_paths,
                tgt_paths,
                bpe_encoder,
                dictionary,
                encoder,
                max_seq_len=max_seq_len,
                append_source_id=append_source_id,
                append_target_id=append_target_id,
                max_merges=max_merges,
                align_paths=align_paths,
                seed=seed,
            )
            if cached_dataset is not None:
                logger.info(
                    f"Found matching cached dataset for {src_paths} and {tgt_paths}"
                )
                return cached_dataset

        features = hf_datasets.Features(_DOCUMENT_JSONL_FEATURE_DICT)
        logger.info(f"Loading src_dataset: {src_paths}")
        src_dataset = hf_datasets.Dataset.from_json(
            src_paths, split="train", chunksize=40 << 20, features=features
        )
        logger.info(f"Loading tgt_dataset: {tgt_paths}")
        tgt_dataset = hf_datasets.Dataset.from_json(
            tgt_paths, split="train", chunksize=40 << 20, features=features
        )

        src_lang = src_dataset[0][KEYS.LANG]
        tgt_lang = tgt_dataset[0][KEYS.LANG]

        if not any("valid" in p for p in src_paths):
            pass

        flip_alignment = False
        align_dataset = None
        if align_paths is not None:
            logger.info(f"Loading alignments: {align_paths}")
            align_dataset = hf_datasets.Dataset.from_json(
                align_paths,
                split="train",  # XXX: @haukurpall, does this always apply?
                chunksize=40 << 20,
                features=hf_datasets.Features(_ALIGNMENTS_JSONL_FEATURE_DICT),
            )
            assert set(align_dataset[0][KEYS.LANGS]) == set([src_lang, tgt_lang])
            if align_dataset[0][KEYS.LANGS] != [src_lang, tgt_lang]:
                flip_alignment = True
                logger.info(f"Flipping alignments: {align_paths}")
        else:
            align_dataset = make_align_dataset_default_alignments(
                src_dataset, tgt_dataset
            )

        flat_align = do_everything(
            src_dataset,
            tgt_dataset,
            align_dataset,
            bpe_encoder,
            dictionary,
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
            flip_alignment=flip_alignment,
        )
        flat_src = flatten_document_dataset(
            src_dataset,
            bpe_encoder,
            dictionary,
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
        )
        flat_tgt = flatten_document_dataset(
            tgt_dataset,
            bpe_encoder,
            dictionary,
            load_from_cache_file=load_from_cache_file,
            num_proc=num_proc,
        )

        flat_src = flat_src.remove_columns(
            [i for i in flat_src.column_names if i != KEYS.SEGMENT]
        )
        flat_tgt = flat_tgt.remove_columns(
            [i for i in flat_tgt.column_names if i != KEYS.SEGMENT]
        )
        fingerprints = IndexedParallelFingerprints.make_fingerprints(
            src_paths, tgt_paths, align_paths, cls.version
        )

        obj = cls(
            flat_align,
            flat_src,
            flat_tgt,
            dictionary,
            encoder=encoder,
            append_source_id=append_source_id,
            append_target_id=append_target_id,
            seed=seed,
            max_seq_len=max_seq_len,
            max_merges=max_merges,
            fingerprints=fingerprints,
            paths=[src_paths, tgt_paths],
        )

        # we do this so that HuggingFace's InMemoryTable/ ConcatenationTable is memorymapped,
        #    that is, it becomes a MemoryMappedTable
        obj.cache_to_disk()
        memorymapped_obj = cls.load_from_cache(
            src_paths,
            tgt_paths,
            bpe_encoder,
            dictionary,
            encoder,
            max_seq_len=max_seq_len,
            append_source_id=append_source_id,
            append_target_id=append_target_id,
            max_merges=max_merges,
            align_paths=align_paths,
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
        logger.info(f"Preparing next epoch")
        self.interleave_indices()
        logger.info(f"Done preparing epoch")

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
            logger.info(f"Merging adjacent segments in source and target")
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


def compute_doc_offsets(
    dataset: HFDataset,
    load_from_cache_file=True,
    num_proc: int = 4,
    fingerprint: Optional[str] = None,
):
    def _get_num_segments(batched_docs):
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
    dataset.set_format(
        type="numpy", columns=[KEYS.NUM_SEGMENTS], output_all_columns=False
    )

    dataset = hf_datasets.Dataset(
        pa.Table.from_arrays(
            [lengths_to_offsets(dataset[KEYS.NUM_SEGMENTS])],
            names=[KEYS.NUM_SEGMENTS],
        ),
        fingerprint=str(fingerprint) + f".{KEYS.DOCUMENT_SEGMENT_OFFSETS}",
    )
    return dataset


def flatten_document_dataset(
    dataset: HFDataset,
    bpe_encoder,
    dictionary: Dictionary,
    load_from_cache_file=True,
    num_proc: int = 4,
):
    def _inner_flatten_document_dataset(
        ex_batched,
        document_index: int,
    ):
        # this should be applied with batched=True
        ex_out = {
            KEYS.DOCUMENT_INDEX: [],
            KEYS.PARAGRAPH_INDEX: [],
            KEYS.SEGMENT: [],
        }
        for doc_idx, doc in zip(document_index, ex_batched[KEYS.DOCUMENT]):
            for pg_idx, pg in enumerate(doc):
                if any((not s) for s in pg for pg in doc):
                    breakpoint()
                    print()
                ex_out[KEYS.SEGMENT].extend(pg)
                ex_out[KEYS.PARAGRAPH_INDEX].extend(len(pg) * [pg_idx])
                ex_out[KEYS.DOCUMENT_INDEX].extend(len(pg) * [doc_idx])
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
    dataset.set_format("numpy", KEYS.SEGMENT)
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
            idxs.extend(
                [
                    [[[pg_idx, s_idx]], [[pg_idx, s_idx]]]
                    for s_idx in range(len(paragraph))
                ]
            )
        return {
            KEYS.UUID: example[KEYS.UUID],
            KEYS.ALIGNMENTS: idxs,
        }

    keep_columns = [KEYS.UUID]
    remove_columns = [k for k in dataset.column_names if k not in keep_columns]
    dataset = dataset.map(
        document_to_one_to_one_alignment,
        remove_columns=remove_columns,
        load_from_cache_file=False,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )
    dataset = dataset.add_column(KEYS.LANGS, len(dataset) * [langs])
    return dataset


def make_align_dataset_default_alignments(
    src_dataset: HFDataset,
    tgt_dataset: HFDataset,
    *,
    load_from_cache_file=True,
    num_proc: int = 4,
):
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
            tgt_dataset.rename_columns(
                {name: f"{name}.tgt" for name in tgt_dataset.column_names}
            ),
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
    dataset = dataset.add_column(KEYS.LANGS, len(dataset) * [langs])
    return dataset


def do_everything(
    src_dataset,
    tgt_dataset,
    align_dataset,
    bpe_encoder,
    dictionary: Dictionary,
    load_from_cache_file=True,
    num_proc: int = 4,
    flip_alignment: bool = False,
):
    num_proc = 1 if len(src_dataset) < 11_000 else num_proc
    logger.info(f"Computing sentence lengths of src documents")
    src_num_bpes = get_mono_document_sentence_lengths_dataset2(
        src_dataset,
        bpe_encoder,
        dictionary,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    ).rename_column(KEYS.SENTENCE_WEIGHTS, KEYS.SOURCE_WEIGHTS, new_fingerprint=None)
    logger.info(f"Computing sentence lengths of tgt documents")
    tgt_num_bpes = get_mono_document_sentence_lengths_dataset2(
        tgt_dataset,
        bpe_encoder,
        dictionary,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    ).rename_column(KEYS.SENTENCE_WEIGHTS, KEYS.TARGET_WEIGHTS, new_fingerprint=None)

    logger.info(f"Computing offsets")
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

    logger.info(f"Flattening alignments")
    align_w_info = hf_datasets.concatenate_datasets(
        [
            align_dataset,
            src_num_bpes,
            tgt_num_bpes,
            src_doc_offsets,
            tgt_doc_offsets,
            src_dataset.remove_columns(
                [c for c in src_dataset.column_names if c != KEYS.DOCUMENT]
            ).rename_column(KEYS.DOCUMENT, "src_doc"),
            tgt_dataset.remove_columns(
                [c for c in tgt_dataset.column_names if c != KEYS.DOCUMENT]
            ).rename_column(KEYS.DOCUMENT, "tgt_doc"),
        ],
        axis=1,
    )

    flat = flatten_alignments2(
        align_w_info,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
        flip_alignment=flip_alignment,
    )
    return flat


def get_mono_document_sentence_lengths_dataset2(
    dataset: HFDataset,
    bpe_encoder,
    dictionary: Dictionary,
    load_from_cache_file=True,
    num_proc: int = 4,
):
    def _mono_document_to_sentence_lengths2(
        example, bpe_encoder=None, dictionary: Dictionary = None
    ):
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
    return hf_datasets.Dataset.from_pandas(dataset.to_pandas()).with_format(
        "numpy", KEYS.SENTENCE_WEIGHTS
    )


def flatten_alignments2(
    align_dataset: HFDataset,
    *,
    load_from_cache_file=True,
    num_proc: int = 4,
    flip_alignment: bool = False,
):
    def _inner(
        ex_batched,
        indices,
        *,
        flip_alignment
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
        for rel_doc_idx, (abs_idx, doc_alignments) in enumerate(
            zip(indices, ex_batched[KEYS.ALIGNMENTS])
        ):
            # semantics of each index:
            # doc_alignments[pair_idx][source or target][segments in pair][pg_idx or seg_idx]
            src_doc_offset = ex_batched[KEYS.SOURCE_OFFSETS][rel_doc_idx]
            tgt_doc_offset = ex_batched[KEYS.TARGET_OFFSETS][rel_doc_idx]
            src_pg_offsets = lengths_to_offsets(
                [len(pg) for pg in ex_batched[KEYS.SOURCE_WEIGHTS][rel_doc_idx]]
            )
            tgt_pg_offsets = lengths_to_offsets(
                [len(pg) for pg in ex_batched[KEYS.TARGET_WEIGHTS][rel_doc_idx]]
            )
            skip_arr = np.repeat(False, len(doc_alignments))
            weight_arr = np.repeat(0, len(doc_alignments))
            ex_out[KEYS.DOCUMENT_INDEX].extend(
                len(doc_alignments) * [indices[rel_doc_idx]]
            )
            for rel_pair_idx, (src_parts, tgt_parts) in enumerate(doc_alignments):
                if flip_alignment:
                    src_parts, tgt_parts = tgt_parts, src_parts
                src_pg_indices, rel_src_seg_indices = zip(*src_parts)
                tgt_pg_indices, rel_tgt_seg_indices = zip(*tgt_parts)
                pg_idx = src_pg_indices[0]
                assert all(pg_idx == idx for idx in src_pg_indices) and all(
                    pg_idx == idx for idx in tgt_pg_indices
                )
                src_weight = sum(
                    ex_batched[KEYS.SOURCE_WEIGHTS][rel_doc_idx][pg_idx][seg_idx]
                    for seg_idx in rel_src_seg_indices
                )
                tgt_weight = sum(
                    ex_batched[KEYS.TARGET_WEIGHTS][rel_doc_idx][pg_idx][seg_idx]
                    for seg_idx in rel_tgt_seg_indices
                )
                weight_arr[rel_pair_idx] = max(src_weight, tgt_weight)
                skip_arr[rel_pair_idx] = min(src_weight, tgt_weight) == 0
                ex_out[KEYS.PARAGRAPH_INDEX].append(pg_idx)
                ex_out[KEYS.SOURCE_INDICES].append(
                    [
                        src_doc_offset + src_pg_offsets[pg_idx] + seg_offset
                        for seg_offset in rel_src_seg_indices
                    ]
                )
                ex_out[KEYS.TARGET_INDICES].append(
                    [
                        tgt_doc_offset + tgt_pg_offsets[pg_idx] + seg_offset
                        for seg_offset in rel_tgt_seg_indices
                    ]
                )

            ex_out[KEYS.SKIP].extend(skip_arr.tolist())
            ex_out[KEYS.WEIGHT].extend(weight_arr.tolist())
        return ex_out

    dataset = align_dataset.map(
        _inner,
        batched=True,
        batch_size=250,
        with_indices=True,
        fn_kwargs={"flip_alignment": flip_alignment},
        remove_columns=align_dataset.column_names,
        load_from_cache_file=load_from_cache_file,
        num_proc=num_proc,
    )

    return dataset


def merge_adjacent_sentences(
    flat_align: HFDataset,
    num_proc: int = 4,
    max_seq_len=None,
    max_merges=10,
    passthrough_prob=0.1,
    seed=1,
):
    def _inner(
        ex_batched,
        abs_align_indices,
        *,
        max_seq_len=1024,
        max_merges=10,
        passthrough_prob=0.1,
        seed,
    ):
        keys = [
            KEYS.DOCUMENT_INDEX,
            KEYS.PARAGRAPH_INDEX,
            KEYS.WEIGHT,
            KEYS.SOURCE_INDICES,
            KEYS.TARGET_INDICES,
            KEYS.SKIP,
        ]
        doc_idxs, pg_idxs, weights, all_src_idxs, all_tgt_idxs, skip = [
            ex_batched[k] for k in keys
        ]

        if KEYS.SOURCE_OFFSETS in ex_batched:
            src_doc_offsets = ex_batched[KEYS.SOURCE_OFFSETS]
            all_src_idxs = [[i + o for i in idxs] for (idxs, o) in zip(all_src_idxs, src_doc_offsets)]
        if KEYS.TARGET_OFFSETS in  ex_batched:
            tgt_doc_offsets = ex_batched[KEYS.TARGET_OFFSETS]
            all_tgt_idxs = [[i + o for i in idxs] for (idxs, o) in zip(all_tgt_idxs, tgt_doc_offsets)]

        ex_out = {
            KEYS.SOURCE_INDICES: [],
            KEYS.TARGET_INDICES: [],
            KEYS.WEIGHT: [],
        }
        # set up reproducible rng state that depends implicitly on batch_size but is invariant to num_proc
        rng = np.random.default_rng((seed, abs_align_indices))
        bin_lengths = np.array([50, 100, 150, 350, max_seq_len], dtype=np.int64)

        # fetch maximum number of rolls to minimize fn calls
        roll_bins = np.clip(
            rng.poisson(_POISSON_MEAN, size=len(doc_idxs)), 0, len(bin_lengths) - 1
        )
        passthrough = (
            rng.random(len(doc_idxs)) > passthrough_prob
            if passthrough_prob is not None
            else np.repeat(True, len(doc_idxs))
        )

        last_doc_idx, last_pg_idx = doc_idxs[0], pg_idxs[0]
        accum_src, accum_tgt, accum_weight = (
            all_src_idxs[0],
            all_tgt_idxs[0],
            weights[0],
        )
        num_outputs, nmerges_in_curr = 0, 1
        for _loop_idx, (
            doc_idx,
            pg_idx,
            weight,
            src_idxs,
            tgt_idxs,
            skip_,
            roll_bin,
        ) in enumerate(
            zip(doc_idxs, pg_idxs, weights, all_src_idxs, all_tgt_idxs, skip, roll_bins)
        ):
            if _loop_idx < 1:
                # skip first, it is already in accumulators
                continue

            skip_ = skip_ or weight > max_seq_len
            rolled_length = bin_lengths[roll_bins[num_outputs]]
            is_boundary = (
                (doc_idx != last_doc_idx)
                or (pg_idx != last_pg_idx)
                or (nmerges_in_curr == max_merges)
            )
            has_budget = accum_weight + weight <= rolled_length

            # if passthrough succeeded store accumulator and reset,
            # passthrough is only performed once at the beginning concatenation_chain
            if nmerges_in_curr == 1 and (not skip_) and passthrough[num_outputs]:
                # ic("passing through")
                ex_out[KEYS.SOURCE_INDICES].append(accum_src)
                ex_out[KEYS.TARGET_INDICES].append(accum_tgt)
                ex_out[KEYS.WEIGHT].append(accum_weight)
                accum_src, accum_tgt, accum_weight = src_idxs, tgt_idxs, weight
                last_doc_idx, last_pg_idx = doc_idx, pg_idx
                num_outputs += 1
                nmerges_in_curr = 1
                continue
            # we did not perform passthrough, perform regular concatenation
            if (not is_boundary) and has_budget and (not skip_):
                accum_src.extend(src_idxs)
                accum_tgt.extend(tgt_idxs)
                accum_weight += weight
                last_doc_idx, last_pg_idx = doc_idx, pg_idx
                nmerges_in_curr += 1
                continue

            num_outputs += 1
            nmerges_in_curr = 1
            # clear accumulators since we exceeded budget, met a boundary or should skip
            # breakpoint()
            if accum_src and accum_tgt:
                # store examples
                ex_out[KEYS.SOURCE_INDICES].append(accum_src)
                ex_out[KEYS.TARGET_INDICES].append(accum_tgt)
                ex_out[KEYS.WEIGHT].append(accum_weight)

            if skip_:
                # discard newest
                accum_src, accum_tgt, accum_weight = [], [], 0
                last_doc_idx, last_pg_idx = None, None
                nmerges_in_curr = 0
            else:
                accum_src, accum_tgt, accum_weight = src_idxs, tgt_idxs, weight
                last_doc_idx, last_pg_idx = doc_idx, pg_idx

        if skip or not accum_src or not accum_tgt or accum_weight == 0:
            # nothing legal to add
            return ex_out

        # there must be something legal to add
        ex_out[KEYS.SOURCE_INDICES].append(accum_src)
        ex_out[KEYS.TARGET_INDICES].append(accum_tgt)
        ex_out[KEYS.WEIGHT].append(accum_weight)
        return ex_out

    assert max_seq_len is not None
    dataset = flat_align.map(
        _inner,
        batched=True,
        with_indices=True,
        batch_size=5_000,
        remove_columns=flat_align.column_names,
        fn_kwargs={"max_seq_len": max_seq_len, "seed": seed, "max_merges": max_merges, "passthrough_prob": passthrough_prob},
        # we never want to load this from cache, since it should be fresh per epoch
        load_from_cache_file=False,
        num_proc=num_proc,
    )
    dataset.set_format("numpy", [KEYS.SOURCE_INDICES, KEYS.TARGET_INDICES, KEYS.WEIGHT])
    return dataset
