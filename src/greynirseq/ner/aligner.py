"""Align NER tags (with enumeration) in a parallel corpus."""
from __future__ import annotations

import argparse
import logging
import os
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Generator, Iterable, List, Optional, Tuple

import spacy
import tqdm
from pyjarowinkler import distance
from scipy.optimize import linear_sum_assignment
from spacy.gold import offsets_from_biluo_tags

nlp = spacy.load("en_core_web_lg")
log = logging.getLogger(__name__)

NULL_TAG = "O"


@dataclass
class NERMarkerIdx:
    """Hold a NER marker."""

    start_idx: int  # Can be token index (is, hf) or character index (sp)
    end_idx: int
    tag: str

    def __str__(self) -> str:
        return f"{self.start_idx}:{self.end_idx}:{self.tag}"


@dataclass
class NERMarker(NERMarkerIdx):
    """Hold a NER marker along with the NEs."""

    named_entity: str

    @staticmethod
    def from_idx(ner_marker_idx: NERMarkerIdx, named_entity: str):
        """Create from idx."""
        return NERMarker(**asdict(ner_marker_idx), named_entity=named_entity)


@dataclass(frozen=True)
class NERAlignment:
    """Hold NER alignment results."""

    distance: float
    marker_1: NERMarker
    marker_2: NERMarker

    def __str__(self) -> str:
        return f"{self.marker_1.start_idx}:{self.marker_1.end_idx}:{self.marker_1.tag}-{self.marker_2.start_idx}:{self.marker_2.end_idx}:{self.marker_2.tag}"  # noqa


@dataclass(frozen=True)
class PairInfo:
    """Hold NERAlignment information over sentences."""

    per_tags_1: List[NERMarker]  # The NE tags
    per_tags_2: List[NERMarker]
    distance: float  # Total distance
    origin_1: str  # Corpus
    origin_2: str
    pair_map: List[NERAlignment]  # The relations which give the total distance


class NERAnalyser:
    """Analyser for NER parsing."""

    # parallel datasets
    # TODO: accept as parameters
    DATASETS = "emea os2018 bible medical ees tatoeba".split()
    DATA_PATHS_IS = ["/data/datasets/{0}/clean/train/{0}.en-is.train.is".format(dataset) for dataset in DATASETS]
    DATA_PATHS_EN = ["/data/datasets/{0}/clean/train/{0}.en-is.train.en".format(dataset) for dataset in DATASETS]
    DATA_PATHS = DATA_PATHS_IS + DATA_PATHS_EN
    # For cleaning
    PUNCTUATION_SYMBOLS = (
        "'ʼ∞¥≈€∂‧Ω÷‐℉†℃‛″£™∙§«»@¯^!½³²˜−{$¼¹≠}º‗®‑#¡´&`|·≥―′¿<≤~?±" '…\\>”_+][°–=*"‘%„“;:-•(),…–`-—!’?;“”:.,'
    )
    DIGIT_PROG = re.compile(r"\d+")
    PUNCT_PROG = re.compile("[" + re.escape(PUNCTUATION_SYMBOLS) + "]")
    SPACE_PROG = re.compile(r"\s+")

    # For statistics
    stats = {
        dataset: {
            "sum_per_1": 0,
            "sum_per_2": 0,
            "number_lines": 0,
            "number_lines_w_per": 0,
            "number_lines_w_per_match": 0,
            "number_lines_w_mper": 0,
            "number_lines_w_per_mism": 0,
            "sum_dist": 0,
        }
        for dataset in DATASETS + ["mixup"]
    }

    def __init__(self) -> None:
        """Initialize counters and stuff."""
        self.provenance_sets = {path: set() for path in self.DATA_PATHS}
        self.ner_hist_1 = defaultdict(int)
        self.ner_hist_2 = defaultdict(int)
        self.ner_pair_hist = defaultdict(int)

    @staticmethod
    def preprocess_sentence(sentence: str) -> str:
        """Clean sentence.

        Lowercase.
        Replace multiple digits with 0.
        Remove all punctuation.
        Remove all multiple spaces.
        """
        sentence = sentence.lower()
        sentence = NERAnalyser.DIGIT_PROG.sub("0", sentence)
        sentence = NERAnalyser.PUNCT_PROG.sub("", sentence)
        sentence = NERAnalyser.SPACE_PROG.sub("", sentence)
        return sentence

    def load_provenance(self):
        """Load the provenance set, i.e. for each dataset we clean the sentences and store uniques."""
        log.info("Loading provenance sets.")
        for path in self.provenance_sets:
            if not os.path.exists(path):
                log.warning(f"{path} does not exist. Provenance incorrectly configured.")
                continue
            with open(path) as fp:
                for line in fp.readlines():
                    self.provenance_sets[path].add(self.preprocess_sentence(line))

    def check_provenance(self, sentence: str) -> List[str]:
        """Return a list datasets the cleaned sentence is present in."""
        hits = []
        cleaned_sentence = self.preprocess_sentence(sentence)
        for path in self.provenance_sets:
            if cleaned_sentence in self.provenance_sets[path]:
                hits.append(path.split("/")[-1].split(".")[0])
        return hits

    def update_stats(self, pair_info: PairInfo):
        origin = pair_info.origin_1
        if pair_info.origin_1 != pair_info.origin_2:
            origin = "mixup"
        # The most frequent NEs
        for pair in pair_info.pair_map:
            self.ner_pair_hist[
                "{}\t{}\t{}\t{}".format(
                    pair.marker_1.named_entity,
                    pair.marker_2.named_entity,
                    pair.marker_1.named_entity == pair.marker_2.named_entity,
                    pair.distance,
                )
            ] += 1
        for per in pair_info.per_tags_1:
            self.ner_hist_1[per.named_entity] += 1
        for per in pair_info.per_tags_2:
            self.ner_hist_2[per.named_entity] += 1
        self.stats[origin]["sum_per_1"] += len(pair_info.per_tags_1)
        self.stats[origin]["sum_per_2"] += len(pair_info.per_tags_2)
        self.stats[origin]["number_lines"] += 1
        if len(pair_info.per_tags_1) and len(pair_info.per_tags_2):
            self.stats[origin]["number_lines_w_per"] += 1
            if len(pair_info.per_tags_1) == len(pair_info.per_tags_2):
                self.stats[origin]["number_lines_w_per_match"] += 1
        if len(pair_info.per_tags_1) > 1 and len(pair_info.per_tags_2) > 1:
            self.stats[origin]["number_lines_w_mper"] += 1
        if len(pair_info.per_tags_1) != len(pair_info.per_tags_2):
            self.stats[origin]["number_lines_w_per_mism"] += 1
        self.stats[origin]["sum_dist"] = pair_info.distance  # type: ignore

    def print_stats(self):
        """Print statistics."""
        tbl_string = "{:>13}   {:>10}    {:>10}    {:>10}    {:>10}    {:>10}    {:>10}"
        tbl_num_string = "{:>13}   {:>10}    {:>10}    {:>10}    {:>10}    {:>10}    {:>10.6f}"
        print(tbl_string.format("Origin", "Lines", "LWPers", "LWMultiPers", "Mism", "LWPersMatch", "Avg.Dist"))
        for origin in self.stats:
            st = self.stats[origin]
            print(
                tbl_num_string.format(
                    origin,
                    st["number_lines"],  # Actual number of lines
                    st["number_lines_w_per"],  # Person on both sides
                    st["number_lines_w_mper"],  # Multiple persons on both sideds
                    st["number_lines_w_per_mism"],  # Actual not same count, but includes,
                    st["number_lines_w_per_match"],
                    st["sum_dist"] / max(st["number_lines_w_per"], 1),  # Avg dist
                )
            )

    def _write_ner_hist(self, hist_file, data):
        with open(hist_file, "w") as of:
            for k, v in {k: v for k, v in sorted(data.items(), key=lambda item: -item[1])}.items():
                of.writelines("{}\t{}\n".format(k, v))

    def write_ner_hist(self, hist_file1, hist_file2, hist_file_pair):
        self._write_ner_hist(hist_file1, self.ner_hist_1)
        self._write_ner_hist(hist_file2, self.ner_hist_2)
        self._write_ner_hist(hist_file_pair, self.ner_pair_hist)


@dataclass(frozen=True)
class NERSentenceParse:
    """Hold a sentence with parsed NER."""

    DEFAULT_MODEL = "is"

    sent: str
    tags: List[NERMarkerIdx]
    model: str
    origins: List[str]

    @staticmethod
    def parse_is(ner: List[str]) -> List[NERMarkerIdx]:
        """Parse a NER tagged Icelandic sentence."""
        # Always B at beginning, then I for following (Icelandic)
        # TODO: check if same approach as for spacy works
        result = []
        found_tag = None
        tag_len = 0
        for i in range(len(ner)):
            tag = ner[i]
            head, tail = split_tag(tag)
            if head == "I" and tag_len != 0:
                # Some issue here, sometimes wrong follow up... But we let it slide.
                tag_len += 1
            if head == "B" or (tag_len == 0 and head == "I"):
                if found_tag is not None:
                    result.append(NERMarkerIdx(i - tag_len, i, found_tag))
                found_tag = tail
                tag_len = 1
            if head == "O":
                if found_tag is not None:
                    result.append(NERMarkerIdx(i - tag_len, i, found_tag))
                found_tag = None
                tag_len = 0
        return result

    @staticmethod
    def parse_hf(ner: List[str]) -> List[NERMarkerIdx]:
        """Parse a NER tagged sentence based on Huggingface."""
        # Only B if there is an I in front (Huggingface)
        result = []
        found_tag = None
        tag_len = 0
        for i in range(len(ner)):
            tag = ner[i]
            head, tail = split_tag(tag)
            if head == "I" and found_tag is not None and tail == found_tag:
                tag_len += 1
            if head == "I" and found_tag is None:
                found_tag = tail
                tag_len = 1
            if head == "B" or (head == "I" and tail != found_tag):
                if found_tag is not None:
                    # Seldom fails, very seldom but is needed
                    result.append(NERMarkerIdx(i - tag_len, i, found_tag))
                found_tag = tail
                tag_len = 1
            if head == "O":
                if found_tag is not None:
                    result.append(NERMarkerIdx(i - tag_len, i, found_tag))
                found_tag = None
                tag_len = 0
        return result

    @staticmethod
    def parse_sp(ner: List[str], sentence: str) -> List[NERMarkerIdx]:
        """Parse a NER tagged sentence based on Spacy (used for longer sentences)."""
        # U for unique, i.e. no B or I if single etc (Spacy)
        doc = nlp(sentence)
        return [NERMarkerIdx(*offset) for offset in offsets_from_biluo_tags(doc, ner)]

    @staticmethod
    def parse_line(
        line: str, provenance: Optional[NERAnalyser]  # pylint: disable=unsubscriptable-object
    ) -> NERSentenceParse:
        r"""Parse a line.

        Args:
            line: A NER tagged line; Sentence \t NER tags [optional \t model]
        Returns:
            A tuple containing the sentence (str), the NER markers, model, dataset origins,
        """
        sp = line.strip().split("\t")

        sentence, ner_tags = sp[:2]
        assert len(sentence.split()) == len(
            ner_tags.split()
        ), "The tokenized sentence should contain as many NER tags as tokens."

        mode = NERSentenceParse.DEFAULT_MODEL  # Assume Icelandic
        if len(sp) > 2:
            mode = sp[2]

        if mode == "is":
            ner_markers = NERSentenceParse.parse_is(ner_tags.split())
        elif mode == "hf":
            ner_markers = NERSentenceParse.parse_hf(ner_tags.split())
        elif mode == "sp":
            ner_markers = NERSentenceParse.parse_sp(ner_tags.split(), sentence)
        else:
            raise ValueError("Unkown mode={mode}")

        if provenance is not None:
            origins = provenance.check_provenance(sentence)
        else:
            origins = ["unknown"]

        return NERSentenceParse(sentence, ner_markers, mode, origins)


def split_tag(tag: str) -> Tuple[str, Optional[str]]:  # pylint: disable=unsubscriptable-object
    """Split a NER tag to HEAD, TAIL."""
    if tag == NULL_TAG:
        return tag, None
    else:
        return tuple(tag.split("-"))


def get_min_hun_distance(words1: List[str], words2: List[str]) -> Tuple[float, List[Tuple[int, int, float]]]:
    """Calculate a similarity score between all pairs of words."""
    values = []
    hits = []
    min_dist = 0
    for i in range(len(words1)):
        w1 = words1[i]
        row = []
        for j in range(len(words2)):
            w2 = words2[j]
            # Jaro-Winkler distance (not similarity score)
            row.append(1 - distance.get_jaro_distance(w1, w2, winkler=True, scaling=0.1))
        values.append(row)
    # Calculate the best pairing based on the similarity score.
    row_ids, col_ids = linear_sum_assignment(values)
    row_ids = list(row_ids)
    col_ids = list(col_ids)
    # The best alignment
    hits = []
    valsum = 0
    for i in range(len(row_ids)):
        row_id = row_ids[i]
        col_id = col_ids[i]
        hits.append((row_id, col_id, values[row_id][col_id]))
        valsum += values[row_id][col_id]

    min_dist = valsum / (len(words1) + len(words2))

    return min_dist, hits


class NERParser:
    """Parses parallel sentences which have been NER tagged."""

    print_data_file = None

    default_mode = "is"

    def __init__(self, lang_1: Iterable[str], lang_2: Iterable[str]):
        """Initialize it.

        Args:
            lang_1: Iterable in which each element is a sentence.
            lang_2: Iterable in which each element is a sentence.
        """
        self.lang_1: Iterable[str] = lang_1
        self.lang_2: Iterable[str] = lang_2

    def print_line(self, p1: NERSentenceParse, p2: NERSentenceParse, pair_info: PairInfo, file_name: str):
        """Print the alignment information."""
        if self.print_data_file is None:
            self.print_data_file = open(file_name, "w")

        # parser1, origin1, parser2, origin2, match_cat, alignment

        num_tags = len(pair_info.per_tags_1) + len(pair_info.per_tags_2)

        if num_tags and len(pair_info.per_tags_1) == len(pair_info.per_tags_2):
            match = 1
        elif num_tags:
            match = 2
        else:
            match = 0

        alignments = []
        max_dist = ""
        for pair in pair_info.pair_map:
            alignments.append(f"{pair}")

        if alignments:
            max_dist = max([p.distance for p in pair_info.pair_map])

        output = "{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(
            p1.model, ",".join(p1.origins), p2.model, ",".join(p2.origins), match, max_dist, " ".join(alignments)
        )
        self.print_data_file.writelines(output)

    def parse_files(self, print_data: str, analyser: Optional[NERAnalyser]):  # pylint: disable=unsubscriptable-object
        """Parse all the files provided and print."""
        for p1, p2, pair_info in self.parse_files_gen(analyser):
            self.print_line(p1, p2, pair_info, print_data)

    def parse_files_gen(
        self, analyser: Optional[NERAnalyser]  # pylint: disable=unsubscriptable-object
    ) -> Generator[Tuple[NERSentenceParse, NERSentenceParse, PairInfo], None, None]:
        """Parse all the files provided."""
        for l1, l2 in tqdm.tqdm(zip(self.lang_1, self.lang_2)):
            p1 = NERSentenceParse.parse_line(l1, analyser)
            p2 = NERSentenceParse.parse_line(l2, analyser)
            pair_info = self.parse_pair(p1, p2)
            if analyser is not None:
                analyser.update_stats(pair_info)
            yield p1, p2, pair_info

    def parse_pair(self, p1: NERSentenceParse, p2: NERSentenceParse) -> PairInfo:
        try:
            corp1 = p1.origins[0]
            corp2 = p2.origins[0]
        except:  # noqa
            corp1 = corp2 = "mixup"

        # Filter based on the tags we support.
        supported_ner_tags = {"per", "person"}
        ner_markers_idx_1 = [
            ner_marker_idx for ner_marker_idx in p1.tags if ner_marker_idx.tag.lower() in supported_ner_tags
        ]
        ner_markers_idx_2 = [
            ner_marker_idx for ner_marker_idx in p2.tags if ner_marker_idx.tag.lower() in supported_ner_tags
        ]

        # Get the NEs strings (i.e. the actual names)
        if p1.model == "sp":
            # Spacy returns character indices.
            ner_markers_1 = [
                NERMarker.from_idx(t, p1.sent[t.start_idx : t.end_idx].lower()) for t in ner_markers_idx_1  # noqa
            ]
        else:
            # Other models return token indices.
            ner_markers_1 = [
                NERMarker.from_idx(
                    ner_marker_idx,
                    " ".join(p1.sent.split()[ner_marker_idx.start_idx : ner_marker_idx.end_idx]).lower(),  # noqa
                )
                for ner_marker_idx in ner_markers_idx_1
            ]
        ner_markers_2 = [
            NERMarker.from_idx(
                ner_marker_idx,
                " ".join(p2.sent.split()[ner_marker_idx.start_idx : ner_marker_idx.end_idx]).lower(),  # noqa
            )
            for ner_marker_idx in ner_markers_idx_2
        ]

        pair_map: List[NERAlignment] = []

        min_dist = 1
        if ner_markers_1 and ner_markers_2:
            min_dist, hits = get_min_hun_distance(
                [ner_marker.named_entity for ner_marker in ner_markers_1],
                [ner_marker.named_entity for ner_marker in ner_markers_2],
            )
            if hits:
                for hit_1, hit_2, cost in hits:
                    pair_map.append(NERAlignment(cost, ner_markers_1[hit_1], ner_markers_2[hit_2]))
        return PairInfo(ner_markers_1, ner_markers_2, min_dist, corp1, corp2, pair_map)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_ent")
    parser.add_argument("--en_ent")
    parser.add_argument("--output")
    parser.add_argument("--provenance", type=bool, default=False)
    parser.add_argument("--name_histograms", type=bool, default=False)

    args = parser.parse_args()

    # We need better handling for files than just plain "open"
    eval_ner = NERParser(open(args.en_ent), open(args.is_ent))

    provenance = None
    if args.provenance:
        provenance = NERAnalyser()
        provenance.load_provenance()
        # TODO: fix print_stats
        # eval_ner.print_stats()

    eval_ner.parse_files(print_data=args.output, analyser=provenance)

    if args.name_histograms:
        provenance.write_ner_hist("en.hist.ner", "is.hist.ner", "en-is.hist.ner")


if __name__ == "__main__":
    main()
