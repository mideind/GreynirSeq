import argparse
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List

from tqdm import tqdm

from greynirseq.ner.aligner import get_min_hun_distance
from greynirseq.ner.ner_extracter import NERMarker, parse_line
from greynirseq.ner.nertagger import ner

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NERAlignment:
    """Hold NER alignment results."""

    distance: float
    marker_1: NERMarker
    marker_2: NERMarker

    def __str__(self) -> str:
        return f"{self.marker_1}-{self.marker_2}-{self.distance}"


def read_markers(lines_iter: Iterable[str]) -> List[List[NERMarker]]:
    all_markers = []
    for idx, line in enumerate(lines_iter):
        line_markers = []
        the_split = line.strip().split("\t")
        if len(the_split) <= 1:
            log.debug(f"Bad NERMarker line: {line}")
            continue
        # The model is at pos 0.
        markers = the_split[1:]
        for marker in markers:
            ner_marker = NERMarker.from_line(marker)
            if ner_marker is not None:
                line_markers.append(ner_marker)
            else:
                log.warning(f"Bad NERMarker in line {idx}={line}")
        all_markers.append(line_markers)
    return all_markers


def get_markers(lang: str, lines_iter: Iterable[str]) -> List[List[NERMarker]]:
    tagged_iter = ner(lang=lang, lines_iter=lines_iter)
    return [parse_line(sentence=sentence, labels=labels, model=model) for sentence, labels, model in tagged_iter]


def get_markers_stats(ner_markers: List[List[NERMarker]]) -> Counter:
    return Counter(marker.tag for line_markers in ner_markers for marker in line_markers)


def align_markers(ner_markers_1: List[NERMarker], ner_markers_2: List[NERMarker]) -> List[NERAlignment]:
    min_dist, hits = get_min_hun_distance(
        [ner_marker.named_entity for ner_marker in ner_markers_1],
        [ner_marker.named_entity for ner_marker in ner_markers_2],
    )
    return [NERAlignment(cost, ner_markers_1[hit_1], ner_markers_2[hit_2]) for hit_1, hit_2, cost in hits]


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref")
    parser.add_argument("--ref_markers")
    parser.add_argument("--sys")
    parser.add_argument("--sys_markers")
    parser.add_argument("--tgt_lang", choices=["is", "en"])
    args = parser.parse_args()
    if not (args.ref_markers and args.sys_markers):
        with open(args.ref) as f_ref, open(args.sys) as f_sys:
            ref_markers = get_markers(lang=args.tgt_lang, lines_iter=tqdm(f_ref.readlines()))
            sys_markers = get_markers(lang=args.tgt_lang, lines_iter=tqdm(f_sys.readlines()))
    else:
        with open(args.ref_markers) as f_ref, open(args.sys_markers) as f_sys:
            ref_markers = read_markers(lines_iter=tqdm(f_ref.readlines()))
            sys_markers = read_markers(lines_iter=tqdm(f_sys.readlines()))
    log.info(f"Ref NER markers: {get_markers_stats(ref_markers)}")
    log.info(f"Sys NER markers: {get_markers_stats(sys_markers)}")
    alignments = [align_markers(ref_marker, sys_marker) for ref_marker, sys_marker in zip(ref_markers, sys_markers)]
    dists = [alignment.distance for line_alignment in alignments for alignment in line_alignment]
    log.info(f"Aligment count: {len(dists)}, avg_dist:{sum(dists)/len(dists)}")


if __name__ == "__main__":
    main()
