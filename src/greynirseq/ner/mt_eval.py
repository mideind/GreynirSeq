import argparse
import logging
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import chain
from typing import Dict, Iterable, List, Tuple
from unicodedata import decimal

import sacrebleu
from tqdm import tqdm

from greynirseq.ner.aligner import NULL_TAG, get_min_hun_distance
from greynirseq.ner.ner_extracter import (
    ENTITY_MARKERS,
    ENTITY_MARKERS_END,
    ENTITY_MARKERS_START,
    TAGS,
    NERMarker,
    embed_tokens,
    parse_line,
)
from greynirseq.ner.nertagger import ner, tok

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class NERAlignment:
    """Hold NER alignment results."""

    distance: float
    marker_1: NERMarker
    marker_2: NERMarker

    def __str__(self) -> str:
        return f"{self.marker_1}-{self.marker_2}-{self.distance}"


def read_embedded_markers(
    lines_iter: Iterable[str], contains_model_marker=False
) -> Tuple[List[List[NERMarker]], Dict[str, int]]:
    """Read embedded NER markers from a collection of lines.
    NERMarkers contain untokenized token offsets and NULL_TAGS."""
    all_markers = []
    bad_markers = defaultdict(int)
    for idx, line in enumerate(lines_iter):
        correct_markers = []
        sentence = line.strip()
        if contains_model_marker:
            the_split = sentence.split("\t")
            # The model is at pos 0.
            sentence = the_split[1]
        # We cannot tokenize the sentence since it will ruin the tag markers.
        all_sentence_markers = ENTITY_MARKERS.findall(sentence)
        # We try to collect all the markers from their start to end - the input is NOT tokenized
        tokens = sentence.split()
        read_start_tag = False
        token_start_idx = 0
        tag = ""
        named_entity_buffer = []
        for token_idx, token in enumerate(tokens):
            # We are searching for a start tag
            if not read_start_tag:
                start = ENTITY_MARKERS_START.search(token)  # Search only returns a single result
                if start:
                    read_start_tag = True
                    tag = start.group(0)[-2]  # Char at -2 = the tag letter
                    # Remove the start_tag from the token
                    token = token[start.end() :]
                    token_start_idx = token_idx
                # We are not inside a tag and nothing found - leave it be
                assert not named_entity_buffer, "The named entity buffer should be empty"

            # We have read the starting tag, now we search for the closing.
            if read_start_tag:

                # Search for the ending after the start - possibly same token as start
                end = ENTITY_MARKERS_END.search(token)
                # We found an ending tag
                if end:
                    token = token[: end.start()]
                    named_entity_buffer.append(token)
                    # We found a correct ending tag
                    if end.group(0)[-2] == tag:
                        named_entity = " ".join(named_entity_buffer)
                        correct_markers.append(
                            NERMarker(
                                start_idx=token_start_idx,
                                end_idx=token_idx + 1,
                                tag=tag,
                                named_entity=named_entity,
                            )
                        )
                    else:
                        log.warning(f"Unable to find correct closing tag for {tag=} in {sentence=}")
                    # We should stop trying to pair this tag
                    read_start_tag = False
                    # Clear the buffer, since we found an ending tag
                    named_entity_buffer.clear()

                # We have read the start tag, and we have not yet read the end,
                else:
                    named_entity_buffer.append(token)

        # We read a start tag but it never closed.
        if read_start_tag:
            log.warning(f"Unable to find correct closing tag for {tag=} in {sentence=}")
        # Gather results on missing tags
        for marker in correct_markers:
            all_sentence_markers.remove(f"<{marker.tag}>")
            all_sentence_markers.remove(f"</{marker.tag}>")
        for remaining_marker in all_sentence_markers:
            bad_markers[remaining_marker] += 1

        all_markers.append(correct_markers)
    return all_markers, bad_markers


def get_markers(lang: str, lines_iter: Iterable[str], device: str) -> List[List[NERMarker]]:
    toks_iter = tok(lang=lang, lines_iter=lines_iter)
    tagged_iter = ner(lang=lang, lines_iter=toks_iter, device=device)
    return [parse_line(sentence=sentence, labels=labels, model=model) for sentence, labels, model in tagged_iter]


def get_markers_stats(ner_markers: List[List[NERMarker]]) -> Counter:
    return Counter(marker.tag for line_markers in ner_markers for marker in line_markers)


def align_markers(ner_markers_1: List[NERMarker], ner_markers_2: List[NERMarker]) -> List[NERAlignment]:
    try:
        min_dist, hits = get_min_hun_distance(
            [ner_marker.named_entity for ner_marker in ner_markers_1],
            [ner_marker.named_entity for ner_marker in ner_markers_2],
        )
    except ValueError:
        log.exception(f"Bad NER markers: {ner_markers_1=}, {ner_markers_2}")
    return [NERAlignment(cost, ner_markers_1[hit_1], ner_markers_2[hit_2]) for hit_1, hit_2, cost in hits]


def get_gold_and_clean(lines_iter: Iterable[str]):
    ref_gold_markers, bad_markers = read_embedded_markers(lines_iter=lines_iter)
    # The ref should be correctly formatter
    if bad_markers:
        log.info(f"Found unpaired markers={bad_markers}")
    ref_clean = [ENTITY_MARKERS.sub("", line) for line in lines_iter]
    return ref_gold_markers, ref_clean


def get_metrics(alignments: List[List[NERAlignment]], upper_bound_ner_alignments: int) -> Dict[str, float]:
    dists = [alignment.distance for line_alignment in alignments for alignment in line_alignment]
    accuracy = sum(
        [
            alignment.marker_1.named_entity == alignment.marker_2.named_entity
            for line_alignment in alignments
            for alignment in line_alignment
        ]
    ) / len(dists)
    return {
        "Alignment count": len(dists),
        "Alignment coverage": len(dists) / upper_bound_ner_alignments,
        "Average alignment distance": sum(dists) / len(dists),
        "Accuracy (exact match)": accuracy,
    }


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref")
    parser.add_argument("--ref-contains-entities", action="store_true", default=False)
    parser.add_argument("--sys")
    parser.add_argument("--sys-contains-entities", action="store_true", default=False)
    parser.add_argument("--device", help="The device to use for NER tagging (if needed)", default="cuda")
    parser.add_argument("--output_dir", help="The folder to write the outputs to.")
    parser.add_argument("--tgt_lang", choices=["is", "en"])
    args = parser.parse_args()
    with open(args.ref) as f_ref, open(args.sys) as f_sys:
        log.info("Reading files")
        ref = f_ref.readlines()
        sys = f_sys.readlines()
    ref_markers = None
    sys_markers = None
    if args.ref_contains_entities:
        log.info("Reading markers from REF")
        ref_markers, ref_clean = get_gold_and_clean(ref)
    else:
        log.info("NER tagging REF")
        ref_clean = ref
        ref_markers = get_markers(lang=args.tgt_lang, lines_iter=tqdm(ref_clean), device=args.device)
    if args.sys_contains_entities:
        log.info("Reading markers from SYS")
        sys_markers, sys_clean = get_gold_and_clean(sys)
    else:
        log.info("NER tagging SYS")
        sys_clean = sys
        sys_markers = get_markers(lang=args.tgt_lang, lines_iter=tqdm(sys_clean), device=args.device)

    log.info(f"BLEU score: {sacrebleu.corpus_bleu(sys_stream=sys_clean, ref_streams=[ref_clean])}")
    log.info(f"Ref NER markers: {get_markers_stats(ref_markers)}")
    log.info(f"Sys NER markers: {get_markers_stats(sys_markers)}")
    alignments = [align_markers(ref_marker, sys_marker) for ref_marker, sys_marker in zip(ref_markers, sys_markers)]
    if alignments:
        upper_bound_ner_alignments = min(
            sum(len(markers) for markers in ref_markers), sum(len(markers) for markers in sys_markers)
        )
        log.info("Metrics over all types:")
        for metric, value in get_metrics(alignments, upper_bound_ner_alignments).items():
            log.info(f"\t{metric}: {value:.3f}")
        groups = {marker.tag for markers in chain(ref_markers, sys_markers) for marker in markers}
        for group in groups:
            upper_bound_ner_alignments = min(
                sum(1 for markers in ref_markers for marker in markers if marker.tag == group),
                sum(1 for markers in sys_markers for marker in markers if marker.tag == group),
            )
            if upper_bound_ner_alignments:
                # Refs are marker_1
                group_alignments = [
                    [alignment for alignment in s_alignment if alignment.marker_1.tag == group]
                    for s_alignment in alignments
                ]
                log.info(f"Metrics over {group}:")
                for metric, value in get_metrics(group_alignments, upper_bound_ner_alignments).items():
                    log.info(f"\t{metric}: {value:.3f}")

    else:
        log.info("No alignments!")


if __name__ == "__main__":
    main()
