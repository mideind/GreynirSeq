import argparse
import logging
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import sacrebleu
from tqdm import tqdm

from greynirseq.ner.aligner import NULL_TAG, get_min_hun_distance
from greynirseq.ner.ner_extracter import TAGS, NERMarker, embed_tokens, parse_line
from greynirseq.ner.nertagger import ner, tok

log = logging.getLogger(__name__)

ENTITY_MARKERS_START = re.compile(f"<[{'|'.join(TAGS)}]>")
ENTITY_MARKERS_END = re.compile(f"</[{'|'.join(TAGS)}]>")

ENTITY_MARKERS = re.compile(f"<(/)?[{'|'.join(TAGS)}]>")


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
    """Read embedded NER markers from a collection of lines. NERMarkers contain untokenized token offsets and NULL_TAGS."""
    all_markers = []
    bad_markers = {f"<{tag}>": 0 for tag in TAGS}
    bad_markers.update({f"</{tag}>": 0 for tag in TAGS})
    for idx, line in enumerate(lines_iter):
        correct_markers = []
        sentence = line.strip()
        if contains_model_marker:
            the_split = sentence.split("\t")
            # The model is at pos 0.
            sentence = the_split[1]
        # We cannot tokenize the sentence since it will ruin the tag markers.
        all_markers = ENTITY_MARKERS.findall(sentence)
        # We try to collect all the markers from their start to end
        tokens = sentence.split()
        read_start_tag = False
        start_idx = 0
        tag = None
        for token_idx, token in enumerate(tokens):
            # We are searching for a start tag
            if not read_start_tag:
                start = ENTITY_MARKERS_START.search(token)  # Search only returns a single result
                if start:
                    read_start_tag = True
                    tag = start.group(0)[-2]
                    start_idx = token_idx
                # We are not inside a tag and nothing found - mark as NULL_TAG
                else:
                    correct_markers.append(
                        NERMarker(start_idx=token_idx, end_idx=token_idx + 1, tag=NULL_TAG, named_entity="")
                    )

            # Search for the ending after the start - possibly same token as start
            if read_start_tag:
                # We ignore starting tags at this stage
                end = ENTITY_MARKERS_END.search(token)
                # We found an ending tag
                if end:
                    # We found a correct ending tag
                    if end.group(0)[-2] == tag:
                        correct_markers.append(
                            NERMarker(
                                start_idx=start_idx,
                                end_idx=token_idx + 1,
                                tag=tag,
                                named_entity=" ".join(tokens[start_idx : token_idx + 1]),
                            )
                        )
                    else:
                        log.warning(f"Unable to find correct closing tag for {tag=} in {sentence=}")
                        # We stop searching for the closing that
                        read_start_tag = False
                        # And mark all the tokens we read as NULL_TAG
                        for i in range(start_idx, token_idx + 1):
                            correct_markers.append(NERMarker(start_idx=i, end_idx=i + 1, tag=NULL_TAG, named_entity=""))
        # We read a start tag but it never closed.
        if read_start_tag:
            for i in range(start_idx, len(tokens) + 1):
                correct_markers.append(NERMarker(start_idx=i, end_idx=i + 1, tag=NULL_TAG, named_entity=""))
        # Gather results on missing tags
        for marker in correct_markers:
            if marker.tag == NULL_TAG:
                continue
            all_markers.remove(f"<{marker.tag}>")
            all_markers.remove(f"</{marker.tag}>")
        for remaining_marker in all_markers:
            bad_markers[remaining_marker] += 1

        all_markers.append(correct_markers)
    return all_markers, bad_markers


def get_markers(lang: str, lines_iter: Iterable[str]) -> List[List[NERMarker]]:
    tagged_iter = ner(lang=lang, lines_iter=lines_iter)
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


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref")
    parser.add_argument("--sys")
    parser.add_argument(
        "--entity_preserving", action="store_true", default=False, help="Does the sys/ref contain entity tags?"
    )
    parser.add_argument("--output_dir", help="The folder to write the outputs to.")
    parser.add_argument("--tgt_lang", choices=["is", "en"])
    args = parser.parse_args()
    with open(args.ref) as f_ref, open(args.sys) as f_sys:
        log.info("Reading files")
        ref = f_ref.readlines()
        sys = f_sys.readlines()
    ref_gold_markers = None
    sys_pred_markers = None
    if args.entity_preserving:
        ref_gold_markers, bad_markers = read_embedded_markers(ref)
        # The ref should be correctly formatter
        assert len(bad_markers) == 0
        sys_pred_markers, bad_markers = read_embedded_markers(sys)
        log.info(f"Read markers from sys. Bad markers={bad_markers}")
        log.info("Removing entity tags from sys and writing it")
        ref_clean = [ENTITY_MARKERS.sub("", line) for line in ref]
        sys_clean = [ENTITY_MARKERS.sub("", line) for line in ref]
        with open(f"{args.output_dir}/ref.clean.{args.tgt_lang}", "w") as f_out:
            f_out.writelines(ref_clean)
        with open(f"{args.output_dir}/sys.clean.{args.tgt_lang}", "w") as f_out:
            f_out.writelines(sys_clean)
    else:
        ref_clean = ref
        sys_clean = sys
    log.info(f"BLEU score: {sacrebleu.corpus_bleu(sys_stream=sys_clean, ref_streams=[ref_clean])}")
    log.info("Finding NER markers")
    ref_clean_tok = list(tok(args.tgt_lang, ref_clean))
    sys_clean_tok = list(tok(args.tgt_lang, sys_clean))
    ref_markers = get_markers(lang=args.tgt_lang, lines_iter=tqdm(ref_clean_tok))
    sys_markers = get_markers(lang=args.tgt_lang, lines_iter=tqdm(sys_clean_tok))
    log.info("Writing NER markers")
    write_ner_markers(f"{args.output_dir}/ref.ner.{args.tgt_lang}", ref_clean_tok, ref_markers)
    write_ner_markers(f"{args.output_dir}/sys.ner.{args.tgt_lang}", sys_clean_tok, sys_markers)
    log.info(f"Ref NER markers: {get_markers_stats(ref_markers)}")
    log.info(f"Sys NER markers: {get_markers_stats(sys_markers)}")
    upper_bound_ner_alignments = min(
        sum(len(markers) for markers in ref_markers), sum(len(markers) for markers in sys_markers)
    )
    # TODO: Write alignments to file.
    alignments = [align_markers(ref_marker, sys_marker) for ref_marker, sys_marker in zip(ref_markers, sys_markers)]
    dists = [alignment.distance for line_alignment in alignments for alignment in line_alignment]
    log.info(
        f"Aligment count: {len(dists)}, aligment_coverage={len(dists)/upper_bound_ner_alignments}, avg_dist:{sum(dists)/len(dists)}"
    )
    dist_per_pair_cat
    for alignment in alignments:
        pass
    # Accuracy of ref ner markers vs sys_clean_tok
    # dist per tag flokk (per pörun)


def write_ner_markers(path: str, ref, ref_markers):
    with open(path, "w") as f_out:
        for sent, ref_markers_line in zip(ref, ref_markers):
            f_out.write(" ".join(embed_tokens(ref_markers_line, sent.split())) + "\n")


if __name__ == "__main__":
    main()
