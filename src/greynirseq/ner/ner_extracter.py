"""Extract NEs."""
import argparse
import logging
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

from tqdm import tqdm

from greynirseq.ner.nertagger import detok

log = logging.getLogger(__name__)

NULL_TAG = "O"
# Uses BIO and all entities start with B.
PER = "P"
LOC = "L"
ORG = "O"
MISC = "M"
DATE = "D"
TIME = "T"
MON = "$"
PERC = "%"
TAGS = {PER, LOC, ORG, MISC, DATE, TIME, MON, PERC}
BIO_MAPPER = {NULL_TAG: NULL_TAG, "B": "B", "I": "I", "U": "B", "L": "I"}
IS_TAGS = {
    "Person": PER,
    "Location": LOC,
    "Organization": ORG,
    "Miscellaneous": MISC,
    "Date": DATE,
    "Time": TIME,
    "Money": MON,
    "Percent": PERC,
}
HF_TAGS = {
    "MISC": MISC,
    "PER": PER,
    "ORG": ORG,
    "LOC": LOC,
}
SP_TAGS = {
    "CARDINAL": MISC,
    "GPE": ORG,
    "ORG": ORG,
    "PERSON": PER,
    "DATE": DATE,
    "EVENT": MISC,
    "FAC": MISC,  # ?
    "LANGUAGE": MISC,
    "LAW": MISC,
    "LOC": LOC,
    "MONEY": MON,
    "NORP": MISC,
    "ORDINAL": MISC,
    "PERCENT": PERC,
    "PRODUCT": MISC,
    "QUANTITY": MISC,
    "TIME": TIME,
    "WORK_OF_ART": MISC,
}
TAG_MAPPER = {"is": IS_TAGS, "hf": HF_TAGS, "sp": SP_TAGS}

ENTITY_MARKERS_START = re.compile(f"<[{'|'.join(TAGS)}]+>")
ENTITY_MARKERS_END = re.compile(f"</[{'|'.join(TAGS)}]+>")

ENTITY_MARKERS = re.compile(f"</?[{'|'.join(TAGS)}]+>")


@dataclass
class NERMarker:
    """Hold a NER marker"""

    start_idx: int  # Can be token index (is, hf) or character index (sp)
    end_idx: int
    tag: str
    named_entity: str

    def __str__(self) -> str:
        return f"{self.start_idx}:{self.end_idx}:{self.tag}:{self.named_entity}"

    @staticmethod
    def from_line(line: str):
        the_split = line.split(":")
        if len(the_split) != 4:
            return None
        start_idx, end_idx, tag, named_entity = the_split
        return NERMarker(int(start_idx), int(end_idx), tag, named_entity)


def split_tag(tag: str) -> Tuple[str, Optional[str]]:  # pylint: disable=unsubscriptable-object
    """Split a NER tag to HEAD, TAIL."""
    tag_s = tag.split("-")
    if len(tag_s) == 1:
        return tag_s[0], None
    else:
        return tag_s[0], tag_s[1]


def parse_line(sentence: List[str], labels: List[str], model: str) -> List[NERMarker]:
    """Parses a single line into NERMarkers."""
    result: List[NERMarker] = []
    found_tag = None
    tag_len = 0
    i = 0  # Just to avoid error below for-loop
    for i in range(len(labels)):
        current_tag = labels[i]
        head, tail = split_tag(current_tag)
        head = BIO_MAPPER[head]
        if head == NULL_TAG:
            # We read a null tag and nothing in buffer -> Continue
            if found_tag is None:
                continue
            # We need to add the last tag into the result
            else:
                result.append(NERMarker(i - tag_len, i, found_tag, " ".join(sentence[i - tag_len : i])))
            found_tag = None
            tag_len = 0
        elif tail is None:
            log.error(f"Incorrect tag={current_tag} in sentence={' '.join(sentence)} with tags={' '.join(labels)}")
        # I can be at the beginning or in the middle of a NE, handle both.
        elif head == "I":
            # We only store the last seen tag.
            found_tag = TAG_MAPPER[model][tail]
            tag_len += 1
        # B means a start of a new NE.
        elif head == "B":
            # We check if our buffer is empty.
            if found_tag is not None:
                result.append(NERMarker(i - tag_len, i, found_tag, " ".join(sentence[i - tag_len : i])))
            found_tag = TAG_MAPPER[model][tail]
            tag_len = 1
        # error
        else:
            log.error(f"Read a bad BIO-head={head} in sentence={' '.join(sentence)} with tags={' '.join(labels)}")
    if found_tag is not None:
        result.append(NERMarker(i - tag_len, i, found_tag, " ".join(sentence[i - tag_len : i])))
    return result


def embed_tokens_to_list(markers: List[NERMarker], tokens: List[str]) -> List[str]:
    """Embed NER marker around NEs."""
    tokens = tokens[:]
    for marker in markers:
        try:
            tokens[marker.start_idx] = f"<{marker.tag}>{tokens[marker.start_idx]}"
            tokens[marker.end_idx - 1] = f"{tokens[marker.end_idx - 1]}</{marker.tag}>"
        except IndexError as e:
            log.exception(f"Unable to wrap {tokens=} with {marker=}")
            raise e
    return tokens


def embed_tokens(markers: List[NERMarker], tokens: List[str]) -> str:
    embedded_token_list = embed_tokens_to_list(markers, tokens)
    # We use correct_spaces for both English and Icelandic
    somewhat_correct_spaces = list(detok([" ".join(embedded_token_list)]))[0]
    # correct_spaces adds spaces around the ner_markers which we do not want.
    found = 0
    for match in ENTITY_MARKERS_START.finditer(somewhat_correct_spaces):
        # Skip the space
        somewhat_correct_spaces = (
            somewhat_correct_spaces[: match.end() - found] + somewhat_correct_spaces[match.end() - found + 1 :]
        )
        found += 1
    found = 0
    for match in ENTITY_MARKERS_END.finditer(somewhat_correct_spaces):
        # Skip the space
        somewhat_correct_spaces = (
            somewhat_correct_spaces[: match.start() - found - 1] + somewhat_correct_spaces[match.start() - found :]
        )
        found += 1
    return somewhat_correct_spaces


def main():
    """Extract NEs given the tokenized sentence, labels and tagger information. Maps the tags to a unified format."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", nargs="?", type=argparse.FileType("r"), default=sys.stdin)
    parser.add_argument("--output", nargs="?", type=argparse.FileType("w"), default=sys.stdout)
    parser.add_argument(
        "--embed_tags_detok",
        default=False,
        action="store_true",
        help="Should we write (embed) the NER tags around the tokens and detokenize?",
    )

    args = parser.parse_args()
    stats = defaultdict(int)
    f_in = args.input
    f_out = args.output
    for line in tqdm(f_in):
        sent, labels, model = line.strip().split("\t")
        ner_markers = parse_line(sent.split(), labels.split(), model)
        for ner_marker in ner_markers:
            stats[ner_marker.tag] += 1
        if not args.embed_tags_detok:
            f_out.write("\t".join([model] + [str(ner_marker) for ner_marker in ner_markers]) + "\n")
        else:
            f_out.write(f"{embed_tokens(ner_markers, sent.split())}\n")

    stats = dict(stats)
    stats = sorted(stats.items())
    print(f"Labels of '{args.input.name}': {stats}")


if __name__ == "__main__":
    main()
