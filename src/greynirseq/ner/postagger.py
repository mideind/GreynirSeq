import argparse
from dataclasses import asdict, dataclass
from typing import List, Optional

from greynirseq.ner.aligner import NERAnalyser, NERMarker, NERParser, NERSentenceParse, PairInfo
from greynirseq.nicenlp.models.multilabel import MultiLabelRobertaHubInterface, MultiLabelRobertaModel
from greynirseq.settings import IceBERT_POS_CONFIG, IceBERT_POS_PATH


@dataclass
class NERPoSMarker(NERMarker):
    # By default we have no PoS tag
    pos_tag: Optional[str] = None  # pylint: disable=unsubscriptable-object

    @staticmethod
    def from_NERMarker(ner_marker: NERMarker):
        return NERPoSMarker(**asdict(ner_marker))


def load_pos_model():
    pos_model = MultiLabelRobertaModel.from_pretrained(IceBERT_POS_PATH, **IceBERT_POS_CONFIG)
    pos_model.to("cuda")
    pos_model.eval()
    return pos_model


def add_marker(marker: NERPoSMarker, tokens: List[str], alignment_num: int, enumerate_marker=False, add_pos_tag=False):
    """Add a simple marker around NEs. Contains no enumeration nor pos tag.

    HAS SIDE-EFFECTS!
    """
    start = "<e"
    end = "</e"
    if enumerate_marker:
        start += f":{alignment_num}"
        end += f":{alignment_num}"
    if add_pos_tag:
        start += f":{marker.pos_tag}"
    start += ">"
    end += ">"
    tokens[marker.start_idx] = f"{start}{tokens[marker.start_idx]}"
    tokens[marker.end_idx - 1] = f"{tokens[marker.end_idx - 1]}{end}"


# TODO accept string literal etc.
def tag_ner_pair(
    pos_model: Optional[MultiLabelRobertaHubInterface],  # pylint: disable=unsubscriptable-object
    p1: NERSentenceParse,
    p2: NERSentenceParse,
    pair_info: PairInfo,
    add_pos_tags=False,
    max_distance=1,
):
    # Alignments within max_distance.
    acceptable_alignments = list(filter(lambda alignment: alignment.distance <= max_distance, pair_info.pair_map))
    en_markers = [NERPoSMarker.from_NERMarker(alignment.marker_1) for alignment in acceptable_alignments]
    is_markers = [NERPoSMarker.from_NERMarker(alignment.marker_2) for alignment in acceptable_alignments]

    hit = False
    en_tokens = p1.sent.split()
    is_tokens = p2.sent.split()

    if add_pos_tags:
        # Assume p2 is Icelandic
        # TODO: parse to IFD tags
        pos_tags, _ = pos_model.predict_labels(p2.sent)
        for is_marker in is_markers:
            pos_tag = pos_tags[is_marker.start_idx]  # We only pick the first tag for the NE
            # Since IDF for some reason uses "e" for foreign names, we ignore those
            if pos_tag == "e":
                continue
            is_marker.pos_tag = pos_tag

    for alignment_num, (en_marker, is_marker) in enumerate(zip(en_markers, is_markers)):
        hit = True
        # Add a complex tag in front of the NE
        add_marker(en_marker, en_tokens, alignment_num)
        add_marker(is_marker, is_tokens, alignment_num)

    if hit:
        return en_tokens, is_tokens
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_ent")
    parser.add_argument("--en_ent")
    parser.add_argument("--output")
    parser.add_argument("--pos_tag", action="store_true")
    args = parser.parse_args()

    pos_model: Optional[MultiLabelRobertaHubInterface] = None  # pylint: disable=unsubscriptable-object
    if args.pos_tag:
        pos_model = load_pos_model()  # type: ignore
    eval_ner = NERParser(open(args.en_ent), open(args.is_ent))
    with open(args.output, "w") as ofile:
        provenance = NERAnalyser()
        provenance.load_provenance()
        for p1, p2, pair_info in eval_ner.parse_files_gen(analyser=provenance):
            if pair_info.pair_map:
                en_sent, is_sent = tag_ner_pair(pos_model, p1, p2, pair_info, args.pos_tag, max_distance=0.9)
                if en_sent is not None and is_sent is not None:
                    ofile.writelines("{}\t{}\n".format(" ".join(en_sent), " ".join(is_sent)))


if __name__ == "__main__":
    main()
