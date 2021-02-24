import argparse
from typing import List

from greynirseq.ner.aligner import (
    NERAnalyser,
    NERMarkerIdx,
    NERParser,
    NERSentenceParse,
    PairInfo,
)
from greynirseq.nicenlp.models.multilabel import MutliLabelRobertaModel
from greynirseq.settings import IceBERT_POS_CONFIG, IceBERT_POS_PATH


def add_marker(ner_marker: NERMarkerIdx, tokens: List[str], idx: int, tag: str):
    """Add a complex marker around NEs.

    HAS SIDE-EFFECTS!
    """
    tokens[ner_marker.start_idx] = f"<e:{idx}:{tag}:>{tokens[ner_marker.start_idx]}"
    if ner_marker.end_idx - ner_marker.start_idx != 1:
        tokens[ner_marker.end_idx - 1] = f"{tokens[ner_marker.end_idx - 1]}</e{idx}>"
    else:
        tokens[ner_marker.start_idx] += f"</e{idx}>"


# TODO accept string literal etc.
def tag_ner_pair(pos_model, p1: NERSentenceParse, p2: NERSentenceParse, pair_info: PairInfo, max_distance=1):
    hit = False
    en_tokens = p1.sent.split()
    is_tokens = p2.sent.split()

    # Assume p2 is Icelandic
    pos_tags = pos_model.predict_to_idf(p2.sent, device="cuda")
    for idx, alignment in enumerate(pair_info.pair_map):
        en_ner_marker, is_ner_marker, distance = alignment.marker_1, alignment.marker_2, alignment.distance
        tags = pos_tags[is_ner_marker.start_idx : is_ner_marker.end_idx]
        if "e" in tags:
            # Since IDF for some reason uses "e" for foreign names, we ignore those
            continue
        if distance > max_distance:
            continue

        hit = True
        # Add a complex tag in front of the NE
        add_marker(en_ner_marker, en_tokens, idx, tags[0])
        add_marker(is_ner_marker, is_tokens, idx, tags[0])

    if hit:
        return en_tokens, is_tokens
    return None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--is_ent")
    parser.add_argument("--en_ent")
    parser.add_argument("--output")
    args = parser.parse_args()

    pos_model = MutliLabelRobertaModel.from_pretrained(IceBERT_POS_PATH, **IceBERT_POS_CONFIG)
    pos_model.to("cuda")
    pos_model.eval()

    eval_ner = NERParser(open(args.en_ent), open(args.is_ent))
    with open(args.output, "w") as ofile:
        provenance = NERAnalyser()
        provenance.load_provenance()
        for p1, p2, pair_info in eval_ner.parse_files_gen(analyser=provenance):
            if pair_info.pair_map:
                en_sent, is_sent = tag_ner_pair(pos_model, p1, p2, pair_info, max_distance=0.9)
                if en_sent is not None and is_sent is not None:
                    ofile.writelines("{}\t{}\n".format(" ".join(en_sent), " ".join(is_sent)))


if __name__ == "__main__":
    main()
