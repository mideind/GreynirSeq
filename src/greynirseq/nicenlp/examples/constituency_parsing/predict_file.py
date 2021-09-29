# flake8: noqa

import time
import os
from pathlib import Path
import sys

import numpy as np
import torch

import greynirseq.nicenlp.utils.constituency.tree_dist as tree_dist
from greynirseq.nicenlp.utils.constituency.token_utils import tokenize
from greynirseq.nicenlp.criterions.parser_criterion import compute_parse_stats, safe_div, f1_score
from greynirseq.nicenlp.models.simple_parser import SimpleParserModel
from greynirseq.nicenlp.utils.constituency.greynir_utils import Node, NonterminalNode, TerminalNode
from greynirseq.nicenlp.utils.label_schema.label_schema import make_dict_idx_to_vec_idx


def chunk(iterable, batch_size=10):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def main(ckpt_path, input_path, max_sentences=10, output_stream=None):
    assert ckpt_path.exists()
    assert input_path.exists()

    model_interface = SimpleParserModel.from_pretrained(  # pylint: disable=undefined-variable
        ckpt_path.parent,
        checkpoint_file=ckpt_path.name,
        data_name_or_path=args.data,
        gpt2_encoder_json="/model/data/icebert-bpe-vocab.json",
        gpt2_vocab_bpe="/model/data/icebert-bpe-merges.txt",
        nonterm_schema="/model/data/nonterm_schema.json",
    )
    with open(args.input_path, "r") as fh_in:
        fh_in = map(lambda x: x.strip(), fh_in)
        for sentence_batch in chunk(fh_in, batch_size=max_sentences):
            pred_trees, _ = model_interface.predict(sentence_batch)
            for pred_tree in pred_trees:
                pred_tree = pred_tree.separate_unary()
                pred_tree.pretty_print(stream=output_stream)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Description")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--input-path", type=str, required=False)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--max-sentences", type=int, required=False, default=32)
    parser.add_argument("--output-path", type=str, required=False, default="-")

    args = parser.parse_args()

    assert args.max_sentences > 0
    if args.output_path and args.output_path == "-":
        main(
            Path(args.checkpoint), Path(args.input_path), max_sentences=args.max_sentences, output_stream=sys.stdout
        )
    else:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(exist_ok=True)
        with output_path.open("w", encoding="utf8") as fh_out:
            main(Path(args.checkpoint), Path(args.input_path), max_sentences=args.max_sentences, output_stream=fh_out)
