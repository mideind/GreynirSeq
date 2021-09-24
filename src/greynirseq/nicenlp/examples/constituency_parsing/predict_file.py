# flake8: noqa

import time
from pathlib import Path
import os

import numpy as np
import torch

import greynirseq.nicenlp.utils.constituency.tree_dist as tree_dist
from greynirseq.nicenlp.utils.constituency.token_utils import tokenize
from greynirseq.nicenlp.criterions.parser_criterion import compute_parse_stats, safe_div, f1_score
from greynirseq.nicenlp.models.simple_parser import SimpleParserModel
from greynirseq.nicenlp.utils.constituency.greynir_utils import Node, NonterminalNode, TerminalNode
from greynirseq.nicenlp.utils.label_schema.label_schema import make_dict_idx_to_vec_idx

try:
    from icecream import ic
    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


def chunk(iterable, batch_size=10):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def parse_file(model_interface, input_path):
    with input_path.open("r") as fh_in:
        fh_in = map(lambda x: x.strip(), fh_in)
        for sentence_batch in chunk(fh_in):
            pred_trees, _ = model_interface.predict(sentence_batch)
            for pred_tree in pred_trees:
                # pred_tree.pretty_print()
                pred_tree = pred_tree.separate_unary()
                pred_tree = pred_tree.remove_null_leaves()
                # pred_tree.pretty_print()
                # print()
                # breakpoint()
                # input("Press any key to continue...")
                yield pred_tree


def parse_many(args):
    ckpt_path = Path(args.checkpoint)
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    suffix = args.input_suffix if not args.input_suffix.startswith(".") else args.input_suffix[1:]

    model_interface = SimpleParserModel.from_pretrained(  # pylint: disable=undefined-variable
        ckpt_path.parent,
        checkpoint_file=ckpt_path.name,
        data_name_or_path=args.data,
    )

    print(f"Looking for *.{suffix} files in {input_dir}")
    for input_path in input_dir.glob(f"**/*.{suffix}"):
        output_path = output_dir / input_path.relative_to(input_dir).with_suffix(".psd")
        output_path.parent.mkdir(exist_ok=True)
        is_first = True
        print(f"Parsing {input_path}")
        with output_path.open("w", encoding="utf8") as out_fh:
            for pred_tree in parse_file(model_interface, input_path):
                if is_first:
                    is_first = False
                else:
                    out_fh.write("\n\n")
                out_fh.write(pred_tree.as_nltk_tree().pformat(margin=2**100))


def main(args):
    ckpt_path = Path(args.checkpoint)
    input_dir = Path(args.input_dir)
    model_interface = SimpleParserModel.from_pretrained(  # pylint: disable=undefined-variable
        ckpt_path.parent,
        checkpoint_file=ckpt_path.name,
        data_name_or_path=args.data,
    )
    with open(args.input_path, "r") as fh_in:
        fh_in = map(lambda x: x.strip(), fh_in)
        for sentence_batch in chunk(fh_in):
            pred_trees, _ = model_interface.predict(sentence_batch)
        for pred_tree in pred_trees:
            pred_tree = pred_tree.separate_unary()
            pred_tree.pretty_print()
            breakpoint()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Description")
    parser.add_argument(
        "--data",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--input-path",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--max-sentences",
        type=int,
        required=False,
        default=32,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--input-suffix",
        type=str,
        required=False,
    )

    args = parser.parse_args()
    assert Path(args.checkpoint).exists()
    assert bool(args.input_path) ^ bool(args.input_dir)
    assert (args.output_dir and args.input_dir and args.input_suffix) or args.input_path
    assert (args.input_path and Path(args.input_path).exists()) or (args.input_dir and Path(args.input_dir).is_dir())
    assert Path(args.data).exists()
    parse_many(args)
    # main(args)
