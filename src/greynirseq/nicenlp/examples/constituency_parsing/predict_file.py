#!/usr/bin/env python3

# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.


import sys
from pathlib import Path

from greynirseq.nicenlp.models.simple_parser import SimpleParserModel

_NUM_SENTENCES_IN_BATCH = 10


def chunk(iterable, batch_size=_NUM_SENTENCES_IN_BATCH):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def load_parser(ckpt_path, data_bin, nonterm_schema, merges_txt, vocab_json, cpu=True):
    ckpt_path = Path(ckpt_path)
    assert ckpt_path.exists(), str(ckpt_path)
    assert Path(nonterm_schema).exists()
    assert Path(merges_txt).exists()
    assert Path(vocab_json).exists()
    model_interface = SimpleParserModel.from_pretrained(
        ckpt_path.parent,
        checkpoint_file=ckpt_path.name,
        data_name_or_path=data_bin,
        gpt2_encoder_json=vocab_json,
        gpt2_vocab_bpe=merges_txt,
        nonterm_schema=nonterm_schema,
    )
    if cpu:
        model_interface = model_interface.cpu()
    return model_interface


def main(model_interface, input_path, output_stream=None, extra_line=False):
    assert input_path.exists()

    outputs = []
    with open(args.input_path, "r") as fh_in:
        fh_in = map(lambda x: x.strip(), fh_in)
        for sentence_batch in chunk(fh_in):
            pred_trees, _ = model_interface.predict(sentence_batch)
            for pred_tree in pred_trees:
                pred_tree = pred_tree.separate_unary()
                outputs.append(pred_tree.as_nltk_tree().pformat(margin=2**100).strip())
    sep = "\n\n" if extra_line else "\n"
    output_stream.write(sep.join(outputs))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Parse a text file using a given parser checkpoint")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--input", dest="input_path", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--nonterm-schema", type=str, required=True)
    parser.add_argument("--vocab-json", type=str, required=True)
    parser.add_argument("--merges-txt", type=str, required=True)
    parser.add_argument("--output", dest="output_path", type=str, required=False, default="-")
    parser.add_argument("--empty-line-separated", action="store_true")

    args = parser.parse_args()

    model_interface = load_parser(args.checkpoint, args.data, args.nonterm_schema, args.merges_txt, args.vocab_json)
    if args.output_path == "-":
        main(model_interface, Path(args.input_path), output_stream=sys.stdout, extra_line=args.empty_line_separated)
    else:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(exist_ok=True)
        with output_path.open("w", encoding="utf8") as fh_out:
            main(model_interface, Path(args.input_path), output_stream=fh_out, extra_line=args.empty_line_separated)
