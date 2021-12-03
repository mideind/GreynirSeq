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


def main(ckpt_path, input_path, vocabulary_dir, output_stream=None):
    assert ckpt_path.exists()
    assert input_path.exists()

    # label_schema is also assumed to be in vocabulary_dir as well
    model_interface = SimpleParserModel.from_pretrained(  # pylint: disable=undefined-variable
        ckpt_path.parent,
        checkpoint_file=ckpt_path.name,
        data_name_or_path=args.data,
        gpt2_encoder_json=str(vocabulary_dir / "icebert-bpe-vocab.json"),
        gpt2_vocab_bpe=str(vocabulary_dir / "icebert-bpe-merges.txt"),
        nonterm_schema=str(vocabulary_dir / "nonterm_schema.json"),
    )
    with open(args.input_path, "r") as fh_in:
        fh_in = map(lambda x: x.strip(), fh_in)
        for sentence_batch in chunk(fh_in):
            pred_trees, _ = model_interface.predict(sentence_batch)
            for pred_tree in pred_trees:
                pred_tree = pred_tree.separate_unary()
                output_stream.write(pred_tree.as_nltk_tree().pformat(margin=2 ** 100).strip())
                output_stream.write("\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Parse a text file using a given parser checkpoint")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--input-path", type=str, required=False)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--vocabulary-dir", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=False, default="-")

    args = parser.parse_args()

    if args.output_path == "-":
        main(Path(args.checkpoint), Path(args.input_path), Path(args.vocabulary_dir), output_stream=sys.stdout)
    else:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(exist_ok=True)
        with output_path.open("w", encoding="utf8") as fh_out:
            main(Path(args.checkpoint), Path(args.input_path), Path(args.vocabulary_dir), output_stream=fh_out)
