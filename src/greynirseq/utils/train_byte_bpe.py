# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import json

from tokenizers import ByteLevelBPETokenizer


def main(args):
    paths = [path for path in args.input.split(":")]

    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    tokenizer.train(
        files=paths,
        vocab_size=args.vocab_size,
        min_frequency=args.min_freq,
        special_tokens=["<s>", "<pad>", "</s>", "<unk>"],
    )

    # Save files to disk
    tokenizer.save("{}.json".format(args.name), pretty=True)

    tok_spec = json.loads(tokenizer.to_str())
    with open("{}-vocab.json".format(args.name), "w") as fp:
        json.dump(tok_spec["model"]["vocab"], fp, indent=4)
    with open("{}-merges.txt".format(args.name), "w") as fp:
        fp.write("\n".join(tok_spec["model"]["merges"]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Description")

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input file",
        metavar="FILE",
    )
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        help="File prefix",
        metavar="NAME",
    )
    parser.add_argument(
        "-s",
        "--vocab-size",
        type=int,
        default=16_000,
        metavar="N",
        help="Total vocabulary size (not merge operations)",
    )
    parser.add_argument(
        "--min-freq",
        type=int,
        default=3,
        metavar="N",
        help="Ignore tokens with lower frequency than this",
    )

    args = parser.parse_args()
    main(args)
