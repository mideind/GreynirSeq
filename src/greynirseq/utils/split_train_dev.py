#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Edited to accept seed at Miðeind ehf
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Split a large file into a train and valid set while respecting document
boundaries. Documents should be separated by a single empty line.
"""

import argparse
import random
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("sample_output", help="train output file")
    parser.add_argument("remainder_output", help="valid output file")
    parser.add_argument("-p", type=float, help="remainder size 0-1")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    parser.add_argument("--lines", action="store_true", help="split lines instead of docs")
    args = parser.parse_args()

    data = []
    remainder = []
    sample = []

    def update_sample(doc):
        data.append(doc.copy())
        doc.clear()

    with open(args.input, "r", encoding="utf-8") as h:
        doc = []
        for i, line in enumerate(h):
            if line.strip() == "":  # empty line indicates new document
                update_sample(doc)
            else:
                doc.append(line)
            if args.lines:
                update_sample(doc)
            if i % 1000 == 0:
                print(i, file=sys.stderr, end="", flush=True)
            elif i % 100 == 0:
                print(".", file=sys.stderr, end="", flush=True)
        if len(doc) > 0:
            update_sample(doc)
    print(file=sys.stderr, flush=True)

    assert args.p is not None
    data_size = len(data)
    remainder_size = int(args.p * data_size)
    random.seed(args.seed)
    remainder_idxs = set(random.sample(range(data_size), remainder_size))

    for i in range(data_size):
        data_item = data[i]
        if i in remainder_idxs:
            remainder.append(data_item)
        else:
            sample.append(data_item)

    print("Data size {}".format(data_size))
    print("Remainder size {}".format(remainder_size))

    with open(args.sample_output, "w", encoding="utf-8") as out:
        first = True
        for doc in sample:
            if not first and not args.lines:
                out.write("\n")
            first = False
            for line in doc:
                out.write(line)

    with open(args.remainder_output, "w", encoding="utf-8") as out:
        first = True
        for doc in remainder:
            if not first and not args.lines:
                out.write("\n")
            first = False
            for line in doc:
                out.write(line)


if __name__ == "__main__":
    main()
