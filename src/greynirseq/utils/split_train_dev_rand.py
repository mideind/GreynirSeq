#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Edited at Miðeind ehf to approximate the percentage split.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Split a large file into a train and valid set while respecting document
boundaries. Documents should be separated by a single empty line.
The percentage split are approximated by sampling.
"""

import argparse
import random
import sys
from typing import List, TextIO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("sample_output", help="train output file")
    parser.add_argument("remainder_output", help="valid output file")
    parser.add_argument("-p", type=float, help="remainder size 0-1")
    parser.add_argument("--seed", default=1, type=int, help="seed")
    parser.add_argument("--lines", action="store_true", help="split lines instead of docs")
    args = parser.parse_args()

    random.seed(args.seed)

    def write_doc(doc: List[str], p: float, f_remainder: TextIO, f_sample: TextIO, separate_docs=True) -> bool:
        """Write the document to either the remainder or sample based on p.
        Return True if the document was written to the remainder."""
        written_to_remainder = False
        file_to_write = f_sample
        if p > random.random():
            file_to_write = f_remainder
            written_to_remainder = True
        file_to_write.writelines(doc)
        if separate_docs:
            file_to_write.write("\n")
        return written_to_remainder

    p = args.p
    separate_docs = not args.lines
    total = 0
    remainder_count = 0
    with open(args.input, "r", encoding="utf-8") as h, open(
        args.sample_output, "w", encoding="utf-8"
    ) as f_sample, open(args.remainder_output, "w", encoding="utf-8") as f_remainder:
        doc = []
        for i, line in enumerate(h):
            if line.strip() == "":  # empty line indicates new document
                total += 1
                remainder_count += write_doc(doc, p, f_remainder, f_sample, separate_docs)
                doc.clear()
            else:
                doc.append(line)
            if not separate_docs and len(doc) != 0:
                total += 1
                remainder_count += write_doc(doc, p, f_remainder, f_sample, separate_docs)
            if i % 1000 == 0:
                print(i, file=sys.stderr, end="", flush=True)
            elif i % 100 == 0:
                print(".", file=sys.stderr, end="", flush=True)
        if len(doc) > 0:
            total += 1
            remainder_count += write_doc(doc, p, f_remainder, f_sample, separate_docs)
    print(file=sys.stderr, flush=True)
    print(f"Wrote {total} lines. {remainder_count/total:.3f} fraction to remainder")


if __name__ == "__main__":
    main()
