#!/usr/bin/env python3
# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.


import json
import random
from collections import Counter
from pathlib import Path

import click

import greynirseq.nicenlp.utils.constituency.greynir_utils as greynir_utils
from greynirseq.nicenlp.utils.constituency.incremental_parsing import NULL_LABEL, ROOT_LABEL


@click.group()
def main():
    pass


@main.command()
@click.argument("input_file", type=click.File("r"))
@click.argument("output_file", type=click.File("w", lazy=True))
@click.option("--seed", type=int, default=1)
@click.option("--ignore-errors", default=False, is_flag=True)
@click.option("--error-log", type=click.File("w"))
@click.option("--limit", default=-1, type=int)
@click.option(
    "--label-file", type=click.File("w"), required=True, help="Label dictionary file, analogous to fairseqs dict.txt"
)
def export_greynir(input_file, output_file, seed, ignore_errors, error_log, limit, label_file):
    print(f"Extracting data from {input_file.name} to: {Path(output_file.name)}")
    random.seed(seed)

    num_skipped = 0
    tree_gen = greynir_utils.Node.from_psd_file_obj(input_file, ignore_errors=ignore_errors, limit=limit)
    label_dict = Counter()
    label_dict.update([ROOT_LABEL, NULL_LABEL, greynir_utils.NULL_LEAF_NONTERM])

    for _tree_idx, (tree, tree_str) in enumerate(tree_gen):
        if tree is None and error_log:
            if error_log is not None:
                error_log.write(tree_str)
                error_log.write("\n")
            continue
        if tree is None:
            num_skipped += 1
            continue

        # construct label dictionary
        for node in tree.preorder_list():
            label_dict.update([node.label_without_flags])
            if node.label_flags:
                label_dict.update(node.label_flags)
        label_dict.update([part for leaf in tree.leaves for part in leaf.terminal.split("_") if part])

        output_file.write(tree.to_json())
        output_file.write("\n")
    for label, freq in label_dict.most_common(2 ** 100):
        label_file.write(f"{label} {freq}\n")
    if ignore_errors:
        print(f"Skipped {num_skipped}/{_tree_idx + 1} trees ({round(100 * num_skipped/(_tree_idx + 1), 3)}%)")


@main.command()
@click.argument("input_file", type=click.File("r"))
def parse_greynir(input_file,):
    for line in input_file:
        tree = greynir_utils.Node.from_json(line)
        tree.pretty_print()
        breakpoint()
        print()


@main.command()
@click.argument("output_file", type=click.File("w"))
@click.option("--simplify/--no-simplify", default=False)
def dump_nonterm_schema(output_file, simplify):
    simplified_type = "simplified" if simplify else "unsimplified"
    print(f"Writing {simplified_type} nonterm schema to file: {output_file.name}")
    if simplify:
        obj = greynir_utils.make_simple_nonterm_labels_decl()
    else:
        obj = greynir_utils.make_nonterm_labels_decl()
    obj["ignore_categories"] = []
    json.dump(obj, output_file, indent=4, ensure_ascii=False)


@main.command()
@click.argument("output_file", type=click.File("w"))
@click.option("--sep", default="<sep>", type=str)
def dump_term_schema(output_file, sep):
    obj = greynir_utils.make_term_label_decl(sep=sep)
    obj["ignore_categories"] = obj.get("ignore_categories", [])
    json.dump(obj, output_file, indent=4, ensure_ascii=False)


@main.command()
@click.argument("input_dictionaries", nargs=-1, type=click.File("r"))
@click.option("--output", type=click.File("w"), required=True)
def merge_dicts(input_dictionaries, output):
    cntr = Counter()
    for curr_fh in input_dictionaries:
        for line in curr_fh:
            line = line.strip()
            if not line:
                break
            if line.count(" ") == 0:
                breakpoint()
            label, freq = line.split(" ")
            cntr[label] += int(freq)
    if not cntr:
        return
    for label, freq in cntr.most_common(2 ** 100):
        output.write(f"{label} {freq}\n")


if __name__ == "__main__":
    main()
