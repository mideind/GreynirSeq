#!/usr/bin/env python3
# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.


import json
import random
from pathlib import Path

import click

import greynirseq.nicenlp.utils.constituency.greynir_utils as greynir_utils


@click.group()
def main():
    pass


@main.command()
@click.argument("input_file", type=click.File("r"))
@click.argument("text_file", type=click.File("w", lazy=True))
@click.argument("term_file", type=click.File("w", lazy=True))
@click.argument("nonterm_file", type=click.File("w", lazy=True))
@click.option("--sep", default="<sep>", type=str)
@click.option("--merge/--no-merge", default=True)
@click.option("--seed", type=int, default=1)
@click.option("--binarize-trees/--no-binarize-trees", default=True)
@click.option("--simplify/--no-simplify", default=False)
@click.option("--append", default=False, is_flag=True)
@click.option("--ignore-errors", default=False, is_flag=True)
@click.option("--error-log", type=click.File("w"))
@click.option("--append-errors", default=False, is_flag=True)
@click.option("--limit", default=-1, type=int)
def export(
    input_file,
    text_file,
    nonterm_file,
    term_file,
    sep,
    merge,
    seed,
    binarize_trees,
    simplify,
    append,
    ignore_errors,
    error_log,
    append_errors,
    limit,
):
    print(f"Extracting data from {input_file.name} to: {Path(nonterm_file.name).parent}")
    random.seed(seed)

    label_sep = "\t"
    nonterm_sep, term_sep, text_sep = "\n", "\n", "\n"
    if merge:
        label_sep, nonterm_sep, text_sep = " ", " ", " "
        term_sep = " {} ".format(sep)

    num_skipped = 0
    tree_gen = greynir_utils.Node.from_psd_file_obj(input_file, ignore_errors=ignore_errors, limit=limit)

    if append_errors:
        error_log.mode = "a"
    if append:
        nonterm_file.mode = "a"
        term_file.mode = "a"
        text_file.mode = "a"

    for _tree_idx, (tree, tree_str) in enumerate(tree_gen):
        if tree is None and not ignore_errors:
            num_skipped += 1
            if error_log is not None:
                error_log.write(tree_str)
                error_log.write("\n")
            continue

        try:
            tree.split_multiword_tokens()
            tree = tree.collapse_unary()
            # We binarize on export so that the length of serialized trees is the same
            # after rebinarization (assertion constraint in parallel dataset workers)
            tree = tree.binarize()

            nterms, terms = tree.labelled_spans(include_null=binarize_trees, simplify=simplify)
        except Exception as _:  # noqa: F841
            num_skipped += 1
            if error_log is not None:
                error_log.write(tree_str)
                error_log.write("\n")
            continue
        if not (nterms and terms):
            continue

        per_nterm_output, per_term_output, text_output = [], [], []
        for lbled_span in terms:
            (start, end), lbl, depth, node, flags = lbled_span
            text = node.text
            lbls = [lbl] + flags

            per_term_output.append(label_sep.join(lbls))
            text_output.append(text)
        for lbled_span in nterms:
            (start, end), lbl, depth, node, flags = lbled_span
            per_nterm_output.append(label_sep.join([str(start), str(end), lbl]))

        nonterm_file.write(nonterm_sep.join(per_nterm_output))
        nonterm_file.write("\n")
        term_file.write(term_sep.join(per_term_output))
        term_file.write("\n")
        text_file.write(text_sep.join(text_output))
        text_file.write("\n")
    if ignore_errors:
        print(f"Skipped {num_skipped}/{_tree_idx + 1} trees ({round(100 * num_skipped/(_tree_idx + 1), 3)}%)")


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
    obj["ignore_categories"] = []
    json.dump(obj, output_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
