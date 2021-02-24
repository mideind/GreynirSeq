"""
Items:
- POS and Constituency parsing
- Consituency featuers
- Consituency featuers
- Probing


Probing task
    - BigramShift
    - Coordination Inversion
    - Depth
    - Length
    - ObjNumber
    - OddManOut
    - SubjNumber
    - Tense
    - TopConst
    - WordContent

"""

import code
import json
import os
import random
import readline
import sys
from pathlib import Path
from pprint import pprint as pp

import click
import reynir
import tokenizer

import greynirseq.nicenlp.utils.constituency.greynir_utils as greynir_utils

try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


@click.group()
def main():
    pass


@main.command()
@click.argument("input_file", type=click.File("r"))
@click.argument("text_file", type=click.File("w"))
@click.argument("term_file", type=click.File("w"))
@click.argument("nonterm_file", type=click.File("w"))
@click.option("--sep", default="<sep>", type=str)
@click.option("--merge/--no-merge", default=True)
@click.option("--anno/--no-anno", default=False)
@click.option("--seed", type=int, default=1)
@click.option("--include-null/--no-include-null", default=True)
@click.option("--simplified/--no-simplified", default=False)
@click.option("--psd", default=False)
def export(input_file, text_file, nonterm_file, term_file, sep, merge, anno, seed, include_null, simplified, use_new):
    print("Extracting data from trees to: {}".format(str(Path(nonterm_file.name).parent)))
    random.seed(seed)

    converter = greynir_utils.Node.from_simple_tree
    if anno:
        converter = greynir_utils.Node.from_anno_dict

    label_sep = "\t"
    nonterm_sep, term_sep, text_sep = "\n", "\n", "\n"
    if merge:
        label_sep, nonterm_sep, text_sep = " ", " ", " "
        term_sep = " {} ".format(sep)

    def generator_old(input_file):
        for _jline_idx, jline in enumerate(input_file):
            tree = json.loads(jline)
            tree = converter(tree)
            yield tree

    print("Using psd reader")
    if use_new:
        tree_gen = greynir_utils.Node.from_psd_file_obj(input_file, limit=None)
    else:
        tree_gen = generator_old(input_file)

    for _tree_idx, tree in enumerate(tree_gen):
        tree.split_multiword_tokens()
        tree = tree.collapse_unary()
        # We binarize on export so that the length of serialized trees is the same
        # after rebinarization (assertion constraint in parallel dataset workers)
        tree = tree.binarize()

        nterms, terms = tree.labelled_spans(include_null=include_null, simplify=False)

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


@main.command()
@click.argument("output_file", type=click.File("w"))
@click.option("--simplify/--no-simplify", default=True)
def dump_nonterm_schema(output_file, simplify):
    if simplify:
        obj = greynir_utils.make_simple_nonterm_labels_decl()
    else:
        obj = greynir_utils.make_nonterm_labels_decl()
    json.dump(obj, output_file, indent=4, ensure_ascii=False)


@main.command()
@click.argument("output_file", type=click.File("w"))
@click.option("--sep", default="<sep>", type=str)
def dump_term_schema(output_file, sep):
    obj = greynir_utils.make_term_label_decl(sep=sep)
    json.dump(obj, output_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
