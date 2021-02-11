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

from pathlib import Path
import os
import json
from pprint import pprint
import random
import json
import sys
import code
from pprint import pprint as pp
import readline


import reynir
import tokenizer

import click

try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from . import greynir_utils
from . import tree_dist as tree_distance


def reformat_annotrees():
    """12/05/2020
    Skjöl að 201 eru tilbúin nema: 150	165	56
    162,	177	,59
    174,	99,	168
    120,	156,	144
    111,	114,	117
    """
    from annotree import AnnoTree
    GOLD_EXCLUDE_MISSING = set(
        [150, 165, 56, 162, 177, 59, 174, 99, 168, 120, 156, 144, 111, 114, 117]
    )
    GOLD_FIX_LATIN1 = set(
        [
            43,  # broken
            101,
            # 129,
        ]
    )

    BASE_PATH = Path("/home/haukur/github/reynir_corpus/gold")
    FILENAMES = [
        "reynir_corpus_{:>05d}.gld".format(idx)
        for idx in range(1, 201 + 1)
        if not (idx in GOLD_EXCLUDE_MISSING)
    ]
    FIX_FILENAMES = ["reynir_corpus_{:>05d}.gld".format(idx) for idx in GOLD_FIX_LATIN1]

    # # Fix latin1 encoding
    # for filename in FIX_FILENAMES:
    #     path = os.path.join(BASE_PATH, filename)
    #     try:
    #         with open(path, "r", encoding="latin1") as fp:
    #             text = fp.read()
    #     except:
    #         continue
    #     with open(path, "w") as fp:
    #         fp.write(text)
    with open("reynir_corpus.json_lines", "w") as fp_out:
        for filename in FILENAMES:
            trees = AnnoTree.read_from_file(BASE_PATH / filename)
            for tree in trees:
                json_line = json.dumps(tree.to_json())
                if "\n" in json_line:
                    raise ValueError("Invalid tree")
                fp_out.write(json_line)
                fp_out.write("\n")


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
@click.option("--use-new/--no-use-new", default=False)
def export(input_file, text_file, nonterm_file, term_file, sep, merge, anno, seed, include_null, simplified, use_new):
    random.seed(seed)
    converter = greynir_utils.Node.from_simple_tree
    print("Extracting data from trees to: {}".format(str(Path(nonterm_file.name).parent)))
    if anno:
        converter = greynir_utils.Node.from_anno_dict
    label_sep = "\t"
    nonterm_sep, term_sep, text_sep = "\n", "\n", "\n"
    if merge:
        # label_sep, nonterm_sep, text_sep = " ", " {} ".format(sep), " "
        label_sep, nonterm_sep, text_sep = " ", " ", " "
        term_sep = " {} ".format(sep)

    trees1 = []
    trees2 = []
    trees_bsz = 700
    def generator_old(input_file):
        for line_idx, jline in enumerate(input_file):
            tree = json.loads(jline)
            tree = converter(tree)
            yield tree
    ic(use_new)
    if use_new:
        # tree_gen = greynir_utils.Node.from_psd_file_obj(input_file, limit=1000000)
        tree_gen = greynir_utils.Node.from_psd_file_obj(input_file, limit=None)
    else:
        tree_gen = generator_old(input_file)
    for _tree_idx, tree in enumerate(tree_gen):
        # tree = converter(tree)

        # try:
        #     tree = converter(tree)
        # except Exception as e:
        #     ic(json.loads(jline))
        #     import traceback
        #     traceback.print_exc()
        #     input()
        #     continue
        #     # raise e

        tree.split_multiword_tokens()
        # tree.pretty_print()
        tree = tree.collapse_unary()

        # tree.pretty_print()

        # We binarize on export so that the length of serialized trees is the same
        # after rebinarization (assertion constraint in parallel dataset workers)
        tree = tree.binarize()
        # tree.pretty_print()

        # if len(trees1) < trees_bsz:
        #     trees1.append(tree.binarize())
        # elif len(trees2) < trees_bsz:
        #     trees2.append(tree.binarize())
        # else:
        #     import sys
        #     import numpy as np
        #     # foo = greynir_utils.LR_keyroots(trees1[0])
        #     # import pdb; pdb.set_trace()
        #     import pyximport; pyximport.install()
        #     import greynirseq.nicenlp.utils.greynir.tree_distance as tree_distance
        #     # ic("before single dist cyx")
        #     # ic()
        #     # dist = tree_distance.tree_distance(trees1[0], trees2[0])
        #     # ic(dist)
        #     # ic()
        #     import time

        #     # they are actually equal, at least with pyximport.install
        #     start = time.time()
        #     ic("cython many with gil")
        #     dists = []
        #     for t1, t2 in zip(trees1, trees2):
        #         # dists.append(tree_distance.tree_distance(t1, t2, "NULL"))
        #         dists.append(tree_distance.tree_distance(t1, t2, None))
        #     ic(dists[:10], "...")
        #     ic(time.time() - start)
        #     print()

        #     start = time.time()
        #     ic("cython many nogil")
        #     dists = tree_distance.tree_distance_multi(trees1, trees2, None)
        #     ic(dists[:10], "...")
        #     ic(time.time() - start)
        #     print()

        #     sys.exit(0)
        #     pass

        # # ####
        # # tree.pretty_print()
        # # # org = tree
        # # # tree = tree.binarize()
        # # # tree.pretty_print()
        # # # tree.as_nltk_tree().draw()
        # nterms, terms = tree.labelled_spans(include_null=include_null)
        # # tokens = [it.node for it in terms]
        # # # new = greynir_utils.Node.from_labelled_spans([it.span for it in nterms], [it.label for it in nterms], tokens)
        # new = greynir_utils.Node.from_labelled_spans([it.span for it in nterms], [it.label for it in nterms])

        # new = new.debinarize()
        # tree1 = new.binarize()
        # tree2 = new.binarize()
        # ic(tree_distance.tree_distance(tree1, tree2, "NULL"))
        # ic(tree_distance.tree_distance(tree1, tree2, None))
        # tree1.pretty_print()
        # # input("any key...")
        # # ####

        nterms, terms = tree.labelled_spans(include_null=include_null)

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
            # per_nterm_output.append(label_sep.join([str(start), str(end), lbl, *flags]))
        nonterm_file.write(nonterm_sep.join(per_nterm_output))
        nonterm_file.write("\n")
        term_file.write(term_sep.join(per_term_output))
        term_file.write("\n")
        text_file.write(text_sep.join(text_output))
        text_file.write("\n")


@main.command()
@click.argument("output_file", type=click.File("w"))
def dump_nonterm_schema(output_file):
    obj = greynir_utils.make_simple_nonterm_labels_decl()
    json.dump(obj, output_file, indent=4, ensure_ascii=False)


@main.command()
@click.argument("output_file", type=click.File("w"))
@click.option("--sep", default="<sep>", type=str)
def dump_term_schema(output_file, sep):
    obj = greynir_utils.make_term_label_decl(sep=sep)
    json.dump(obj, output_file, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
