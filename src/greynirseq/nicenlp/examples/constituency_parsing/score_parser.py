# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.


from pathlib import Path

import nltk
import parsingtestpipe.helpers as helpers
import tokenizer

from greynirseq.nicenlp.utils.constituency.greynir_utils import Node, NonterminalNode


def simplify_node_tree(tree):
    def _inner(node):
        if node.terminal:
            return [node]
        assert node.nonterminal
        new_children = []
        for child in node.children:
            new_children.extend(_inner(child))
        simple_label = helpers.GENERALIZE.get(node.nonterminal, node.nonterminal)
        if simple_label == "":
            # this nonterminal should be deleted and the parent node should contain its children
            return new_children
        return [NonterminalNode(simple_label, children=new_children)]

    # tree.pretty_print()
    result = _inner(tree)
    assert len(result) == 1
    # tree.pretty_print()
    # breakpoint()
    tree = result[0]

    # if any(t.text == "júní" for t in tree.leaves):
    #     tree.pretty_print()
    tokenizer_tokens = [t for t in tokenizer.tokenize(tree.text) if t.txt]
    if len(tokenizer_tokens) == len(tree.leaves):
        for leaf, tok in zip(tree.leaves, tokenizer_tokens):
            if tok.kind == tokenizer.TOK.PUNCTUATION and leaf.terminal in ("x", "null"):
                leaf._terminal = "p"
            elif leaf.terminal == "null":
                leaf._terminal = "n"
    else:
        for leaf in tree.leaves:
            tok = [t for t in tokenizer.tokenize(leaf.text) if t.txt][0]
            if tok.kind == tokenizer.TOK.PUNCTUATION:
                leaf._terminal = "p"
            else:
                leaf._terminal = "n"
        pass
    # if any(t.text == "júní" for t in tree.leaves):
    #     tree.pretty_print()
    #     breakpoint()
    return result[0]


def simplify_filepaths(input_dir, output_dir, in_suffix):
    print(f"Simplifying {input_dir} and writing to {output_dir}")
    for input_path in input_dir.glob(f"**/*.{in_suffix}"):
        output_path = output_dir / input_path.relative_to(input_dir).with_suffix(".psd")
        output_path.parent.mkdir(exist_ok=True)
        print(f"Parsing {input_path}")
        with output_path.open("w", encoding="utf8") as out_fh:
            for tree_str in input_path.open("r").read().split("\n\n"):
                # crude hack (until re-export)
                tree_str = tree_str.replace(r"\(", "&#40;").replace(r"\)", "&#41;")
                tree = nltk.tree.Tree.fromstring(tree_str)
                tree = Node.from_nltk_tree(tree)
                tree = simplify_node_tree(tree)
                out_fh.write(tree.as_nltk_tree().pformat(margin=2 ** 100).strip())
                out_fh.write("\n")


if __name__ == "__main__":
    import argparse

    try:
        import argcomplete  # noqa: F401
    except ImportError as e:  # noqa: F841
        pass
    parser = argparse.ArgumentParser("Description")

    parser.add_argument(
        "--input",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--suffix",
        type=str,
        required=False,
        default="psd",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        required=False,
        default="psd",
    )
    parser.add_argument(
        "--gold-dir",
        type=str,
        required=False,
        default="psd",
    )

    args = parser.parse_args()
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    simplified_dir = output_dir
    gold_dir = Path(args.gold_dir)
    report_dir = Path(args.report_dir)

    report_dir.mkdir(exist_ok=True)
    simplified_dir.mkdir(exist_ok=True)

    simplify_filepaths(input_dir, output_dir, args.suffix)
    if args.report_dir and args.gold_dir:
        print("Retrieving results from evalb")

        test_file_suffix = ".psd"  # model output
        gold_file_suffix = ".br"
        output_file_suffix = ".out"  # intermediate results for helpers.py
        suffixes = [(test_file_suffix, gold_file_suffix, output_file_suffix)]
        helpers.get_results(gold_dir, simplified_dir, report_dir, suffixes, exclude=True)
        # helpers.get_results(gold_dir, simplified_dir, report_dir, suffixes, exclude=False)
        print("Combining reports")
        helpers.combine_reports(report_dir)
