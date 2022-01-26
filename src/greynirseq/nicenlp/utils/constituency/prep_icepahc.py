#!/usr/bin/env python3
# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

from pathlib import Path

import click
import nltk
from icecream import ic
from nltk.corpus.reader import BracketParseCorpusReader

import greynirseq.nicenlp.utils.constituency.icepahc_utils as icepahc_utils
from greynirseq.nicenlp.utils.constituency import incremental_parsing


def _nltk_version_guard():
    # 3.5 might not work, but 3.4.5 does NOT work
    nltk_major, nltk_minor, *_ = nltk.__version__.split(".")
    nltk_major, nltk_minor = int(nltk_major), int(nltk_minor)
    assert (nltk_major == 3) and (nltk_minor >= 6), "Invalid version of NLTK, expected 3.6 or higher"


def _validate_icepahc_root(ctx, param, value):
    path_str = value
    path = Path(path_str)
    if path.name == "psd":
        return path.parent
    elif (path / "psd").is_dir():
        return path
    raise ValueError(f"Expected {path_str} to be a valid directory")


@click.group()
def main():
    pass


@main.command()
@click.argument("icepahc_root_dir", type=str, callback=_validate_icepahc_root)
@click.argument("text_file", type=click.File("w", lazy=True))
@click.argument("nonterm_file", type=click.File("w", lazy=True))
@click.option("--binarize-trees/--no-binarize-trees", default=True)
@click.option("--decompose-nonterm-flags/--no-decompose-nonterm-flags", default=True)
@click.option("--log-file", required=False, default=None, type=str)
def export(icepahc_root_dir, text_file, nonterm_file, binarize_trees, decompose_nonterm_flags, log_file):
    _nltk_version_guard()

    import itertools
    from collections import Counter
    foo = BracketParseCorpusReader(str(icepahc_root_dir), "psd/.*\.psd")
    ctr = Counter()
    ctr_pos = Counter()

    exports = []
    i = 0
    all_nts = []

    label_sep = "\t"
    sublabel_sep = text_sep = " "

    # path = "psd/1859.hugvekjur.rel-ser.psd"
    # for sent_idx, (sent, tagged_sent, tree) in enumerate(zip(foo.sents(path), foo.tagged_sents(path), foo.parsed_sents(path))):
    for sent_idx, (sent, tagged_sent, tree) in enumerate(zip(foo.sents(), foo.tagged_sents(), foo.parsed_sents())):
        if sent_idx < 108:
            continue
        try:
            res, tree_id = icepahc_utils.convert_icepahc_nltk_tree_to_node_tree(tree, dummy_preterminal="LEAF")
        except icepahc_utils.DiscardTreeException as e:
            if log_file:
                log_file.write(f"{e}")
            continue
        if res is None or res.nonterminal is None:
            continue

        # res.pretty_print()
        icepahc_utils.merge_split_leaves(res)
        icepahc_utils.maybe_unwrap_nonterminals_in_tree(res)
        # res.pretty_print()
        res = res.collapse_unary()

        # try:
        # except ValueError as e:
        #     res.pretty_print()
        #     breakpoint()
        #     print()

        export_str = res.as_nltk_tree().pformat(margin=2 ** 100)
        exports.append(export_str)

        # output_text = " ".join([n.text for n in res.leaves])
        # print(output_text)
        # ic(res.labelled_spans(include_null=binarize_trees, simplify=False))
        # breakpoint()

        # output_nterms, output_terms = res.

        pos_strs = [t.terminal for t in res.leaves]

        # if any("SBJ" in t for t in pos_strs):
        #     res.pretty_print()
        #     breakpoint()

        ctr_pos.update([t.terminal for t in res.leaves])
        ctr.update(icepahc_utils.get_nonterminals(res))

        # res.pretty_print()
        # actions = incremental_parsing.get_incremental_parse_actions(res, collapse=False)
        # ic.enable()
        # ic(actions)
        # breakpoint()

        nterms_in_seq = []
        nterms, _terms = res.labelled_spans(include_null=binarize_trees, simplify=False)
        for lbled_span in nterms:
            (start, end), lbl, depth, node, flags = lbled_span
            items = [str(start), str(end)]
            nterms_in_seq.append(label_sep.join([str(start), str(end), lbl]))
            # items.extend(lbl.split("-"))
            # nterms_in_seq.append(sublabel_sep.join(items))
        text_output = [leaf.text for leaf in res.leaves]

        # text_file.write(text_sep.join(text_output))
        # text_file.write("\n")

        # nonterm_file.write(label_sep.join(nterms_in_seq))
        # nonterm_file.write("\n")

        # tagged_words = [
        #     (word_lemma_str, pos_str)
        #     for (word_lemma_str, pos_str) in tagged_sent
        #     if (pos_str not in ICEPAHC_SKIP_TAGS)
        #     and (not skip_leaf_by_pos(pos_str))
        #     and (not should_skip_by_word_str(word_lemma_str))
        # ]
        # tagged_words = [
        #     (split_wordform_lemma(word_lemma_str, pos_str)[0], pos_str) for (word_lemma_str, pos_str) in tagged_words
        # ]
        # assert tagged_sent[-1][0] == tree_id
        # ic(tagged_words)

        # res.pretty_print()
        # ic(tree_id)
        # breakpoint()
        # # input()

    import json
    with open("/tmp/icepahc_nt_tags.json", "w") as fh_out:
        json.dump(dict(list(ctr.most_common(100000))), fh_out, indent=4)
        print("Exported nt_tags to /tmp/icepahc_nt_tags.json")

    with open("/tmp/icepahc_exports.psd", "w") as fh_out:
        for export_str in exports:
            fh_out.write(export_str)
            fh_out.write("\n")
        print("Exported trees to to /tmp/icepahc_exports.psd")

    with open("/tmp/icepahc_pos_tags.json", "w") as fh_out:
        json.dump(dict(list(ctr_pos.most_common(100000))), fh_out, indent=4)

    # # for word_idx, (word_lemma_str, pos_str) in enumerate(foo.tagged_words()):
    # #     if pos_str == "CPDE":
    # #         pos_str = "CODE"
    # #     if skip_leaf_by_pos(pos_str):
    # #         continue
    # #     if should_skip_by_word_str(word_lemma_str):
    # #         continue
    # #     wordform, lemma = split_wordform_lemma(word_lemma_str, pos_str)
    # #     ctr.update([pos_str])
    # from pprint import pprint
    # mylist = [(t.split("-")[0] if "-" in t else t,c) for (t,c) in ctr.items() if "+" in t]
    # mylist = sorted(mylist, key=lambda x: x[1])
    # pprint(mylist)
    # breakpoint()
    # mylist = set([t for (t,c) in mylist])
    # pprint(mylist)

    # ic(Node.from_nltk_tree(foo.tagged_sents()[0]))
    # breakpoint()


@main.command()
@click.argument("download_dir", type=str)
def download(download_dir):
    import urllib.request
    import zipfile
    icepahc_str = "icepahc-v0.9"
    filename = Path(f"{icepahc_str}.zip")
    path = download_dir / filename
    print(f"Downloading icepahc to: {path}")
    urllib.request.urlretrieve("http://github.com/downloads/antonkarl/icecorpus/icepahc-v0.9.zip", path)
    # http://github.com/downloads/antonkarl/icecorpus/icepahc-v0.9.zip
    print(f"Finished downloading")
    if not Path(path).exists():
        print(f"Could not found icepahc at {path}")
    print(f"Extracting: {path}")
    with zipfile.ZipFile(path, mode="r") as zip_h:
        zip_h.extractall(download_dir)
    print(f"Done")


if __name__ == "__main__":
    main()
