#!/usr/bin/env python3
# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import itertools
import random
from collections import Counter
from pathlib import Path

import click
import nltk
from icecream import ic
from nltk.corpus.reader import BracketParseCorpusReader

import greynirseq.nicenlp.utils.constituency.greynir_utils as greynir_utils
import greynirseq.nicenlp.utils.constituency.icepahc_utils as icepahc_utils
from greynirseq.nicenlp.utils.constituency import incremental_parsing
from greynirseq.nicenlp.utils.constituency.incremental_parsing import NULL_LABEL, ROOT_LABEL


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
@click.option("--train", type=click.File("w", lazy=True), required=True)
@click.option("--valid", type=click.File("w", lazy=True), required=True)
@click.option("--test", type=click.File("w", lazy=True), required=True)
@click.option("--max-seq-len", type=int, required=False, default=100, help="Max sequence length in tokens (words)")
@click.option("--ignore-errors", default=False, is_flag=True)
@click.option("--log-file", required=False, default=None, type=str)
@click.option(
    "--label-file", type=click.File("w"), required=True, help="Label dictionary file, analogous to fairseqs dict.txt"
)
@click.option("--seed", type=int, default=1)
def export(icepahc_root_dir, train, valid, test, max_seq_len, ignore_errors, log_file, label_file, seed):
    print(f"Extracting icepahc from {icepahc_root_dir.name} to: {Path(train.name)}")
    random.seed(seed)
    _nltk_version_guard()

    label_dict = Counter()
    label_dict.update([ROOT_LABEL, NULL_LABEL, greynir_utils.NULL_LEAF_NONTERM])

    corpus_reader = BracketParseCorpusReader(str(icepahc_root_dir), "psd/.*\.psd")
    ctr = Counter()
    ctr_pos = Counter()
    num_skipped = 0
    _tree_idx = 0
    ntrees = 0

    fileids = sorted(corpus_reader.fileids())
    random.shuffle(fileids)
    nfiles_valid = int(len(fileids) * 0.1) + 1
    nfiles_test = int(len(fileids) * 0.1) + 1
    nfiles_train = len(fileids) - nfiles_valid - nfiles_test
    assert nfiles_train > 2 * (nfiles_valid + nfiles_test)

    for file_idx, fileid in enumerate(fileids):
        # fileid = "psd/1902.fossar.nar-fic.psd"
        for _tree_idx, (sent, tagged_sent, tree) in enumerate(
            zip(corpus_reader.sents(fileid), corpus_reader.tagged_sents(fileid), corpus_reader.parsed_sents(fileid))
        ):
            output_file = train
            if file_idx < nfiles_valid:
                output_file = valid
            elif file_idx < nfiles_valid + nfiles_test:
                output_file = test

            ntrees += 1
            try:
                # res, tree_id = icepahc_utils.convert_icepahc_nltk_tree_to_node_tree(tree, dummy_preterminal="LEAF")
                node_tree, tree_id = icepahc_utils.convert_icepahc_nltk_tree_to_node_tree(tree, lowercase_pos=True)
            except icepahc_utils.DiscardTreeException as e:
                if not ignore_errors:
                    raise e
                num_skipped += 1
                # if log_file:
                #     log_file.write(f"{e}")
                continue

            if node_tree is None or node_tree.nonterminal is None:
                num_skipped += 1
                continue
            if len(node_tree.leaves) > max_seq_len:
                num_skipped += 1
                continue

            # if "$" in " ".join(sent):
            #     tree.pretty_print()
            #     print()
            #     node_tree.pretty_print()
            #     breakpoint()

            # construct label dictionary
            for node in node_tree.preorder_list():
                label_dict.update([node.label_without_flags])
                if node.label_flags:
                    label_dict.update(node.label_flags)

            output_file.write(node_tree.to_json())
            output_file.write("\n")
    for label, freq in label_dict.most_common(2 ** 100):
        label_file.write(f"{label} {freq}\n")
    if ignore_errors:
        print(f"Skipped {num_skipped}/{ntrees + 1} trees ({round(100 * num_skipped/(ntrees + 1), 3)}%)")

        ctr_pos.update([t.terminal for t in node_tree.leaves])
    print("Done")


@main.command()
@click.argument("download_dir", type=str)
def download(download_dir):
    import os
    import urllib.request
    import zipfile

    ICEPAHC_NAME = "icepahc-v0.9"
    ICEPAHC_URL = "http://github.com/downloads/antonkarl/icecorpus/icepahc-v0.9.zip"
    filename = f"{ICEPAHC_NAME}.zip"

    download_dir = Path(download_dir)
    download_dir.mkdir(exist_ok=True)
    download_path = download_dir / filename
    print(f"Downloading icepahc to: {download_path}")
    urllib.request.urlretrieve(ICEPAHC_URL, download_path)
    print(f"Finished downloading")
    if not download_path.exists():
        print(f"Could not found icepahc at {download_path}")
    print(f"Extracting into: {download_dir / ICEPAHC_NAME}")
    with zipfile.ZipFile(download_path, mode="r") as zip_h:
        zip_h.extractall(download_dir)
    print(f"Deleting zipfile: {download_path}")
    os.remove(download_path)
    print(f"Done")


if __name__ == "__main__":
    main()
