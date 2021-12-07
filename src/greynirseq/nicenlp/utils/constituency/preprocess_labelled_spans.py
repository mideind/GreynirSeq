#!/usr/bin/env python3
# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.
# Built on top of fairseqs fairseq_cli/preprocess.py

"""
Data pre-processing: build vocabularies and binarize training data.
"""

import logging
import os
import shutil
import sys
from collections import Counter
from multiprocessing import Pool

import torch
from fairseq import options, tasks, utils
from fairseq.binarizer import Binarizer
from fairseq.data import indexed_dataset

"""
example usage:
    python preprocess_labelled_spans.py --only-source \
    --trainpref data/nonterm.gold \
    --nonterm_suffix txt \
    --task multi_span_prediction
"""

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger("fairseq_cli.preprocess")


def main(args):
    utils.import_user_module(args)

    os.makedirs(args.destdir, exist_ok=True)

    logger.addHandler(logging.FileHandler(filename=os.path.join(args.destdir, "preprocess.log")))
    logger.info(args)

    task = tasks.get_task(args.task)

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    def build_dictionary(filenames, src=False, tgt=False):
        assert src ^ tgt
        return task.build_dictionary(
            filenames,
            workers=args.workers,
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor,
        )

    label_dictionary, label_schema = task.load_label_dictionary(args, args.label_schema)
    labelled_span_parser = make_parse_labelled_spans(label_dictionary, label_schema)

    def make_binary_labelled_spans_dataset(input_prefix, output_prefix, num_workers):
        nseq = [0]

        def merge_result(worker_result):
            nseq[0] += worker_result["nseq"]

        input_file = input_prefix
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize_labelled_spans,
                    (
                        args,
                        input_file,
                        labelled_span_parser,
                        prefix,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                    ),
                    callback=merge_result,
                )
            pool.close()

        ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, None, "bin"), impl=args.dataset_impl)

        merge_result(
            Binarizer.binarize_alignments(
                input_file,
                labelled_span_parser,
                lambda t: ds.add_item(t),
                offset=0,
                end=offsets[1],
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = "{}/{}".format(args.destdir, prefix)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))

        logger.info("[labelled spans] {}: parsed {} sentences".format(input_file, nseq[0]))

    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers):
        logger.info("[{}] Dictionary: {} types".format(lang, len(vocab) - 1))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(input_prefix, ("." + lang) if lang is not None else "")
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        vocab,
                        prefix,
                        lang,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                    ),
                    callback=merge_result,
                )
            pool.close()

        ds = indexed_dataset.make_builder(
            dataset_dest_file(args, output_prefix, lang, "bin"),
            impl=args.dataset_impl,
            vocab_size=len(vocab),
        )
        merge_result(Binarizer.binarize(input_file, vocab, lambda t: ds.add_item(t), offset=0, end=offsets[1]))
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        logger.info(
            "[{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                vocab.unk_word,
            )
        )

    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args.dataset_impl == "raw":
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)
        else:
            make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers)

    if args.nonterm_suffix:
        if args.trainpref and os.path.exists("{}.{}".format(args.trainpref, args.nonterm_suffix)):
            make_binary_labelled_spans_dataset(
                "{}.{}".format(args.trainpref, args.nonterm_suffix),
                "train.nonterm",
                args.workers,
            )
        if args.validpref and os.path.exists("{}.{}".format(args.validpref, args.nonterm_suffix)):
            make_binary_labelled_spans_dataset(
                "{}.{}".format(args.validpref, args.nonterm_suffix),
                "valid.nonterm",
                args.workers,
            )
        if args.testpref and os.path.exists("{}.{}".format(args.testpref, args.nonterm_suffix)):
            make_binary_labelled_spans_dataset(
                "{}.{}".format(args.testpref, args.nonterm_suffix),
                "test.nonterm",
                args.workers,
            )
    elif args.term_suffix:
        if args.trainpref:
            make_dataset(
                label_dictionary,
                args.trainpref + "." + args.term_suffix,
                "train.term",
                args.source_lang,
                num_workers=args.workers,
            )
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid.term{}".format(k) if k > 0 else "valid.term"
                make_dataset(
                    label_dictionary,
                    validpref + "." + args.term_suffix,
                    outprefix,
                    args.source_lang,
                    num_workers=args.workers,
                )
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test.term{}".format(k) if k > 0 else "test.term"
                make_dataset(
                    label_dictionary,
                    testpref + "." + args.term_suffix,
                    outprefix,
                    args.source_lang,
                    num_workers=args.workers,
                )


def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=True):
    ds = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, lang, "bin"),
        impl=args.dataset_impl,
        vocab_size=len(vocab),
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, vocab, consumer, append_eos=append_eos, offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


def binarize_alignments(args, filename, parse_alignment, output_prefix, offset, end):
    ds = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, None, "bin"),
        impl=args.dataset_impl,
        vocab_size=None,
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize_alignments(filename, parse_alignment, consumer, offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))
    return res


def binarize_labelled_spans(args, filename, parse_spans, output_prefix, offset, end):
    ds = indexed_dataset.make_builder(
        dataset_dest_file(args, output_prefix, None, "bin"),
        impl=args.dataset_impl,
        vocab_size=None,
    )

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize_alignments(filename, parse_spans, consumer, offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))
    return res


def make_parse_labelled_spans(label_dictionary, label_schema):
    cat_set = set(label_schema.label_categories)
    assert len(cat_set) == len(label_schema.label_categories)

    def parse_labelled_spans(line):
        items = line.strip().split()
        assert len(items) % 3 == 0, "Expected labelled span items to be multiple of 3"
        parsed_spans = torch.zeros(len(items), dtype=torch.int)
        for span_idx in range(len(items) // 3):
            span_start, span_end, span_label = items[3 * span_idx : 3 * span_idx + 3]
            parsed_spans[3 * span_idx + 0] = int(span_start)
            parsed_spans[3 * span_idx + 1] = int(span_end)
            encoded_label = label_dictionary.index(span_label)
            parsed_spans[3 * span_idx + 2] = encoded_label
            if not (0 <= encoded_label - label_dictionary.nspecial <= len(cat_set)) or span_label not in cat_set:
                print(
                    (
                        span_label,
                        encoded_label,
                        encoded_label - label_dictionary.nspecial,
                        len(cat_set),
                        span_label in cat_set,
                    ),
                    line,
                )
                import pdb

                pdb.set_trace()
            assert span_label in cat_set
            assert encoded_label - label_dictionary.nspecial <= len(cat_set)
            if encoded_label == label_dictionary.unk():
                print(span_label, line)
        return parsed_spans

    return parse_labelled_spans


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    if lang is not None:
        lang_part = ".{}-{}.{}".format(args.source_lang, args.target_lang, lang)
    elif args.only_source:
        lang_part = ""
    else:
        lang_part = ".{}-{}".format(args.source_lang, args.target_lang)

    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def get_offsets(input_file, num_workers):
    return Binarizer.find_offsets(input_file, num_workers)


def cli_main():
    parser = options.get_preprocessing_parser()
    parser.add_argument("--label_schema", type=str, default=None)
    parser.add_argument("--nonterm_suffix", type=str, default=None)
    parser.add_argument("--term_suffix", type=str, default=None)
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    cli_main()
