#!/usr/bin/env python

import argparse
import copy
import json
import sys
from itertools import chain
from typing import Any, Dict, Generator, Iterable, List

import torch
import tqdm
from fairseq.hub_utils import GeneratorHubInterface
from fairseq.models.transformer import TransformerModel

from greynirseq.nicenlp.models import MBART_PRETRAINED_MODELS, TF_PRETRAINED_MODELS
from greynirseq.nicenlp.models.bart import GreynirBARTModel


class GreynirSeqIO:
    def __init__(self, device: str, batch_size: int, show_progress: bool, max_input_words_split: int) -> None:
        self.device = device
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.max_input_length = max_input_words_split
        self.model = self.build_model()

    def build_model(self) -> GeneratorHubInterface:
        raise NotImplementedError

    def infer(self, batch: List[str]) -> List[str]:
        raise NotImplementedError

    def run(self, input: Iterable[str]) -> Generator[str, None, None]:
        batch: List[str] = []
        input_iterable = map(str.rstrip, input)
        if self.show_progress:
            input_iterable = tqdm.tqdm(input_iterable)

        for line in input_iterable:
            if not line:
                # If input is empty write empty line to preserve order
                if batch:
                    yield from self.infer(batch)
                    batch.clear()
                yield ""
                continue

            split_line = line.split()
            split_line_length = len(split_line)
            if split_line_length > self.max_input_length:
                if batch:
                    yield from self.infer(batch)
                    batch.clear()

                ranges = range(0, split_line_length, self.max_input_length)
                temp_batch = []
                for i in ranges:
                    temp_batch.append(" ".join(split_line[i : i + self.max_input_length]))

                sub_batches = range(0, len(temp_batch), self.batch_size)
                results = []
                for i in sub_batches:
                    inference = self.infer(temp_batch[i : i + self.batch_size])
                    results += inference
                yield " ".join(results)
                continue

            batch.append(line)
            if len(batch) == self.batch_size:
                yield from self.infer(batch)
                batch.clear()
        if batch:
            yield from self.infer(batch)


class NER(GreynirSeqIO):
    def build_model(self) -> GeneratorHubInterface:
        model = torch.hub.load("mideind/GreynirSeq:main", "icebert.ner")
        model.to(self.device)
        model.eval()
        return model

    def infer(self, batch) -> List[str]:
        batch_labels = list(self.model.predict_labels(batch))  # type: ignore
        formated_labels = []

        for sentence_labels in batch_labels:
            formated_labels.append(" ".join(sentence_labels))
        return formated_labels


class POS(GreynirSeqIO):
    def build_model(self) -> GeneratorHubInterface:
        model = torch.hub.load("mideind/GreynirSeq:main", "icebert.pos")
        model.to(self.device)
        model.eval()
        return model

    def infer(self, batch) -> List[str]:
        batch_labels = self.model.predict_ifd_labels(batch)  # type: ignore
        formated_labels = []
        for sentence_labels in batch_labels:
            formated_labels.append(" ".join(sentence_labels))
        return formated_labels


class TranslateBART(GreynirSeqIO):
    def __init__(
        self, device: str, batch_size: int, show_progress: bool, max_input_words_split: int, model_args
    ) -> None:
        self.model_args = model_args
        super().__init__(device, batch_size, show_progress, max_input_words_split)

    def build_model(self) -> GeneratorHubInterface:
        model = GreynirBARTModel.from_pretrained(**self.model_args)
        model.to(self.device)
        model.eval()
        return model

    def infer(self, batch) -> List[str]:
        # We return immediately if there are no sentences to translate
        if len(batch) == 0:
            return []
        # We cannot use the encode method of the model. It prepends a "bos" on the source.
        # The "prefix_tokens" in generate also distorts the translations.
        # We thus implement our own sample method (based on the original implementation).
        tokens = [self.model.encode(sentence) for sentence in batch]
        has_bos = tokens[0][0] == self.model.task.src_dict.bos()
        # mbart didnt have bos, bos so we strip it
        if has_bos:
            tokens = [seq[1:] for seq in tokens]
        sample = self.model._build_sample(tokens)  # type: ignore
        gen_args = copy.copy(self.model.args)
        # TODO: Support using generation flags from self.kwargs
        gen_args.beam = 1
        generator = self.model.task.build_generator([self.model.model], gen_args)
        # note: outout from inference_step is a list of lists.
        hypos = self.model.task.inference_step(generator, [self.model.model], sample)
        # The first element is the highest hypotheses
        hypos = [x[0] for x in hypos]
        # We need to order them in the same order.
        hypos = [v for _, v in sorted(zip(sample["id"].tolist(), hypos), key=lambda x: x[0])]
        output_texts = [self.model.decode(x["tokens"]) for x in hypos]
        output_texts = [
            t[:-7] for t in output_texts
        ]  # Remove language code (multilingual BART specific): [IS_is]/[EN_xx]
        return output_texts


class TranslateTransformer(GreynirSeqIO):
    def __init__(
        self,
        device: str,
        batch_size: int,
        show_progress: bool,
        max_input_words_split: int,
        model_args: Dict[str, Any],
    ) -> None:
        self.model_args = model_args
        super().__init__(device, batch_size, show_progress, max_input_words_split)
        self.bos_idx_tensor = torch.LongTensor([self.model.task.src_dict.bos()])

    def build_model(self) -> GeneratorHubInterface:
        model = TransformerModel.from_pretrained(**self.model_args)
        model.to(self.device)
        model.eval()
        return model

    def infer(self, batch) -> List[str]:
        if self.model.task.args.prepend_bos:
            tokenized_sentences = [torch.cat([self.bos_idx_tensor, self.model.encode(sentence)]) for sentence in batch]
        else:
            tokenized_sentences = [self.model.encode(" " + sentence) for sentence in batch]
        # TODO: Support using generation flags from self.kwargs
        batched_hypos = self.model.generate(tokenized_sentences, 1)  # type: ignore
        translations = []
        for s_hypos in batched_hypos:
            translations.append([self.model.decode(hypo["tokens"]) for hypo in s_hypos])  # type: ignore
        return [t[0] for t in translations]


def main():
    DEFAULT_MODEL = "mbart25-cont-ees"
    parser = argparse.ArgumentParser()

    # Use subparsers if implementing command specific handling, i.e. choice of output format
    parser.add_argument(
        "command",
        type=str.lower,
        action="store",
        choices=["ner", "pos", "translate"],
    )
    parser.add_argument("--input", action="store", dest="input", type=argparse.FileType("r"), default=sys.stdin)
    parser.add_argument("--output", action="store", dest="output", type=argparse.FileType("w"), default=sys.stdout)
    parser.add_argument("--device", action="store", dest="device", type=str, default="cpu")
    parser.add_argument("--batch-size", action="store", dest="bsz", type=int, default=1)
    parser.add_argument("--progressbar", action="store_true", dest="progress", default=False)
    parser.add_argument(
        "--source-lang",
        action="store",
        dest="source_lang",
        type=str.lower,
        default=None,
        choices=["is", "en"],
        help="Source language. Needs to specified for translation with --model-name.",
    )
    parser.add_argument(
        "--target-lang",
        action="store",
        dest="target_lang",
        type=str.lower,
        default=None,
        choices=["is", "en"],
        help="Target language. Needs to specified for translation with --model-name.",
    )
    parser.add_argument(
        "--model-name",
        action="store",
        dest="model_name",
        type=str.lower,
        # We remove the ".xxxx" from the model name, i.e. the target and source language.
        choices=list(set(m[:-5] for m in chain(MBART_PRETRAINED_MODELS, TF_PRETRAINED_MODELS))),
        default=None,
        help=f"The translation model to use. Will default to '{DEFAULT_MODEL}'. \
Needs to be set if 'model_name_or_path' in --additional-arguments is not provided. \
Overwrites 'model_name_or_path' in --additional-arguments.",
    )
    parser.add_argument(
        "--max-input-words-split",
        action="store",
        dest="max_input_words_split",
        default=250,
        help="Splits input-string before inference if spaces exceed this amount, re-joins afterwards.",
    )
    parser.add_argument(
        "--additional-arguments",
        action="store",
        dest="model_args",
        default=None,
        type=str,
        help="Additional arguments for translation model loading which are passed to fairseq.",
    )

    args = parser.parse_args()

    # We load any additional arguments
    model_args = {}
    if args.model_args:
        with open(args.model_args) as f:
            model_args = json.load(f)

    if args.command == "ner":
        handler: GreynirSeqIO = NER(args.device, args.bsz, args.progress, args.max_input_words_split)

    elif args.command == "pos":
        handler: GreynirSeqIO = POS(args.device, args.bsz, args.progress, args.max_input_words_split)

    elif args.command == "translate":
        # If no model_name is provided and no model_args are provided, we use the default model
        if "model_name_or_path" not in model_args and not args.model_name:
            args.model_name = DEFAULT_MODEL
        if args.model_name:
            # Validate the source and target languages
            if args.source_lang is None or args.target_lang is None:
                raise SystemExit("--source-lang and --target-lang must be specified for translation with --model-name.")
            if args.source_lang == args.target_lang:
                raise SystemExit("--source-lang and --target-lang must not be the same.")
            model_args["model_name_or_path"] = args.model_name + "-" + args.source_lang + args.target_lang
        if "model_name_or_path" not in model_args:
            raise SystemExit("Please set a valid --model_name or 'model_name_or_path' in --additional_arguments")
        # Figure out the model type
        if (
            model_args["model_name_or_path"] in TF_PRETRAINED_MODELS
            or model_args.get("model_type", None) == "transformer"
        ):
            handler: GreynirSeqIO = TranslateTransformer(
                args.device, args.bsz, args.progress, args.max_input_words_split, model_args
            )
        elif (
            model_args["model_name_or_path"] in MBART_PRETRAINED_MODELS or model_args.get("model_type", None) == "mbart"
        ):
            # We need to specify the "bpe" argument for MBART before we call from_pretrained.
            if "bpe" not in model_args:
                model_args["bpe"] = "sentencepiece"
            handler: GreynirSeqIO = TranslateBART(
                args.device, args.bsz, args.progress, args.max_input_words_split, model_args
            )
        else:
            raise SystemExit(f"Unknown model type {model_args.get('model_name_or_path')}.")

    else:
        raise SystemExit(f"Unknown command: {args.command}")

    for result in handler.run(args.input):
        args.output.write(result + "\n")


if __name__ == "__main__":
    main()
