#!/usr/bin/env python

import argparse
import io
import sys
from typing import Generator, Iterable, List, Union

import torch
import tqdm
from fairseq.hub_utils import GeneratorHubInterface


class GreynirSeqAPI:
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


class NER(GreynirSeqAPI):
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


class POS(GreynirSeqAPI):
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


def main():
    parser = argparse.ArgumentParser()

    # Use subparsers if implementing command specific handling, i.e. choice of output format
    parser.add_argument("command", action="store", choices=["ner", "pos", "NER", "POS"])
    parser.add_argument("--input", action="store", dest="input", type=argparse.FileType("r"), default=sys.stdin)
    parser.add_argument("--output", action="store", dest="output", type=argparse.FileType("w"), default=sys.stdout)
    parser.add_argument("--device", action="store", dest="device", type=str, default="cpu")
    parser.add_argument("--batch-size", action="store", dest="bsz", type=int, default=1)
    parser.add_argument("--progressbar", action="store_true", dest="progress", default=False)
    parser.add_argument(
        "--max-input-words-split",
        action="store",
        dest="max_input_words_split",
        default=250,
        help="Splits input-string before inference if spaces exceed this amount, re-joins afterwards.",
    )

    args = parser.parse_args()
    if args.command.lower() == "ner":
        handler: GreynirSeqAPI = NER(args.device, args.bsz, args.progress, args.max_input_words_split)

    elif args.command.lower() == "pos":
        handler: GreynirSeqAPI = POS(args.device, args.bsz, args.progress, args.max_input_words_split)
    else:
        raise ValueError(f"Unknown command: {args.command}")

    for result in handler.run(args.input):
        args.output.write(result + "\n")


if __name__ == "__main__":
    main()
