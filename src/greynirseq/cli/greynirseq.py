#!/usr/bin/env python

import argparse
import io
import sys
from typing import List, Union

import torch
import tqdm
from fairseq.hub_utils import GeneratorHubInterface


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

    def handle_batch(self, batch: List[str], output: io.IOBase) -> None:
        results = self.infer(batch)
        for result in results:
            output.write(result + "\n")

    def run(self, input: io.IOBase, output: io.IOBase) -> Union[None, io.IOBase]:
        batch = []
        input_iterable = map(str.rstrip, input)
        if self.show_progress:
            input_iterable = tqdm.tqdm(input_iterable)

        for line in input_iterable:
            if not line:
                # If input is empty write empty line to preserve order
                if batch:
                    self.handle_batch(batch, output)
                    batch = []
                output.write("\n")
                continue

            split_line = line.split()
            split_line_length = len(split_line)
            if split_line_length > self.max_input_length:
                if batch:
                    self.handle_batch(batch, output)
                    batch = []

                ranges = range(0, split_line_length, self.max_input_length)
                temp_batch = []
                for i in ranges:
                    temp_batch.append(" ".join(split_line[i : i + self.max_input_length]))

                sub_batches = range(0, len(temp_batch), self.batch_size)
                results = []
                for i in sub_batches:
                    inference = self.infer(temp_batch[i : i + self.batch_size])
                    results += inference
                output.write(" ".join(results) + "\n")
                continue

            batch.append(line)
            if len(batch) == self.batch_size:
                self.handle_batch(batch, output)
                batch = []
        if batch:
            self.handle_batch(batch, output)


class NER(GreynirSeqIO):
    def build_model(self) -> GeneratorHubInterface:
        model = torch.hub.load("mideind/GreynirSeq:hub", "icebert.ner")
        model.to(self.device)
        model.eval()
        return model

    def infer(self, batch) -> List[str]:
        batch_labels = list(self.model.predict_labels(batch))
        formated_labels = []

        for sentence_labels in batch_labels:
            formated_labels.append(" ".join(sentence_labels))
        return formated_labels


class POS(GreynirSeqIO):
    def build_model(self) -> GeneratorHubInterface:
        model = torch.hub.load("mideind/GreynirSeq:hub", "icebert.pos")
        model.to(self.device)
        model.eval()
        return model

    def infer(self, batch) -> List[str]:
        batch_labels = self.model.predict_ifd_labels(batch)
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
        command = NER

    elif args.command.lower() == "pos":
        command = POS

    handler = command(args.device, args.bsz, args.progress, args.max_input_words_split)
    handler.run(args.input, args.output)


if __name__ == "__main__":
    main()
