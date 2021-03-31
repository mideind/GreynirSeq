#!/usr/bin/env python

import argparse
import sys

import torch
import tqdm


class GreynirSeqIO:
    def __init__(self, input, output, device, batch_size, show_progress):
        self.input = input
        self.output = output
        self.device = device
        self.batch_size = batch_size
        self.show_progress = show_progress
        self.model = self.build_model()

    def build_model(self):
        raise NotImplementedError

    def infer(self, batch):
        raise NotImplementedError

    def run(self):
        batch = []
        input_iterable = map(str.rstrip, self.input)
        if self.show_progress:
            input_iterable = tqdm.tqdm(input_iterable)

        for line in input_iterable:
            if not line:
                # If input is empty write empty line to preserve order
                self.output.write("\n")
                continue

            batch.append(line)
            if len(batch) == self.batch_size:
                results = self.infer(batch)
                for result in results:
                    self.output.write(result + "\n")
                batch = []
        if batch:
            results = self.infer(batch)
            for result in results:
                self.output.write(result + "\n")


class NER(GreynirSeqIO):
    def build_model(self):
        model = torch.hub.load("mideind/GreynirSeq:hub", "icebert.ner")
        model.to(self.device)
        model.eval()
        return model

    def infer(self, batch):
        batch_labels = list(self.model.predict_labels(batch))
        formated_labels = []

        for sentence_labels in batch_labels:
            formated_labels.append(" ".join(sentence_labels))
        return formated_labels


class POS(GreynirSeqIO):
    def build_model(self):
        model = torch.hub.load("mideind/GreynirSeq:hub", "icebert.pos")
        model.to(self.device)
        model.eval()
        return model

    def infer(self, batch):
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

    args = parser.parse_args()
    if args.command.lower() == "ner":
        command = NER

    elif args.command.lower() == "pos":
        command = POS

    handler = command(args.input, args.output, args.device, args.bsz, args.progress)
    handler.run()


if __name__ == "__main__":
    main()
