#!/usr/bin/env python

import argparse
import sys

import torch


class GreynirSeqIO:
    def __init__(self, input, output, device, batch_size):
        self.input = input
        self.output = output
        self.device = device
        self.batch_size = batch_size
        self.model = self.build_model()

    def build_model(self):
        raise NotImplementedError

    def infer(self, batch):
        raise NotImplementedError

    def run(self):
        batch = []
        for line in map(str.rstrip, self.input):
            batch.append(line)
            if len(batch) == self.batch_size:
                results = self.infer(batch)
            for result in results:
                self.output.write(result + "\n")
            batch = []


class NER(GreynirSeqIO):
    def build_model(self):
        return torch.hub.load("mideind/GreynirSeq:hub", "icebert.ner")

    def infer(self, batch):
        batch_labels = list(self.model.predict_labels(batch))
        formated_labels = []
        for sentence_labels in batch_labels:
            formated_labels.append(" ".join(sentence_labels))
        return formated_labels


class POS(GreynirSeqIO):
    def build_model(self):
        return torch.hub.load("mideind/GreynirSeq:hub", "icebert.pos")

    def infer(self, batch):
        batch_labels = self.model.predict_ifd_labels(batch)
        formated_labels = []
        for sentence_labels in batch_labels:
            formated_labels.append(" ".join(sentence_labels))
        return formated_labels


def main():
    parser = argparse.ArgumentParser()

    # Use subparsers if implementing command specific handling, i.e. choice of output format
    parser.add_argument('command', action="store", choices=["ner", "pos", "NER", "POS"])
    parser.add_argument('--input', action="store", dest="input", type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument('--output', action="store", dest="output", type=argparse.FileType('w'), default=sys.stdout)
    parser.add_argument('--device', action="store", dest="device", type=str, default="cpu")
    parser.add_argument('--batch-size', action="store", dest="bsz", type=int, default=1)

    args = parser.parse_args()
    if args.command.lower() == "ner":
        command = NER

    elif args.command.lower() == "pos":
        command = POS

    handler = command(args.input, args.output, args.device, args.bsz)
    handler.run()


if __name__ == '__main__':
    main()
