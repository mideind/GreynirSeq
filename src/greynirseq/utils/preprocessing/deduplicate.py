#!/usr/bin/env python3

import argparse
import hashlib

from tokenizer import split_into_sentences
from tokenizers import ByteLevelBPETokenizer


class Corpus:
    def __init__(self, files, max_lines, min_lines, max_bpe_length, merge_file, vocab_file):
        self.files = files
        self.pg_hashes = set()
        self.line_hashes = {i: set() for i in range(min_lines, max_lines + 1)}

        self.max_lines = max_lines
        self.min_lines = min_lines
        self.max_bpe_length = max_bpe_length

        self.bpe_tokenizer = self.load_tokenizer(merge_file, vocab_file)

    def load_tokenizer(self, merge_file, vocab_file):
        return ByteLevelBPETokenizer(merges_file=merge_file, vocab_file=vocab_file)

    @classmethod
    def hash(cls, text):
        return hashlib.md5(text.encode()).hexdigest()

    def is_new_pg(self, pg):
        hash = self.hash(pg)
        if hash not in self.pg_hashes:
            self.pg_hashes.add(hash)
            return True
        return False

    def add_pg_to_line_hashes(self, sentences):
        for wsz in range(self.min_lines, self.max_lines + 1):
            for i in range(0, len(sentences) + 1 - wsz):
                window = " ".join(sentences[i * wsz : (i + 1) * wsz])  # noqa
                wdw_hash = self.hash(window)
                self.line_hashes[wsz].add(wdw_hash)

    def check_sentence(self, sentence):
        if len(self.bpe_tokenizer.encode(sentence)) > 512:
            return False
        return True

    def clean_pg(self, pg):
        sentences = [s for s in split_into_sentences(pg) if self.check_sentence(s)]
        n_sentences = len(sentences)

        n_pg = []
        idx = 0

        while idx < n_sentences - self.min_lines + 1:
            for j in range(self.max_lines, self.min_lines - 1, -1):
                if idx + j > n_sentences:
                    continue

                sentence_batch = " ".join(sentences[idx : idx + j])  # noqa
                sh = self.hash(sentence_batch)
                if sh in self.line_hashes[j]:
                    idx += j
                    break

            if idx < n_sentences:
                n_pg.append(sentences[idx])
                idx += 1

        n_pg += sentences[idx:]

        if n_pg:
            self.add_pg_to_line_hashes(sentences)

        return "\n".join(n_pg)

    def deduplicate_file(self, f, of):
        line = f.readline()
        while line:
            pg = ""
            while line and line != "\n":
                pg += line
                line = f.readline()
            line = f.readline()

            if not self.is_new_pg(pg):
                continue

            pg = self.clean_pg(pg)

            of.writelines(pg)
            of.writelines("\n\n")

    def deduplicate(self, outfile):
        of = open(outfile, "w")
        for f in self.files:
            self.deduplicate_file(f, of)
        of.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpe-merges")
    parser.add_argument("--bpe-vocab")
    parser.add_argument("--max-sentences", type=int)
    parser.add_argument("--min-sentences", type=int)
    parser.add_argument("--max-bpe-length", type=int, default=512)
    parser.add_argument("--output")
    parser.add_argument("files", type=argparse.FileType("r"), nargs="+")
    args = parser.parse_args()

    corpus = Corpus(
        args.files,
        args.max_sentences,
        args.min_sentences,
        args.max_bpe_length,
        args.bpe_merges,
        args.bpe_vocab,
    )
    corpus.deduplicate(args.output)


if __name__ == "__main__":
    main()
