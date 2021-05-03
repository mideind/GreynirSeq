#!/usr/bin/env python

"""
    GreynirSeq: Neural natural language processing for Icelandic

    Copyright (C) 2020 Miðeind ehf.
       This program is free software: you can redistribute it and/or modify
       it under the terms of the GNU General Public License as published by
       the Free Software Foundation, either version 3 of the License, or
       (at your option) any later version.
       This program is distributed in the hope that it will be useful,
       but WITHOUT ANY WARRANTY; without even the implied warranty of
       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
       GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see http://www.gnu.org/licenses/.
"""


import argparse
from typing import List, Tuple

from reynir import Greynir, _Sentence
from reynir.bintokenizer import TokenList
from reynir.ifdtagger import IFD_Tagset

from greynirseq.nicenlp.models.icebert import IcebertModel  # pylint: disable=no-name-in-module

TokLem = Tuple[List[str], List[str]]


class Lemmatizer:

    SPLIT_WC = 100

    g = None  # Placeholder for Greynir instance
    ib = None  # Placeholder for IceBERT model instance
    device = "cpu"

    def __init__(self, use_icebert: bool = True) -> None:
        self.g = Greynir()
        if use_icebert:
            self.ib = IcebertModel.pos_from_settings()

    def lemmatize_sentence(self, sentence: _Sentence) -> TokLem:
        if sentence.tree is not None:
            return self.g_lemmatize(sentence)
        a_lemmas = []
        a_tokens = []
        tokens = sentence.tokens
        # Split words to not hit 512 token limit in IceBERT
        # Consider making this smarter if dealing with a lot of long sentences.
        for i in range(0, len(tokens), self.SPLIT_WC):
            p_lemmas, p_tokens = self.ib_lemmatize(tokens[i * 100 : (i + 1) * 100])  # noqa
            a_lemmas += p_lemmas
            a_tokens += p_tokens
        return a_lemmas, a_tokens

    def lemmatize(self, text: str) -> List[Tuple[List[str], List[str], str]]:
        lemmatized = []
        parsed = self.parse(text)
        for sentence in parsed:
            lemmas, tokens = self.lemmatize_sentence(sentence)
            lemmatized.append(([l for l in lemmas], tokens, sentence.tidy_text))  # noqa
        return lemmatized

    def lemmatize_pretty(self, text: str) -> None:
        lemmatized_data = self.lemmatize(text)
        for lemmatized, tokens, _ in lemmatized_data:
            print("---")
            print("\t".join("{:10}".format(v) for v in tokens))
            print("\t".join("{:10}".format(v.lower()) for v in lemmatized))

    def g_lemmatize(self, g_sentence: _Sentence) -> TokLem:
        tokens = [t.txt for t in g_sentence.tokens]
        if g_sentence.tree is None:
            return tokens, tokens
        return g_sentence.tree.lemmas, tokens

    def ib_lemmatize(self, g_tokens: TokenList) -> TokLem:
        tokens = [t.txt for t in g_tokens]
        sent = " ".join(tokens)
        if self.ib is None:
            raise ValueError("Lemmatizer needs to be instantiated with use_icebert.")
        ifds = self.ib.predict_to_idf(sent, device=self.device)
        lemmas = []

        for idx, tok in enumerate(g_tokens):

            cands = tok.val
            if isinstance(cands, int) or isinstance(cands, float):
                # Number
                lemmas.append(tok.txt)
                continue
            if cands and len(cands) > 1 and (isinstance(cands[0], int) or isinstance(cands[0], float)):
                # Punctuation
                lemmas.append(tok.txt)
                continue
            if not cands:
                lemmas.append(tok.txt)
                continue

            lemm_cands = set(c.stofn for c in cands if hasattr(c, "stofn"))
            if len(lemm_cands) == 1:
                # Only one candidate, we use that one
                lemmas.append(lemm_cands.pop())
                continue

            found = False
            for c in cands:
                if hasattr(c, "name"):
                    lemmas.append(c.name)
                    found = True
                    break
                if isinstance(c[0], int):
                    lemmas.append(tok.txt)
                    found = True
                    break
                try:
                    ifd = IFD_Tagset(
                        k=tok.kind,
                        c=c.ordfl,
                        t=c.ordfl,
                        f=c.fl,
                        txt=tok.txt,
                        s=c.stofn,
                        b=c.beyging,
                    )
                except:  # noqa
                    lemmas.append(tok.txt)
                    found = True
                    break
                try:
                    str_ifd = str(ifd)
                except TypeError:
                    # Some oddity in ifdtagger
                    str_ifd = ""
                if str_ifd == ifds[idx]:
                    lemmas.append(c.stofn)
                    found = True
                    break
            if not found:
                lemmas.append(tok.txt)

        return lemmas, tokens

    def parse(self, text: str) -> List[_Sentence]:
        if self.g is None:
            raise ValueError("Greynir needs to be instantiated.")
        text = text.replace("\n", " ").replace("  ", " ")
        return self.g.parse(text)["sentences"]  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Lemmatize Icelandic text")
    parser.add_argument("--sentence", type=str)
    args = parser.parse_args()
    sentence = args.sentence

    lem = Lemmatizer()
    lem.lemmatize_pretty(sentence)


if __name__ == "__main__":
    main()
