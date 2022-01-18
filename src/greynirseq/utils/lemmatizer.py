"""A module for non-neural lemmatizers."""
from abc import ABCMeta
from typing import List

import spacy
import tokenizer
from islenska import Bin


class Lemmatizer(metaclass=ABCMeta):
    """A base class for Lemmatizers."""
    def lemmatize(self, sent: str) -> List[str]:
        ...


class ENLemmatizer(Lemmatizer):
    """A basic English Lemmatizer"""

    def __init__(self):
        self.nlp = spacy.load("en_core_web_md")

    def lemmatize(self, sent) -> List[str]:
        lemmas = []
        for sent in self.nlp(sent).sents:
            lemmas.extend(sent.lemma_.split(" "))
        return lemmas


class ISLemmatizer(Lemmatizer):
    """A basic Icelandic Lemmatizer"""

    def __init__(self):
        self.bin = Bin()

    def lemmatize(self, sent) -> List[str]:
        sents = tokenizer.split_into_sentences(sent)
        return [self.lemma_is(w) for s in sents for w in s.split(" ")]

    def lemma_is(self, word: str, **kwargs) -> str:
        _, m = self.bin.lookup(word, **kwargs)
        if m:
            return m[0].ord.replace("-", "")
        return word


def get_lemmatizer_for_lang(lang: str) -> Lemmatizer:
    if lang.lower() in {"is_is", "is"}:
        return ISLemmatizer()
    elif lang.lower() in {"en_xx", "en"}:
        return ENLemmatizer()
    else:
        raise ValueError(f"Missing lemmatizer for {lang}")
