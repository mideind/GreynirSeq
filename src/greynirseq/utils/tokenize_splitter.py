# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

from typing import List

import spacy
import tokenizer
from reynir import bintokenizer
from reynir.binparser import BIN_Token


def prep_text_for_tokenizer(text):
    return "[[ " + " ]] [[ ".join(text.split("\n\n")) + " ]]"


def index_text(text, correct_spaces: bool = False):
    """Segments contiguous (Icelandic) text into paragraphs and sentences
    and returns:
        dictionary of sentence indices to sentences
        dictionary of paragraph index to constituent sentence indices"""
    text = prep_text_for_tokenizer(text)
    tok_stream = bintokenizer.tokenize(text)

    pgs = tokenizer.paragraphs(tok_stream)
    pg_idx_to_sent_idx = dict()
    sent_idx_to_sent = dict()
    curr_sent_idx = 0
    curr_pg_idx = 0

    for pg in pgs:
        sent_idxs = []
        for _, sent in pg:
            curr_sent = list(filter(BIN_Token.is_understood, sent))
            curr_sent = tokenizer.normalized_text_from_tokens(curr_sent)
            if correct_spaces:
                curr_sent = tokenizer.correct_spaces(curr_sent)
            sent_idxs.append(curr_sent_idx)
            sent_idx_to_sent[curr_sent_idx] = curr_sent
            curr_sent_idx += 1
        pg_idx_to_sent_idx[curr_pg_idx] = sent_idxs
        curr_pg_idx += 1
    return pg_idx_to_sent_idx, sent_idx_to_sent


def split_text(text):
    """Segments contiguous (Icelandic) text into paragraphs and sentences
    and returns a list of lists
    """
    text = prep_text_for_tokenizer(text)
    tok_stream = bintokenizer.tokenize(text)
    pgs = tokenizer.paragraphs(tok_stream)
    data = []
    for pg in pgs:
        pg_data = []
        for _, sentence in pg:
            sentence = list(filter(BIN_Token.is_understood, sentence))
            sentence = tokenizer.normalized_text_from_tokens(sentence)
            pg_data.append(sentence)
        data.append(pg_data)
    return data


class SentenceSegmenter:
    """Sentence segmenter for English and Icelandic."""

    def __init__(self, lang: str):
        self.lang = lang
        if lang == "en":
            # Try to make the sentence segmenter faster
            cls = spacy.util.get_lang_class(lang)  # 1. Get Language class, e.g. English
            nlp = cls()  # 2. Initialize it
            nlp.add_pipe("sentencizer")
            self.spacy = nlp

    def segment_text(self, text: str) -> List[str]:
        """Segments text into sentences."""
        if self.lang == "en":
            return self._en_segmenter(text)
        elif self.lang == "is":
            return self._is_segmenter(text)
        else:
            raise NotImplementedError()

    def _is_segmenter(self, text: str) -> List[str]:
        """Sentence segmenter for Icelandic."""
        return [sentence.lstrip(" ") for sentence in tokenizer.split_into_sentences(text, original=True)]

    def _en_segmenter(self, text: str) -> List[str]:
        """Sentence segmenter for English."""
        return [sent.text for sent in self.spacy(text).sents]
