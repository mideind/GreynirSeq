import argparse
from typing import Dict, Optional

from fairseq.data import Dictionary, encoders


def get_word_beginnings(args: argparse.Namespace, dictionary: Dictionary) -> Optional[Dict[int, int]]:
    bpe = encoders.build_bpe(args)
    if bpe is not None:

        def is_beginning_of_word(i):
            if i < dictionary.nspecial:
                return True
            tok = dictionary[i]
            if tok.startswith("madeupword"):
                return True
            try:
                return bpe.is_beginning_of_word(tok)
            except ValueError:
                return True

        is_word_initial = {}
        for i in range(len(dictionary)):
            is_word_initial[i] = int(is_beginning_of_word(i))
        return is_word_initial
    return None
