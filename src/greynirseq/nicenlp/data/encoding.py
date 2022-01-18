import argparse
from typing import Dict, List, Optional

from fairseq.data import Dictionary, encoders
from torch.functional import Tensor


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


def decode_line(sample: Tensor, bpe, dictionary: Dictionary, extra_symbols_to_ignore: List[int]) -> str:
    """Decode a sample of 1D (Sequence) using the supplied fairseq dict and bpe encoder."""
    assert len(sample.shape) == 1, "Sample must be a 1D tensor."
    bpe_string = dictionary.string(sample, extra_symbols_to_ignore=extra_symbols_to_ignore)
    return bpe.decode(bpe_string)


def encode_line(sentence: str, bpe, dictionary: Dictionary) -> Tensor:
    """Encodes a sentence to a tensor of 1D (Sequence) using the supplied fairseq dict and bpe encoder."""
    bpe_string = bpe.encode(sentence)
    return dictionary.encode_line(bpe_string, append_eos=False, add_if_not_exist=False)
