"""Utitilities for Dictionary objects."""
import logging
from typing import List

from fairseq.data import Dictionary

logger = logging.getLogger(__name__)


def remove_madeupwords_from_dictionary(dictionary: Dictionary) -> int:
    """
    Remove madeupwords from the dictionary.
    Changes are done in-place. Returns an int of the number of madeupwords removed.
    """
    # We remove the madeupwords from the dictionary.
    # In symbols, a list of string symbols is stored.
    # In indices, a dict of sym -> int is stored.
    # In count, a list of ints is stored, corresponding to the symbols.
    idx = 0
    while (madeupword := f"madeupword{idx:04d}") in dictionary:
        removed_elements = idx
        sym_idx = dictionary.indices[madeupword]
        # Since sym_idx refers to a index in a list we are removing from,
        # we need to take into account how many elements we have removed from the list.
        del dictionary.symbols[sym_idx - removed_elements]
        del dictionary.indices[madeupword]
        del dictionary.count[sym_idx - removed_elements]
        idx += 1
    return idx


def ensure_symbols_are_present(dictionary: Dictionary, symbols: List[str], ok_to_increase_dict_size: bool) -> None:
    """
    Ensure that the symbols in the source and target dictionary are present.

    Makes changes to the dictionaries in-place.
    """
    original_size = len(dictionary)
    _ = remove_madeupwords_from_dictionary(dictionary)
    for symbol in symbols:
        dictionary.add_symbol(symbol)
    dictionary.pad_to_multiple_(8)
    if not ok_to_increase_dict_size:
        # Let's not crash - but rather point out that we are not allowed to increase the dictionary size.
        if len(dictionary) != original_size:
            logger.warning("The dictionary size changed. The model loading will probably fail.")
