#!/usr/bin/env python3

# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

"""
filters.py is a module to which defines filters and transformations which can be used to
clean up monolingual and bilingual data.

Example usage from command line:
python src/greynirseq/utils/preprocessing/filters.py -i <INPUT> \
    --languages en \
    --functions normalize_spaces \
        merge_spaces \
        replace_control_format \
        replace_dashes \
        remove_leading_bullet \
        null_sentence \
        whitelist_symbol \
        deduplicate \
    --summary -q > <OUTPUT>

The above command will read the input file which is monolingual English, normalize spaces, merge spaces, etc.
When the --summary flag is set, the output will be a summary of the filters and transformations applied.
-q is a shorthand for --quiet so the command will not print to stdout the filtered (i.e. removed) examples.

The module can also be used as a library:

from greynirseq.utils.preprocessing.filters import Pipeline
pipeline = Pipeline(functions=["normalize_spaces", "merge_spaces"])
examples = [
    {"en": "This is a   sentence."},
    {"en": "This is another\n sentence."},
]
for clean_ex, transformed, filtered in pipeline.run(examples):
    print(clean_ex)
"""

import argparse
import collections
import itertools
import json
import os
import re
import sys
from io import StringIO
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import editdistance
import tokenizer
import tqdm
from langid.langid import LanguageIdentifier, model

from greynirseq.utils.preprocessing.symbols import (
    BANNED_SYMBOLS,
    ICE_QUOTE,
    NON_STANDARD_SPACES,
    PUNCTUATION_SYMBOLS,
    QUOTE_LIKE,
    SEPARATORS,
    SUBSTITUTE_FOR_NULL,
    SYMBOL_WHITELIST,
)

_SCRIPT_DIR = os.path.dirname(os.path.realpath("__file__"))

LANGID_IDENTIFIER = LanguageIdentifier.from_modelstring(model, norm_probs=True)
LANGID_IDENTIFIER.set_languages(["en", "is"])
MAX_SUBTOKENS_PER_SENTENCE = 256
MAX_CHARS_PER_SENTENCE = 500
BANNED_SYMBOLS_PAT = "[" + re.escape(BANNED_SYMBOLS) + "]"
SUBWORD_TOKENIZER_AVAILABLE = False


def safe_div(a, b):
    if b == 0:
        return 0
    return a / b


# TODO: Add support for BPE encoding
def encode(text: str):
    """BPE encodes a string."""
    return text


class Deduplifier:
    """Deduplify sentence pairs using Tilde's approach (Tilde 2018)
    Along with a few others."""

    _set: Set[int] = set()

    @classmethod
    def preprocess_sentence(cls, sentence):
        digit_prog = RegexCache.compile_rx(r"\d+")
        punct_prog = RegexCache.compile_rx("[" + re.escape(PUNCTUATION_SYMBOLS) + "]")
        space_prog = RegexCache.compile_rx(r"\s+")
        sentence = sentence.lower()
        sentence = digit_prog.sub("0", sentence)
        sentence = punct_prog.sub("", sentence)
        sentence = space_prog.sub("", sentence)
        return sentence

    @classmethod
    def preprocess_example(cls, ex):
        return "\t".join([cls.preprocess_sentence(s) for s in ex.values()])

    @classmethod
    def is_unique_example(cls, ex):
        """Returns True if the example is unique."""
        key = hash(cls.preprocess_example(ex))
        if key in cls._set:
            return False
        cls._set.add(key)
        return True


class Transformations:
    """Transformations of examples to be used in a pipeline"""

    _transforms: Dict[str, Callable[[Dict[str, str]], Dict[str, str]]] = {}  # registered transformations

    @classmethod
    def apply(cls, name: str, ex: Dict[str, str]):
        if name not in cls._transforms:
            raise KeyError("Could not find transformation {0}".format(name))
        else:
            return cls._transforms[name](ex)

    @classmethod
    def register(cls, fun: Callable[[Dict[str, str]], Dict[str, str]]):
        if fun.__name__ not in cls._transforms:
            cls._transforms[fun.__name__] = fun
            return fun
        else:
            raise ValueError("Tried to register transform {0} more than once".format(fun.__name__))

    @classmethod
    def __contains__(cls, name: str):
        return name in cls._transforms


class RegexCache:
    _programs: Dict[str, re.Pattern] = {}  # compiled regular expressions

    @classmethod
    def compile_rx(cls, pattern):
        if pattern not in cls._programs:
            program = re.compile(pattern)
            cls._programs[pattern] = program
        return cls._programs[pattern]


class Filters:
    """Filter storage. Filters return True if an example is OK."""

    _filters: Dict[str, Callable[[Dict[str, str]], bool]] = {}  # registered filters

    @classmethod
    def register(cls, fun) -> Callable[[Dict[str, str]], bool]:
        if fun.__name__ not in cls._filters:
            cls._filters[fun.__name__] = fun
            return fun
        else:
            raise ValueError("Tried to register filter {0} more than once".format(fun.__name__))

    @classmethod
    def apply(cls, filter_name: str, ex: Dict[str, str]) -> bool:
        """Filters return True if example is OK. If filter name starts with 'inv_' its application is inversed."""
        inverted = False
        if filter_name.startswith("inv_"):
            filter_name = filter_name[4:]
            inverted = True
        if filter_name not in cls._filters:
            raise KeyError("Could not find filter {0}".format(filter_name))
        else:
            res = cls._filters[filter_name](ex)
            return not res if inverted else res

    @classmethod
    def __contains__(cls, filter_name: str):
        if filter_name.startswith("inv_"):
            filter_name = filter_name[4:]
        return filter_name in cls._filters


@Transformations.register
def fix_ice_quotes(ex: Dict[str, str]):
    """Fixes the quotes in the Icelandic sentence.
    Other languages are silently ignored."""
    new_ex = dict(ex)
    if "is" in ex:
        ice = ex["is"]
        ice = ice.replace("”", ICE_QUOTE.PRIMARY.RIGHT)  # 0x201d  RIGHT DOUBLE QUOTATION MARK

        if not (set(ICE_QUOTE.PRIMARY.BOTH) & set(ice)):
            ice = ice.replace(ICE_QUOTE.SECONDARY.LEFT, ICE_QUOTE.PRIMARY.LEFT)
            ice = ice.replace(ICE_QUOTE.SECONDARY.RIGHT, ICE_QUOTE.PRIMARY.RIGHT)
        new_ex["is"] = ice

    return new_ex


@Transformations.register
def fix_improper_line_split(ex: Dict[str, str]):
    """Fixes imporper line splits in English and Icelandic sentences.
    Other languages are silently ignored."""
    new_ex = dict(ex)
    ice_prog = RegexCache.compile_rx(r"(\b(?!\d)\w+)- (\b(?!eða|og)\w+\b)")
    eng_prog = RegexCache.compile_rx(r"(\b(?!\d)\w+)- (\b(?!or|and)\w+\b)")
    if "is" in ex:
        new_ex["is"] = ice_prog.sub(r"\1\2", ex["is"])
    if "en" in ex:
        new_ex["en"] = eng_prog.sub(r"\1\2", ex["en"])
    return new_ex


@Transformations.register
def remove_leading_bullet(ex: Dict[str, str]):
    # bullet, hyphen-minus, n-dash, horizontal bar
    prog = RegexCache.compile_rx(r"(^(• ?|- |– |― |\.\s?)+)")
    return {lang: prog.sub("", text) for lang, text in ex.items()}


@Transformations.register
def replace_dashes(ex: Dict[str, str]):
    """Replace hyphen, n-dash, horizontal bar, m-dash, minus sign, figure dash with "standard" dash."""
    prog = RegexCache.compile_rx(r"(‐|–|―|—|−|‒)")
    return {lang: prog.sub("-", text) for lang, text in ex.items()}


@Transformations.register
def replace_control_format(ex: Dict[str, str]):
    """Replace control and format unicode charaters with an empty string."""
    prog = RegexCache.compile_rx("[" + "|".join(SUBSTITUTE_FOR_NULL) + "]")
    return {lang: prog.sub("", text) for lang, text in ex.items()}


@Transformations.register
def merge_spaces(ex: Dict[str, str]):
    """Merge multiple sequential spaces into a single space."""
    prog = RegexCache.compile_rx(r"\s+")
    return {lang: prog.sub(" ", text).strip(" ") for lang, text in ex.items()}


@Transformations.register
def normalize_spaces(ex: Dict[str, str]):
    """Normalize different space unicode characters to a single space."""
    prog = RegexCache.compile_rx("[" + "|".join(NON_STANDARD_SPACES + SEPARATORS) + "]")
    return {lang: prog.sub(" ", text) for lang, text in ex.items()}


@Filters.register
def deduplicate(ex: Dict[str, str]):
    """Return True if example has not been seen before."""
    return Deduplifier.is_unique_example(ex)


@Filters.register
def banned_symbol(ex: Dict[str, str]):
    """Return True if example contains no banned symbols."""
    prog = RegexCache.compile_rx(BANNED_SYMBOLS_PAT)
    backslash = "\\"
    found = False
    for text in ex.values():
        found = found or backslash in text or prog.search(text)
    return not found


@Filters.register
def whitelist_symbol(ex: Dict[str, str]):
    """Return True if example contains only whitelisted symbols."""
    for text in ex.values():
        if not set(text) <= SYMBOL_WHITELIST:
            return False
    return True


@Filters.register
def null_sentence(ex: Dict[str, str]):
    """Return True if example is not empty/null."""
    prog = RegexCache.compile_rx(r"^\s+$")
    is_null = False
    for text in ex.values():
        is_null = is_null or prog.match(text) or not text
    return not is_null


@Filters.register
def quote_inside_word(ex: Dict[str, str]):
    """Return True if example does not have a quote (") inside a word."""
    prog = RegexCache.compile_rx(r'\w+"\w')
    has_error = False
    for text in ex.values():
        has_error = has_error or prog.search(text)
    return not has_error


@Filters.register
def ocr_wrong_symbol(ex: Dict[str, str]):
    """Return True if Icelandic example does not have a wrong OCR symbol."""
    if "is" in ex:
        prog = RegexCache.compile_rx(r",,")
        return not prog.search(ex["is"])
    return True


@Filters.register
def ocr_word_boundary_avg_length(ex: Dict[str, str], avg_length=1.8):
    """Return True if average word length exceeds a threshold."""
    prog = RegexCache.compile_rx(r"\w+")
    avg_lengths_ok = True
    for text in ex.values():
        lens = [len(m) for m in prog.findall(text)]
        avg = safe_div(sum(lens), len(lens))
        avg_lengths_ok = avg_lengths_ok and (avg > avg_length)
    return avg_lengths_ok


@Filters.register
def missing_letter(ex: Dict[str, str]):
    """Return True if Icelandic example does not contain certain missing letters."""
    if "is" in ex:
        substrings = [r" a ", r" e ", r" i ", r" o ", r" u ", r" y ", r"\bvi\b", r"\bess\b", r"\bme\b", r"\bessum\b"]
        found = re.search("(" + "|".join(substrings) + ")", ex["is"])
        if found is not None:
            return False
    return True


@Filters.register
def alphanumeric(ex: Dict[str, str]):
    """Return True if example is only alpha-numeric"""
    is_alpha_num = True
    prog = RegexCache.compile_rx(r"[^\d\W]")
    for text in ex.values():
        is_alpha_num = is_alpha_num and prog.search(text)
    return is_alpha_num


def _sentence_length_ratio(ex: Dict[str, str], ratio_below_min=5.0, ratio_above_min=1.5, min_count=3):
    """Return True if sentence length ratio between the examples is acceptable.
    We compare the longest sentence and the shortest sentence.
    If either side is 'small' (<min_count) then we use a larger ratio.
    if both sides are sufficiently large, we use a stricter ratio."""
    longest_sent = max(len(ex[lang]) for lang in ex.keys())
    shortest_sent = min(len(ex[lang]) for lang in ex.keys())

    ratio_to_use = ratio_above_min
    if shortest_sent < min_count:
        ratio_to_use = ratio_below_min

    return (longest_sent / shortest_sent) <= ratio_to_use


@Filters.register
def strict_sentence_length_ratio(ex: Dict[str, str]):
    return _sentence_length_ratio(ex, min_count=0)


@Filters.register
def subtoken_count_ratio(ex: Dict[str, str], min_count=4):
    if not SUBWORD_TOKENIZER_AVAILABLE:
        return True
    # TODO: implement subword tokenization
    return _sentence_length_ratio(
        {lang: encode(ex[lang]) for lang in ex.keys()}, ratio_above_min=2.0, min_count=min_count
    )


@Filters.register
def case_mismatch(ex: Dict[str, str]):
    """Return True if example does not have a case mismatch.
    Either all should be uppercase or all should not be uppcase."""
    is_upper = []
    for text in ex.values():
        is_upper.append(text.isupper())
    return not any(is_upper) or all(is_upper)


@Filters.register
def digit_mismatch(ex: Dict[str, str]):
    """Return True if all examples have the same collection of numbers in them."""
    prog = RegexCache.compile_rx(r"\D+")
    first_counter = None
    for text in ex.values():
        # TODO: Make clock references pass ('1 pm' vs 'kl. eitt')
        # Remove non-digit characters, group consecutive numbers, filter empty string
        digit_counter = collections.Counter([w for w in prog.sub(" ", text).strip(" ").split(" ") if w])
        if first_counter is None:
            first_counter = digit_counter
        else:
            if digit_counter != first_counter:
                return False
    return True


@Filters.register
def only_even_quotes(ex: Dict[str, str]):
    """Return True if all examples have even number of quotes in them."""
    prog = RegexCache.compile_rx(re.escape('"'))
    for text in ex.values():
        if len(prog.findall(text)) % 2 != 0:
            return False
    return True


@Filters.register
def abs_min_string_edit(ex: Dict[str, str], min_dist=3):
    """Return True if all examples have a minimum edit distance to each other."""
    all_combinations = itertools.combinations(ex.values(), 2)
    for a, b in all_combinations:
        if editdistance.eval(a, b) < min_dist:
            return False
    return True


# TODO: Fix this filter so it can support more than two languages and/or not just "is" and "en".
@Filters.register
def rel_min_string_edit(ex: Dict[str, str], min_ratio=0.10):
    ice, eng = ex["is"], ex["en"]
    num_edits = editdistance.eval(ice, eng)
    lengths = [len(ice), len(eng)]
    min_edits = safe_div(num_edits, min(lengths))
    max_edits = safe_div(num_edits, max(lengths))
    return (max_edits >= min_ratio) and (min_edits >= min_ratio)


# TODO: Fix this filter so it can support more than two languages and/or not just "is" and "en".
@Filters.register
def abs_min_subtoken_edit(ex: Dict[str, str], min_dist=2):
    if not SUBWORD_TOKENIZER_AVAILABLE:
        return True
    ice = encode(ex["is"])
    eng = encode(ex["en"])
    dist = editdistance.eval(ice, eng)
    return dist >= min_dist


# TODO: Fix this filter so it can support more than two languages and/or not just "is" and "en".
@Filters.register
def rel_min_subtoken_edit(ex: Dict[str, str]):
    if not SUBWORD_TOKENIZER_AVAILABLE:
        return True
    ice = encode(ex["is"])
    eng = encode(ex["en"])
    num_edits = editdistance.eval(ice, eng)
    lengths = [len(ice), len(eng)]
    min_ratio = safe_div(num_edits, min(lengths))
    max_ratio = safe_div(num_edits, max(lengths))
    return (max_ratio >= 0.10) and (min_ratio >= 0.10)


@Filters.register
def colon_mismatch(ex: Dict[str, str]):
    """Return True if examples do not have colon mismatch. Accepted colons are ":" and ";"."""
    prog = RegexCache.compile_rx(r"[:;]")
    num_found = None
    for text in ex.values():
        found = len(prog.findall(text))
        if num_found is None:
            num_found = found
        else:
            if found != num_found:
                return False
    return True


@Filters.register
def corrupt_symbol(ex: Dict[str, str]):
    """Return True if examples do not contain a question mark "?" inside a word."""
    symbols = re.escape("?")
    pat = r"\w[" + symbols + r"]+\w"
    prog = RegexCache.compile_rx(pat)
    for text in ex.values():
        if prog.search(text):
            return False


@Filters.register
def improper_line_split(ex: Dict[str, str]):
    """Return True if Icelandic or English sentences do not contain improper line splits."""
    if "is" in ex:
        ice_prog = RegexCache.compile_rx(r"(\b(?!\d)\w+)- (\b(?!eða|og)\w+\b)")
        ice_matches = ice_prog.findall(ex["is"])
        if len(ice_matches) == 0:
            return True
    if "en" in ex:
        eng_prog = RegexCache.compile_rx(r"(\b(?!\d)\w+- \b(?!or|and)\w+\b)")
        eng_matches = eng_prog.findall(ex["en"])
        if len(eng_matches) == 0:
            return True
    return False


@Filters.register
def dot_pattern(ex: Dict[str, str]):
    """Return True if examples do not contain a specific dot pattern."""
    prog = RegexCache.compile_rx("(" + r"\.\s+" + "){2,}")
    for text in ex.values():
        if prog.search(text):
            return False
    return True


@Filters.register
def bullet_mismatch(ex: Dict[str, str]):
    """Return True if examples contain equal number of bullets."""
    count = None
    for text in ex.values():
        found = len(re.findall(r"•", text))
        if count is None:
            count = found
        else:
            if found != count:
                return False
    return True


@Filters.register
def max_word_length(ex: Dict[str, str], maximum_word_length=50):
    """Return True if examples do not contain words longer than the supplied maximum."""
    word_prog = RegexCache.compile_rx(r"\b\w+\b")
    for text in ex.values():
        for word in word_prog.findall(text):
            if len(word) > maximum_word_length:
                return False
    return True


@Filters.register
def min_word_count(ex: Dict[str, str], minimum_word_count=3):
    """Return True if examples contain at least the supplied minimum number of words (according tokenizer)."""
    try:
        for text in ex.values():
            toks = [tok for tok in tokenizer.tokenize(text) if tok.txt is not None and tok.kind == tokenizer.TOK.WORD]
            if len(toks) < minimum_word_count:
                return False
    except TypeError:
        pass
    return True


@Filters.register
def wrong_quotes(ex: Dict[str, str]):
    """Return True if examples do not contain wrong quotes."""
    ALLOWED_QUOTES = set("'\"," + "".join(ICE_QUOTE.ALL))
    DISALLOWED_QUOTES = set(QUOTE_LIKE).difference(ALLOWED_QUOTES)
    for text in ex.values():
        if not set(text) <= DISALLOWED_QUOTES:
            return False
    return True


def probably_english(text, lower_bound=0.800):
    """Return True if text (other than English) is likely to be English."""
    pred_lang_code, prob = LANGID_IDENTIFIER.classify(text.lower())
    is_english = pred_lang_code == "en"
    if is_english and prob >= lower_bound:
        return True
    return False


@Filters.register
def remove_english(ex):
    """Return True if examples are (probably) not English."""
    for key, value in ex.items():
        if key == "en":
            continue
        if probably_english(value):
            return False
    return True


class Gather:

    _store = None
    _prev_store = None
    _initialized_with_file = False

    @classmethod
    def _idemp_init(cls):
        """Idempotently initialize, if gather data exists from last run
        it will be used to filter/transform. Otherwise, pipeline needs
        to be executed twice."""
        raise NotImplementedError

    @classmethod
    def gather(cls, ex):
        raise NotImplementedError

    @classmethod
    def save_to_file(cls):
        # Write gathered data to file
        # This depends on the data structure the gatherer uses
        raise NotImplementedError

    @classmethod
    def _file_path(cls):
        return os.path.join(_SCRIPT_DIR, f"{cls.__name__}.json")


class MinFrequency(Gather):

    auto_pass_len = 3  # for numbers and abbreviations
    min_freq = 2
    word_prog = RegexCache.compile_rx(r"\b\w+\b")
    num_prog = RegexCache.compile_rx(r"^\d+$")

    @classmethod
    def _idemp_init(cls):
        if cls._store is not None:
            return
        try:
            with open(cls._file_path(), "r", encoding="utf8") as fh:
                cls._prev_store = json.load(fh)
        except (FileNotFoundError, json.decoder.JSONDecodeError):
            pass
        cls._store = {"eng": collections.Counter(), "ice": collections.Counter()}

    @classmethod
    def save_to_file(cls):
        try:
            with open(cls._file_path(), "w", encoding="utf8") as fh:
                json.dump(cls._store, fh, indent=2)
        except FileNotFoundError:
            pass

    @classmethod
    def gather(cls, ex):
        cls._idemp_init()
        ice, eng = ex["is"], ex["en"]
        ice_words = cls.word_prog.findall(ice)
        eng_words = cls.word_prog.findall(eng)
        ice_words = [w.lower() for w in ice_words if len(w) > cls.auto_pass_len and not cls.num_prog.match(w)]
        eng_words = [w.lower() for w in eng_words if len(w) > cls.auto_pass_len and not cls.num_prog.match(w)]
        cls._store["ice"].update(ice_words)
        cls._store["eng"].update(eng_words)

        if cls._prev_store:
            pice = all(cls._prev_store["ice"].get(w, 0) >= cls.min_freq for w in ice_words)
            peng = all(cls._prev_store["eng"].get(w, 0) >= cls.min_freq for w in eng_words)
            return pice and peng
        return True


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


class Pipeline:
    """A Pipeline holds a list of functions (transformations or filters) to be applied to a sequence of examples.

    Each example is a dictionary of lang: text."""

    def __init__(self, functions: List[str], view_function: Optional[str] = None) -> None:
        is_filter = [True if Filters.__contains__(f) else False for f in functions]
        is_transform = [True if Transformations.__contains__(f) else False for f in functions]
        is_view = [True if f_name == view_function else False for f_name in functions]
        self.functions = list(zip(functions, is_filter, is_transform, is_view))
        self.counter: Dict[str, int] = dict()

    def run(
        self, it_examples: Iterable[Dict[str, str]], quiet: bool = True
    ) -> Iterable[Tuple[Dict[str, str], bool, bool]]:
        """Run the functions defined in the Pipeline on the provided examples.

        Returns: An iterable of the (possibly) updated examples along with flags indicating if the example
        is transformed and whether it is filtered out."""
        # TODO: add support for view_function
        for ex in it_examples:
            self.counter["total"] = self.counter.get("total", 0) + 1
            is_transformed = False
            is_filtered_out = False
            for f_name, is_filter, is_transform, is_view in self.functions:
                if is_transform:
                    upd_ex = Transformations.apply(f_name, ex)
                    if upd_ex != ex:
                        self.counter[f_name] = self.counter.get(f_name, 0) + 1
                        is_transformed = True
                        # We might make multiple transformations, so we need to update the example
                        ex = upd_ex
                if is_filter:
                    if not Filters.apply(f_name, ex):
                        # We count EACH filter application - even though one is enough to filter out an example.
                        # This will cause strange relative numbers in the report.
                        self.counter[f_name] = self.counter.get(f_name, 0) + 1
                        is_filtered_out = True
                        if not quiet:
                            print(f"({bcolors.OKCYAN}{f_name}{bcolors.ENDC})" + "\t".join(ex.values()), file=sys.stderr)

            self.counter["total_filtered"] = self.counter.get("total_filtered", 0) + int(is_filtered_out)
            self.counter["total_transformed"] = self.counter.get("total_transformed", 0) + int(is_transformed)
            yield ex, is_transformed, is_filtered_out

    def function_summary(self):
        """Return a dict with filtering summary statistics."""
        return {
            "total": self.counter["total"],
            "filtered": self.counter["total_filtered"],
            "transformed": self.counter["total_transformed"],
            "functions": {
                f_name: {
                    "total": self.counter[f_name],
                    "percent": 100 * safe_div(self.counter[f_name], self.counter["total"]),
                    "is_filter": is_filter,
                    "is_transform": is_transform,
                    "rel_percent": 100
                    * safe_div(
                        self.counter[f_name],
                        self.counter["total_filtered"] if is_filter else self.counter["total_transformed"],
                    ),
                }
                for f_name, is_filter, is_transform, is_view in self.functions
            },
        }


def print_function_summary(f_summary, indent=4, out_file=sys.stderr):
    """Print the filtering summary statistics for the pipeline."""
    print(
        "Examples remaining:  {rem:>8d} / {total:<8d}  {pct:5.2f}%".format(
            rem=f_summary["total"] - f_summary["filtered"],
            total=f_summary["total"],
            pct=100 * safe_div(f_summary["total"] - f_summary["filtered"], f_summary["total"]),
        ),
        file=out_file,
    )
    print("-" * 80, file=out_file)
    print(
        "{indent}{name:<30s}  {count:>8s}   {pct:>5s}  {rel:>13s}  (   Type  )".format(
            indent=" " * indent, name="Name", count="Count", pct="Total", rel="Relative"
        ),
        file=out_file,
    )
    for name, f_data in f_summary["functions"].items():
        print(
            "{indent}{name:<30s}  {count:>8d}  {pct:>5.2f}%  {rel:>13.2f}% ({type:>9})".format(
                indent=" " * indent,
                name=name,
                count=f_data["total"],
                pct=f_data["percent"],
                rel=f_data["rel_percent"],
                type="filter" if f_data["is_filter"] else "transform",
            ),
            file=out_file,
        )


def lines_to_examples(lines: StringIO, langs: List[str]) -> Iterator[Dict[str, str]]:
    """Read tsv input and yield examples."""
    num_langs = len(langs)
    for line in lines:
        line = line.strip("\n")
        splits = line.split("\t")
        if len(splits) != num_langs:
            ValueError("Expected {} tab-splittable columns, got {}".format(num_langs, len(splits)))
        ex = dict(zip(langs, splits))
        yield ex


def run_pipeline_with_jsonl(pipeline: Pipeline, in_file: StringIO, out_file: StringIO, quiet: bool):
    """Run the pipeline on JSONL file and write the transformed results out."""
    for json_str in in_file:
        json_str = json_str.rstrip("\n")
        document_json = json.loads(json_str)
        lang = document_json["lang"]
        new_document_value = []
        for paragraph in document_json["document"]:
            new_paragraph_value = []
            for sent in paragraph:
                for ex_upd, is_transformed, is_filtered_out in pipeline.run([{lang: sent}], quiet=quiet):
                    if not is_filtered_out:
                        new_paragraph_value.append(ex_upd[lang])
            # We do not want to add empty paragraphs
            if len(new_paragraph_value) != 0:
                new_document_value.append(new_paragraph_value)
        # If the whole document is filtered out, we do not want to add it to the output
        if len(new_document_value) != 0:
            document_json["document"] = new_document_value
            out_file.write(json.dumps(document_json, ensure_ascii=False) + "\n")


def valid_view_function(string):
    """Check if the view function is valid."""
    fn_names = [fn for fn in Filters._filters] + [fn for fn in Transformations._transforms]
    if string in fn_names:
        return string
    raise argparse.ArgumentError(string, "Invalid filter/transformation name")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        prog="filters.py",
        description="Filters and transformation pipelines for monolingual and parallel corpora. \
Input defaults to stdin if no input file is supplied.",
    )

    parser.add_argument(
        "-i",
        "--in_file",
        dest="in_file",
        type=argparse.FileType("r"),
        default=sys.stdin,
        required=False,
        help="File (tsv or jsonl with --jsonl flag) to run filter through, defaults to stdin.",
    )
    parser.add_argument(
        "-f",
        "--functions",
        dest="functions",
        type=str,
        required=False,
        default=[],
        nargs="+",
        help="Functions to run sample through. Can be transformations or filters.",
    )
    parser.add_argument(
        "-l",
        "--languages",
        dest="langs",
        type=str,
        required=True,
        default=[],
        nargs="+",
        help="The languages present in the input-file (is, en, etc.). If multiple, the file should tab delimented.",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        dest="quiet",
        required=False,
        default=False,
        action="store_true",
        help="Output lines after transformations and filtering.",
    )
    parser.add_argument(
        "-s",
        "--summary",
        dest="summary",
        action="store_true",
        required=False,
        default=False,
        help="Print a summary of function application.",
    )
    parser.add_argument(
        "--jsonl",
        action="store_true",
        required=False,
        default=False,
        help="Is the input file in JSONL format? We will then parse each line as a JSON object and write it as well.",
    )
    parser.add_argument(
        "--hook",
        dest="view_function",
        type=valid_view_function,
        required=False,
        default=None,
        help="Display filter or transformation output as occurs in the pipeline.",
    )
    parser.add_argument(
        "-o",
        "--out_file",
        dest="out_file",
        type=argparse.FileType("w"),
        required=False,
        default=sys.stdout,
        help="File where pipline output will be written",
    )

    args = parser.parse_args()
    p = Pipeline(args.functions, args.view_function)
    unit = " lines"
    if args.jsonl:
        unit = " documents"

    in_file = tqdm.tqdm(args.in_file, unit=unit, desc="Running pipeline")
    if args.jsonl:
        run_pipeline_with_jsonl(p, in_file, args.out_file, args.quiet)  # type: ignore
    else:
        examples = lines_to_examples(in_file, args.langsl)  # type: ignore
        for upd_ex, transformed, filtered in p.run(examples):
            if filtered:
                continue
            print("\t".join(upd_ex.values()), file=args.out_file)
    if args.summary:
        print_function_summary(p.function_summary())
