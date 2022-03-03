#!/usr/bin/env python4

# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import argparse
import collections
import itertools
import json
import os
import re
import sys
import time
from typing import Callable, Dict, List, Set

import editdistance
import tokenizer

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

# from langid.langid import LanguageIdentifier, model
# import pycld2 as cld2

_SCRIPT_DIR = os.path.dirname(os.path.realpath("__file__"))

# LANGID_IDENTIFIER = LanguageIdentifier.from_modelstring(model, norm_probs=True)
# LANGID_IDENTIFIER.set_languages(["en", "is"])
# LANGID_IDENTIFIER = None
# if LANGID_IDENTIFIER is None:
#     LANGID_IDENTIFIER = LanguageIdentifier.from_modelstring(model, norm_probs=True)

MAX_SUBTOKENS_PER_SENTENCE = 256
MAX_CHARS_PER_SENTENCE = 500
BANNED_SYMBOLS_PAT = "[" + re.escape(BANNED_SYMBOLS) + "]"
SUBWORD_TOKENIZER_AVAILABLE = False


def safe_div(a, b):
    if b == 0:
        return 0
    return a / b


def print_ex(ex):
    print("\t".join([ex["en"], ex["is"]]))


def get_or_initialize_encoder():
    global ENC
    # This used to have subword encoder from t2t for subword edit-distance.
    # Removed until this is implemented with fairseq/huggingface/spm
    raise NotImplementedError


# def probably_correct_language2(text, lang_code, lower_bound=0.8):
#     global LANGID_IDENTIFIER
#     pred_lang_code, prob = LANGID_IDENTIFIER.classify(text.lower())
#     res = pred_lang_code == lang_code and prob >= lower_bound
#     if not res:
#         msg = "{0:<8.7f}  {1}/{2}  {3}".format(prob, pred_lang_code, lang_code, text)
#         print(msg)
#     return res

# def probably_correct_language(text, lang_code, lower_bound=0.8):
#     isReliable, bytesFound, *rest = list(cld2.detect(text.lower()))
#     langName, langCode, prob, _ = rest[0][0]
#     if isReliable and langCode == langCode and prob > 80:
#         return True
#     # msg = "{0:<8.7f}  {1}/{2}  {3}".format(prob, langCode, lang_code, text)
#     # print(msg)
#     return False


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
    def apply(cls, filter_name: str, ex: Dict[str, str], inverted=False) -> bool:
        """Filters return True if example is OK."""
        if filter_name not in cls._filters:
            raise KeyError("Could not find filter {0}".format(filter_name))
        else:
            res = cls._filters[filter_name](ex)
            return not res if inverted else res


def register_filter(fun: Callable[[Dict[str, str]], bool]):
    Filters.register(fun)
    return fun


def register_transformation(fun: Callable[[Dict[str, str]], Dict[str, str]]):
    Transformations.register(fun)
    return fun


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
    return {lang: prog.sub("", text).strip(" ") for lang, text in ex.items()}


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


# TODO: Maybe fix this filter.
@Filters.register
def abs_min_subtoken_edit(ex: Dict[str, str], min_dist=2):
    if not SUBWORD_TOKENIZER_AVAILABLE:
        return True
    enc = get_or_initialize_encoder()
    ice = enc.encode(ex["is"])
    eng = enc.encode(ex["en"])
    dist = editdistance.eval(ice, eng)
    return dist >= min_dist


# TODO: Maybe fix this filter.
@Filters.register
def rel_min_subtoken_edit(ex: Dict[str, str]):
    if not SUBWORD_TOKENIZER_AVAILABLE:
        return True
    # pylint: disable=assignment-from-no-return
    enc = get_or_initialize_encoder()
    ice = enc.encode(ex["is"])
    eng = enc.encode(ex["en"])
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
    ALLOWED_QUOTES = set("'\"," + "".join(ICE_QUOTE.ALL))
    DISALLOWED_QUOTES = set(QUOTE_LIKE).difference(ALLOWED_QUOTES)
    for text in ex.values():
        if not set(text) <= DISALLOWED_QUOTES:
            return False
    return True


# @Filters.register
# def language(ex):
#     ice, eng = ex["is"], ex["en"]
#     correct = probably_correct_language(ice, "is", lower_bound=0.995)
#     correct = correct and probably_correct_language(eng, "en", lower_bound=0.995)
#     return correct


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


class Pipeline:

    counter = dict()
    _fns = [
        # filters
        null_sentence,
        alphanumeric,
        banned_symbol,
        whitelist_symbol,
        digit_mismatch,
        deduplicate,
        # case_mismatch,
        max_word_length,
        min_word_count,
        # transformations
        fix_improper_line_split,
        # remove_leading_bullet,
        soft_hyphen,
        replace_dashes,
        merge_spaces,
        # fix_ice_quotes,
        # filters
        # bullet_mismatch,
        # only_even_quotes,
        ocr_wrong_symbol,
        # colon_mismatch,
        # missing_letter,
        # corrupt_symbol,
        quote_inside_word,
        abs_min_string_edit,
        rel_min_string_edit,
        # abs_min_subtoken_edit,
        # rel_min_subtoken_edit,
        strict_sentence_length_ratio,
        # subtoken_count_ratio,
        ocr_word_boundary_avg_length,
        # dot_pattern,
        # wrong_quotes,
        #
        # language,
        # MinFrequency.gather,
        # MinFrequency,
        # MostCommon50k,
    ]
    start_time = None
    end_time = None

    @classmethod
    def run(cls, examples, view_function=None, inverted=False, **kwargs):
        cls.start_time = time.time()
        cls.counter.clear()
        cls.counter["total"] = 0
        for ex in examples:
            cls.counter["total"] += 1
            ex = cls.process(ex, view_function=view_function, inverted=inverted)
            if ex is not None:
                yield ex

        for obj in cls._fns:
            if not isinstance(obj, Gather):
                continue
            obj.save_to_file()
        cls.end_time = time.time()

    @classmethod
    def process(cls, ex, view_function=None, inverted=False):
        for obj in cls._fns:
            fn = obj.gather if isinstance(obj, type) else obj
            name = obj.__name__ if isinstance(obj, type) else fn.__name__
            display = view_function == name
            if name in Transformations._transforms:
                old = dict(ex)
                ex = obj(ex)
                if ex != old:
                    out_ex = old if inverted else ex
                    if display:
                        print_ex(out_ex)
                    cls.counter[name] = cls.counter.get(name, 0) + 1
            else:
                if display and inverted:
                    print_ex(ex)
                if not fn(ex):
                    cls.counter[name] = cls.counter.get(name, 0) + 1
                    if display and not inverted:
                        print_ex(ex)
                    return None

        return ex

    @classmethod
    def summarize_counter(cls, indent=4, file=sys.stdout):
        total = cls.counter["total"]
        cls.counter.pop("total")
        num_filtered = sum(count for name, count in cls.counter.items() if name not in Transformations._transforms)
        # num_transformed = sum(count for name, count in cls.counter.items() if name in Transformations._transforms)
        print(
            "Examples remaining:  {rem:>8d} / {total:<8d}  {pct:5.2f}%  in {elaps:>5.1f} seconds".format(
                rem=total - num_filtered,
                total=total,
                pct=100 * safe_div(total - num_filtered, total),
                elaps=cls.end_time - cls.start_time,
            )
        )
        print("-" * 80)
        print(
            "{indent}{name:<30s}  {count:>8s}   {pct:>5s}  {rel:<10s}".format(
                indent=" " * indent, name="Name", count="Count", pct="Total", rel="Total filtered"
            )
        )
        for fn in cls._fns:
            name = fn.__name__
            if name not in cls.counter:
                continue
            count = cls.counter[name]

            filter_msg = "{indent}{name:<30s}  {count:>8d}  {pct:>5.2f}%  {rel:>13.2f}%".format(
                indent=" " * indent,
                name=name,
                count=count,
                pct=100 * safe_div(count, total),
                rel=100 * safe_div(count, num_filtered),
            )

            transform_msg = "{indent}{name:<30s}  {count:>8d}  {pct:>5.2f}%".format(
                indent=" " * indent, name=name, count=count, pct=100 * safe_div(count, total)
            )
            msg = filter_msg if name in Filters._filters else transform_msg
            print(msg)


class MinimalPipeline(Pipeline):

    counter = dict()
    _fns = [
        # filters
        null_sentence,
        alphanumeric,
        whitelist_symbol,
        digit_mismatch,
        deduplicate,
        case_mismatch,
        max_word_length,
        min_word_count,
        # transformations
        fix_improper_line_split,
        # remove_leading_bullet,
        soft_hyphen,
        replace_dashes,
        merge_spaces,
        fix_ice_quotes,
        wrong_quotes,
        # filters
        bullet_mismatch,
        only_even_quotes,
        ocr_wrong_symbol,
        colon_mismatch,
        missing_letter,
        corrupt_symbol,
        quote_inside_word,
        ocr_word_boundary_avg_length,
    ]
    start_time = None
    end_time = None


def do_pipeline(in_file, out_file, quiet, summary, view_function, inverted, langs, **kwargs):
    examples = lines_to_examples(in_file, langs)
    # pipeline = Pipeline
    pipeline = MinimalPipeline
    for ex in pipeline.run(examples, view_function=view_function, inverted=inverted):
        if not quiet and view_function is None:
            print("\t".join([ex["en"], ex["is"]]), file=out_file)
    if summary:
        pipeline.summarize_counter()


def lines_to_examples(lines, langs: List[str]):
    num_langs = len(langs)
    for line in lines:
        line = line.strip("\n")
        splits = line.split("\t")
        if len(splits) != num_langs:
            ValueError("Expected {} tab-splittable columns, got {}".format(num_langs, len(splits)))
        ex = dict(zip(langs, splits))
        yield ex


class TransformationPipeline(Pipeline):
    _fns = [fix_improper_line_split, remove_leading_bullet, soft_hyphen, replace_dashes, merge_spaces, fix_ice_quotes]


def do_fns(in_file, out_file, transforms, filters, quiet, summary, langs: List[str], **kwargs):
    transforms = [] if transforms is None else transforms
    filters = [] if filters is None else filters
    counter = dict()
    for ex in lines_to_examples(in_file, langs):
        counter["total"] = counter.get("total", 0) + 1
        is_filtered_out = False
        for transform in transforms:
            ex = Transformations.apply(transform, ex)
        for filter_name in filters:
            inverted = False
            if filter_name.endswith("_inv"):
                inverted = True
                filter_name = filter_name.rstrip("_inv")
            if not Filters.apply(filter_name, ex, inverted=inverted):
                counter[filter_name] = counter.get(filter_name, 0) + 1
                is_filtered_out = True
                if not quiet:
                    print("\t".join([ex[lang] for lang in langs]), file=sys.stderr)
        if not is_filtered_out:
            print("\t".join([ex[lang] for lang in langs]), file=out_file)
    if summary:
        print(counter)


def valid_view_function(string):

    fn_names = [fn.__name__ for fn in Pipeline._fns] + [fn.__name__ for fn in MinimalPipeline._fns]
    if string in fn_names:
        return string
    raise argparse.ArgumentError(string, "Invalid filter/transformation name")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        "Filters and transformation pipelines for monolingual and parallel corpora (tab delimiter)."
        "Input defaults to stdin if no input file is supplied."
    )

    parser.add_argument(
        "-i",
        "--in_file",
        dest="in_file",
        type=argparse.FileType("r"),
        default=sys.stdin,
        required=0,
        help="Sample file to run filter through, defaults to stdin",
    )
    parser.add_argument(
        "-f",
        "--filters",
        dest="filters",
        type=str,
        required=False,
        default=[],
        nargs="+",
        help="Filters to run sample through.",
    )
    parser.add_argument(
        "-t",
        "--transforms",
        dest="transforms",
        type=str,
        required=False,
        default=[],
        nargs="+",
        help="Transforms to run sample through.",
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
        "--no_examples",
        dest="quiet",
        required=False,
        default=False,
        action="store_true",
        help="Output lines after transformations and filtering.",
    )
    parser.add_argument(
        "-s", "--summary", dest="summary", action="store_true", required=False, default=False, help="Help"
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
        "-v",
        "--invert",
        dest="inverted",
        action="store_true",
        required=False,
        default=False,
        help=(
            "Print inverse of filter as it occurs in the pipeline"
            "(or the output that is unaffected by a hooked transformation)."
        ),
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
    if args.filters or args.transforms:
        do_fns(**vars(args))
    else:
        do_pipeline(**vars(args))
