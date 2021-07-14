#!/usr/bin/env python4

# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

import collections
import functools
import json
import os
import re
import sys
import time

import editdistance
import tokenizer
from symbols import BANNED_SYMBOLS, ICE_QUOTE, PUNCTUATION_SYMBOLS, QUOTE_LIKE, SUBSTITUTE_FOR_NULL, SYMBOL_WHITELIST
# from langid.langid import LanguageIdentifier, model
# import pycld2 as cld2

_SCRIPT_DIR = os.path.dirname(os.path.realpath("__file__"))

T2T_AVAILABLE = False
ENC = None

# LANGID_IDENTIFIER = LanguageIdentifier.from_modelstring(model, norm_probs=True)
# LANGID_IDENTIFIER.set_languages(["en", "is"])
# LANGID_IDENTIFIER = None
# if LANGID_IDENTIFIER is None:
#     LANGID_IDENTIFIER = LanguageIdentifier.from_modelstring(model, norm_probs=True)

MAX_SUBTOKENS_PER_SENTENCE = 256
MAX_CHARS_PER_SENTENCE = 500
BANNED_SYMBOLS_PAT = "[" + re.escape(BANNED_SYMBOLS) + "]"
DEFAULT_MIN_WORD_COUNT = 3


def safe_div(a, b):
    if b == 0:
        return 0
    return a / b


class Example:
    def __init__(self, src, tgt, file_id=None, align_score=None):
        self.source = src
        self.target = tgt
        self.file_id = file_id
        self.align_score = align_score

    def __str__(self):
        return "\t".join([self.source, self.target])

    def __repr__(self):
        ret = [self.source, self.target, self.file_id, self.align_score]
        return "[" + "\t".join(ret) + "]"


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

    _set = set()

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
        ice, eng = ex["is"], ex["en"]
        ice = cls.preprocess_sentence(ice)
        eng = cls.preprocess_sentence(eng)
        return eng + "\t" + ice

    @classmethod
    def is_unique_example(cls, ex):
        key = hash(cls.preprocess_example(ex))
        if key in cls._set:
            return False
        cls._set.add(key)
        return True


class Transformations:
    """Transformations of examples to be used in a pipeline"""

    _transforms = {}  # registered transformations

    @classmethod
    def apply(cls, name, ex):
        if name not in cls._transforms:
            raise KeyError("Could not find transformation {0}".format(name))
        else:
            return cls._transforms[name](ex)

    @classmethod
    def register(cls, fun):
        if fun.__name__ not in cls._transforms:
            cls._transforms[fun.__name__] = fun
        else:
            raise ValueError("Tried to register transform {0} more than once".format(fun.__name__))


class RegexCache:
    _programs = {}  # compiled regular expressions

    @classmethod
    def compile_rx(cls, pattern):
        if pattern not in cls._programs:
            program = re.compile(pattern)
            cls._programs[pattern] = program
        return cls._programs[pattern]


class Filters:

    _filters = {}  # registered filters

    @classmethod
    def register(cls, fun):
        if fun.__name__ not in cls._filters:
            cls._filters[fun.__name__] = fun
        else:
            raise ValueError("Tried to register filter {0} more than once".format(fun.__name__))

    @classmethod
    def apply(cls, filter_name, ex, inverted=False):
        if filter_name not in cls._filters:
            raise KeyError("Could not find filter {0}".format(filter_name))
        else:
            res = cls._filters[filter_name](ex)
            return not res if inverted else res


def register_filter(fun):
    Filters.register(fun)
    return fun


def register_transformation(fun):
    Transformations.register(fun)
    return fun


@register_transformation
def fix_ice_quotes(ex):
    ice, eng = ex["is"], ex["en"]
    ice = ice.replace("”", ICE_QUOTE.PRIMARY.RIGHT)  # 0x201d  RIGHT DOUBLE QUOTATION MARK

    if not (set(ICE_QUOTE.PRIMARY.BOTH) & set(ice)):
        ice = ice.replace(ICE_QUOTE.SECONDARY.LEFT, ICE_QUOTE.PRIMARY.LEFT)
        ice = ice.replace(ICE_QUOTE.SECONDARY.RIGHT, ICE_QUOTE.PRIMARY.RIGHT)

    return {"is": ice, "en": eng}


@register_transformation
def fix_improper_line_split(ex):
    ice_prog = RegexCache.compile_rx(r"(\b(?!\d)\w+)- (\b(?!eða|og)\w+\b)")
    eng_prog = RegexCache.compile_rx(r"(\b(?!\d)\w+)- (\b(?!or|and)\w+\b)")
    ice = ice_prog.sub(r"\1\2", ex["is"])
    eng = eng_prog.sub(r"\1\2", ex["en"])
    return {"is": ice, "en": eng}


@register_transformation
def remove_leading_bullet(ex):
    ice, eng = ex["is"], ex["en"]
    # bullet, hyphen-minus, n-dash, horizontal bar
    prog = RegexCache.compile_rx(r"(^(• ?|- |– |― |\.\s?)+)")
    sice = prog.sub("", ice)
    seng = prog.sub("", eng)
    return {"is": sice, "en": seng}


@register_transformation
def replace_dashes(ex):
    ice, eng = ex["is"], ex["en"]
    # hyphen, n-dash, horizontal bar, m-dash, minus sign, figure dash
    prog = RegexCache.compile_rx(r"(‐|–|―|—|−|‒)")
    sice = prog.sub("-", ice)  # hyphen-minus
    seng = prog.sub("-", eng)  # hyphen-minus
    return {"is": sice, "en": seng}


@register_transformation
def soft_hyphen(ex):
    ice, eng = ex["is"], ex["en"]
    prog = RegexCache.compile_rx(SUBSTITUTE_FOR_NULL)
    return {"is": prog.sub("", ice), "en": prog.sub("", eng)}


@register_transformation
def merge_spaces(ex):
    ice, eng = ex["is"], ex["en"]
    prog = RegexCache.compile_rx(r"\s+")
    ice = prog.sub(" ", ice).strip(" ")
    eng = prog.sub(" ", eng).strip(" ")
    return {"is": ice, "en": eng}


@register_filter
def deduplicate(ex):
    return Deduplifier.is_unique_example(ex)


@register_filter
def banned_symbol(ex):
    # TODO(haukurb): this filter is to gather file ids for filtering
    ice, eng = ex["is"], ex["en"]
    prog = RegexCache.compile_rx(BANNED_SYMBOLS_PAT)
    backslash = "\\"
    found = backslash in ice
    found = found or backslash in eng
    found = found or prog.search(ice) or prog.search(eng)
    return not found


@register_filter
def whitelist_symbol(ex):
    ice, eng = ex["is"], ex["en"]
    chars_ex = set(ice)
    chars_ex.update(eng)
    has_non_whitelisted = bool(chars_ex - SYMBOL_WHITELIST)
    return not has_non_whitelisted


@register_filter
def null_sentence(ex):
    prog = RegexCache.compile_rx(r"^\s+$")
    is_only_spaces = prog.match(ex["is"]) or prog.match(ex["en"])
    is_empty = not ex["is"] or not ex["en"]
    return not is_only_spaces and not is_empty


@register_filter
def quote_inside_word(ex):
    # TODO(haukurb): gather file ids from this filter
    prog = RegexCache.compile_rx(r'\w+"\w')
    has_error = prog.search(ex["is"]) or prog.search(ex["en"])
    return not has_error


@register_filter
def ocr_wrong_symbol(ex):
    # TODO(haukurb): gather file ids from this filter
    ice = ex["is"]
    prog = RegexCache.compile_rx(r",,")
    return not prog.search(ice)


@register_filter
def ocr_word_boundary_avg_length(ex):
    ice, eng = ex["is"], ex["en"]
    prog = RegexCache.compile_rx(r"\w+")
    lens_ice = [len(m) for m in prog.findall(ice)]
    lens_eng = [len(m) for m in prog.findall(eng)]
    avg_ice = safe_div(sum(lens_ice), len(lens_ice))
    avg_eng = safe_div(sum(lens_eng), len(lens_eng))
    return avg_ice > 1.8 and avg_eng > 1.8


@register_filter
def missing_letter(ex):
    # TODO(haukurb): gather file ids from this filter
    ice = ex["is"]
    substrings = [r" a ", r" e ", r" i ", r" o ", r" u ", r" y ", r"\bvi\b", r"\bess\b", r"\bme\b", r"\bessum\b"]
    found = re.search("(" + "|".join(substrings) + ")", ice)
    return not found


@register_filter
def alphanumeric(ex):
    ice, eng = ex["is"], ex["en"]
    prog = RegexCache.compile_rx(r"[^\d\W]")
    mice = prog.search(ice)
    meng = prog.search(eng)
    return mice and meng


def _sentence_length_ratio(ex, ratio_below_min=5.0, ratio_above_min=1.5, min_count=3):
    """Enforce length ratios between source and target, if either side is 'small'
    then we use a larger ratio (such as a sentence of one word becomes 5 words on the other side)
    if both sides are sufficiently large, we use a stricter ratio"""
    # note: this is a pass-filter (True if item should remain)
    ice, eng = len(ex["is"]), len(ex["en"])
    recip_ratio_below_min = safe_div(1, ratio_below_min)
    if ice < min_count or eng < min_count:
        return (recip_ratio_below_min * eng <= ice <= ratio_below_min * eng) and (
            recip_ratio_below_min * ice <= eng <= ratio_below_min * ice
        )
    recip_ratio_above_min = safe_div(1, ratio_above_min)

    return (recip_ratio_above_min * eng <= ice <= ratio_above_min * eng) and (
        recip_ratio_above_min * ice <= eng <= ratio_above_min * ice
    )


@register_filter
def strict_sentence_length_ratio(ex):
    return _sentence_length_ratio(ex, ratio_below_min=1.5, min_count=0)


def _subtoken_count_ratio(ex, ratio_below_min=5.0, ratio_above_min=1.5, min_count=3):
    """Same as _sentence_length_ratio, but at the subtoken level"""
    if not T2T_AVAILABLE:
        return ex
    enc = get_or_initialize_encoder()  # pylint: disable=assignment-from-no-return
    ice = len(enc.encode(ex["is"]))
    eng = len(enc.encode(ex["en"]))
    recip_ratio_below_min = safe_div(1, ratio_below_min)
    if ice < min_count or eng < min_count:
        return (recip_ratio_below_min * eng <= ice <= ratio_below_min * eng) and (
            recip_ratio_below_min * ice <= eng <= ratio_below_min * ice
        )
    recip_ratio_above_min = safe_div(1, ratio_above_min)
    return (recip_ratio_above_min * eng <= ice <= ratio_above_min * eng) and (
        recip_ratio_above_min * ice <= eng <= ratio_above_min * ice
    )


@register_filter
def subtoken_count_ratio(ex, min_count=4):
    return _subtoken_count_ratio(ex, ratio_below_min=float("inf"), ratio_above_min=2.0, min_count=min_count)


@register_filter
def case_mismatch(ex):
    ice, eng = ex["is"], ex["en"]
    return not (ice.isupper() ^ eng.isupper())  # xor


@register_filter
def digit_mismatch(ex):
    prog = RegexCache.compile_rx(r"\D+")
    # Remove non-digit characters, group consecutive numbers, filter empty string
    # TODO: Make clock references pass ('1 pm' vs 'kl. eitt')
    ice = [w for w in prog.sub(" ", ex["is"]).strip(" ").split(" ") if w]
    eng = [w for w in prog.sub(" ", ex["en"]).strip(" ").split(" ") if w]
    dice = collections.Counter(ice)
    deng = collections.Counter(eng)
    mismatch = bool(dice - deng) or bool(deng - dice)
    return not mismatch


@register_filter
def only_even_quotes(ex):
    ice, eng = ex["is"], ex["en"]
    prog = RegexCache.compile_rx(re.escape('"'))
    nice = len(prog.findall(ice))
    neng = len(prog.findall(eng))
    return nice % 2 == 0 or neng % 2 == 0


@register_filter
def abs_min_string_edit(ex, min_dist=3):
    ice, eng = ex["is"], ex["en"]
    dist = editdistance.eval(ice, eng)
    return dist >= min_dist


@register_filter
def rel_min_string_edit(ex, min_ratio=0.10):
    ice, eng = ex["is"], ex["en"]
    num_edits = editdistance.eval(ice, eng)
    lengths = [len(ice), len(eng)]
    min_edits = safe_div(num_edits, min(lengths))
    max_edits = safe_div(num_edits, max(lengths))
    return (max_edits >= min_ratio) and (min_edits >= min_ratio)


@register_filter
def abs_min_subtoken_edit(ex, min_dist=2):
    if not T2T_AVAILABLE:
        return ex
    enc = get_or_initialize_encoder()  # pylint: disable=assignment-from-no-return
    ice = enc.encode(ex["is"])
    eng = enc.encode(ex["en"])
    dist = editdistance.eval(ice, eng)
    return dist >= min_dist


@register_filter
def rel_min_subtoken_edit(ex):
    if not T2T_AVAILABLE:
        return ex
    enc = get_or_initialize_encoder()  # pylint: disable=assignment-from-no-return
    ice = enc.encode(ex["is"])
    eng = enc.encode(ex["en"])
    num_edits = editdistance.eval(ice, eng)
    lengths = [len(ice), len(eng)]
    min_ratio = safe_div(num_edits, min(lengths))
    max_ratio = safe_div(num_edits, max(lengths))
    return (max_ratio >= 0.10) and (min_ratio >= 0.10)


@register_filter
def colon_mismatch(ex):
    ice, eng = ex["is"], ex["en"]
    mismatch = ":" in ice and (":" not in eng and ";" not in eng)
    mismatch = mismatch or (":" in eng and ":" not in ice)
    return not mismatch


@register_filter
def corrupt_symbol(ex):
    ice, eng = ex["is"], ex["en"]
    symbols = re.escape("?")
    pat = r"\w[" + symbols + "]+\w"
    prog = RegexCache.compile_rx(pat)
    found = prog.search(ice) or prog.search(eng)
    return not found


@register_filter
def improper_line_split(ex):
    ice, eng = ex["is"], ex["en"]
    ice_prog = RegexCache.compile_rx(r"(\b(?!\d)\w+)- (\b(?!eða|og)\w+\b)")
    ice_matches = ice_prog.findall(ice)
    eng_prog = RegexCache.compile_rx(r"(\b(?!\d)\w+- \b(?!or|and)\w+\b)")
    eng_matches = eng_prog.findall(eng)
    return len(ice_matches) == 0 or len(eng_matches) == 0


@register_filter
def dot_pattern(ex):
    ice, eng = ex["is"], ex["en"]
    prog = RegexCache.compile_rx("(" + r"\.\s+" + "){2,}")
    found = prog.search(ice) or prog.search(eng)
    return not found


@register_filter
def bullet_mismatch(ex):
    # Suggests misalignment since a bullet is a sentence boundary
    ice, eng = ex["is"], ex["en"]
    nice = ice.count("•")
    neng = eng.count("•")
    return nice == neng


@register_filter
def max_word_length(ex, max_word_length=50):
    ice, eng = ex["is"], ex["en"]
    word_prog = RegexCache.compile_rx(r"\b\w+\b")
    ice_words = word_prog.findall(ice)
    eng_words = word_prog.findall(eng)
    max_ice_len = max_word_length
    max_eng_len = max_word_length
    return max(len(w) for w in ice_words) <= max_ice_len and max(len(w) for w in eng_words) <= max_eng_len


@register_filter
def min_word_count(ex):
    try:
        isl_toks = [
            tok for tok in tokenizer.tokenize(ex["is"]) if tok.txt is not None and tok.kind == tokenizer.TOK.WORD
        ]
        eng_toks = [
            tok for tok in tokenizer.tokenize(ex["en"]) if tok.txt is not None and tok.kind == tokenizer.TOK.WORD
        ]
    except TypeError as e:
        return True
    return len(isl_toks) >= DEFAULT_MIN_WORD_COUNT and len(eng_toks) >= DEFAULT_MIN_WORD_COUNT


class Gather:

    _store = None
    _prev_store = None
    _initialized_with_file = False

    @classmethod
    def _idemp_init(cls):
        """ Idempotently initialize, if gather data exists from last run
        it will be used to filter/transform. Otherwise, pipeline needs
        to be executed twice. """
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


@register_filter
def wrong_quotes(ex):
    ice, eng = ex["is"], ex["en"]

    ALLOWED_QUOTES = set("'\"," + "".join(ICE_QUOTE.ALL))
    DISALLOWED_QUOTES = set(QUOTE_LIKE).difference(ALLOWED_QUOTES)

    chars_ex = set(ice)
    chars_ex.update(eng)

    return not bool(chars_ex & DISALLOWED_QUOTES)


# @register_filter
# def language(ex):
#     ice, eng = ex["is"], ex["en"]
#     correct = probably_correct_language(ice, "is", lower_bound=0.995)
#     correct = correct and probably_correct_language(eng, "en", lower_bound=0.995)
#     return correct


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
        num_transformed = sum(count for name, count in cls.counter.items() if name in Transformations._transforms)
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


def do_pipeline(
    in_file=None, out_file=sys.stdout, quiet=False, summary=False, view_function=None, inverted=False, **kwargs
):
    examples = lines_to_examples(in_file)
    # pipeline = Pipeline
    pipeline = MinimalPipeline
    for ex in pipeline.run(examples, view_function=view_function, inverted=inverted):
        if not quiet and view_function is None:
            print("\t".join([ex["en"], ex["is"]]), file=out_file)
    if summary:
        pipeline.summarize_counter()


def lines_to_examples(lines):
    for line in lines:
        line = line.strip("\n")
        src, tgt = line.split("\t")[:2]
        ex = {"en": src, "is": tgt}
        yield ex


class TransformationPipeline(Pipeline):
    _fns = [fix_improper_line_split, remove_leading_bullet, soft_hyphen, replace_dashes, merge_spaces, fix_ice_quotes]


def do_fns(in_file=None, transforms=[], filters=[], quiet=False, **kwargs):
    for ex in lines_to_examples(in_file):
        for transform in transforms:
            ex = Transformations.apply(transform, ex)
        for filter_name in filters:
            inverted = False
            if filter_name.endswith("_inv"):
                inverted = True
                filter_name = filter_name.rstrip("_inv")
            if Filters.apply(filter_name, ex, inverted=inverted):
                if not quiet:
                    print("\t".join([ex["en"], ex["is"]]))


def valid_view_function(string):
    import argparse

    fn_names = [fn.__name__ for fn in Pipeline._fns] + [fn.__name__ for fn in MinimalPipeline._fns]
    if string in fn_names:
        return string
    raise argparse.ArgumentError("Invalid filter/transformation name")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        "Filters and transformation pipelines for monolingual and parallel corpora. "
        "Input defaults to stdin (with tab delimiter) if no input file is supplied."
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
