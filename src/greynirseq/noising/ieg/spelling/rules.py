import pathlib
import re
import string
from dataclasses import dataclass
from typing import Any, Callable

from .fixed_random import coinflip, exp_len, random

LOCAL_DIR = pathlib.Path(__file__).parent.absolute()

"""
List of rules that can be applied to words to create spelling errors.
Each rule has:
    'can_apply' - Returns true iff the rule can be applied to the word
    'apply' - Applies the rule to a word and returns the noised word
    Some bits to track use statistics.
"""
Rules = []


@dataclass
class Rule:
    can_apply: Callable[..., bool]
    apply: Callable
    details: Any = None
    relative_use_ratio: float = 1.0
    used_count: int = 0
    could_use_count: int = 0

    def __str__(self):
        return f"Rule: {self.details} ratio: {self.relative_use_ratio} uses: {self.used_count}/{self.could_use_count}"

    def use_ratio(self):
        return self.used_count / self.could_use_count


DOUBLE_ARROW = "<->"
SINGLE_ARROW = "->"
RATIO_SEPARATOR = "::"


def parse_simple_rule(rule):
    """
    Turn a rule string into a list of (leftside, rightside) pairs that represent the rule.

    The rule string can have these forms:
        a -> b      # a can be turned into b
        a <-> b     # a can be turned into b and b can be turned into a
        a -> b,c    # a can be turned into b or c
        a <-> b,c   # a can be turned into b or c and b and c can be turned into a

    Additionally, at the end of a rule string a relative use ratio may be specified of this form:
        :: x
    where x is a number. This will modify the target use ratio for this rule by multiplying it
    with x.

    Whitespace is not significant (as determined by the python strip function).
    It is not possible to use whitespace or any of ,<-> in a or b.
    Any errors will cause an exception.
    """

    if DOUBLE_ARROW in rule:
        arrow = DOUBLE_ARROW
    elif SINGLE_ARROW in rule:
        arrow = SINGLE_ARROW
    else:
        raise Exception(f"Didn't find an arrow in this line: {rule}")

    arrow_splits = rule.split(arrow)
    assert len(arrow_splits) == 2, f"Found more than one arrow in this line: {rule}"
    left = arrow_splits[0].strip()

    if RATIO_SEPARATOR in arrow_splits[1]:
        ratio_splits = arrow_splits[1].split(RATIO_SEPARATOR)
        assert len(ratio_splits) == 2, f"Found more than one ratio separator in this line: {rule}"
        right = ratio_splits[0].strip()
        ratio = float(ratio_splits[1].strip())
    else:
        right = arrow_splits[1].strip()
        ratio = 1.0

    rights = [s.strip() for s in right.split(",")]

    rules = [(left, r, ratio) for r in rights]
    if arrow == DOUBLE_ARROW:
        rules.extend([(r, left, ratio) for r in rights])

    return rules


# Can't do this inline due to scoping and capture-by-reference rules
def create_simple_rule_apply(left, right):
    def replace_rule(x):
        return x.replace(left, right)

    return replace_rule


def create_simple_rule_can_apply(left):
    def can_apply_rule(x):
        return left in x

    return can_apply_rule


def simple_rules():
    global Rules
    with open(f"{LOCAL_DIR}/rules/simple.txt") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == "#":
                continue
            rules = parse_simple_rule(line)
            for left, right, ratio in rules:
                if len(left.strip()) > 0 and len(right.strip()) > 0:
                    Rules.append(
                        Rule(
                            create_simple_rule_can_apply(left),
                            create_simple_rule_apply(left, right),
                            (left, right),
                            ratio,
                        )
                    )


def create_word_substitution_rule(left, right):
    def replace_rule(x):
        return x.replace(left, right)

    return replace_rule


def word_substitution_rules():
    global Rules
    substitutions = {}
    with open(f"{LOCAL_DIR}/rules/word_pairs_combined.txt") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == "#":
                continue
            rules = parse_simple_rule(line)
            for left, right, ratio in rules:
                if left not in substitutions:
                    substitutions[left] = []
                substitutions[left].append(right)

    def word_sub_can_apply(word):
        return word in substitutions

    def word_sub_apply(word):
        candidates = substitutions[word]
        return candidates[random() % len(candidates)]

    Rules.append(Rule(word_sub_can_apply, word_sub_apply, "word_subsitution", 1.0))


def parse_regex_rule(rule):
    """
    Turn a rule string into a (regex pattern, rightside) pair that represent that rule.

    The rule string should have this form:
        a -> b

    where a is a regular expression and b is a string. If a matches, the match should be
    turned into b.

    Additionally, at the end of a rule string a relative use ratio may be specified of this form:
        :: x
    where x is a number. This will modify the target use ratio for this rule by multiplying it
    with x.

    Whitespace is not significant (as determined by the python strip function).
    It is not possible to use whitespace or any of ,<-> in a or b.
    (warning: something sensible may happen when using these, but that's outside the function contract!)
    Any errors will cause an exception.
    """

    if SINGLE_ARROW not in rule:
        raise Exception(f"Didn't find an arrow in this line: {rule}")

    arrow_splits = rule.split(SINGLE_ARROW)
    assert len(arrow_splits) == 2, f"Found more than one arrow in this line: {rule}"
    left = arrow_splits[0].strip()

    if RATIO_SEPARATOR in arrow_splits[1]:
        ratio_splits = arrow_splits[1].split(RATIO_SEPARATOR)
        assert len(ratio_splits) == 2, f"Found more than one ratio separator in this line: {rule}"
        right = ratio_splits[0].strip()
        ratio = float(ratio_splits[1].strip())
    else:
        right = arrow_splits[1].strip()
        ratio = 1.0

    return (left, right, ratio)


def create_regex_rule_apply(regex, r):
    def regex_rule(word):
        return regex.sub(r, word)

    return regex_rule


def create_regex_rule_can_apply(regex):
    def can_apply_rule(x):
        return regex.search(x)

    return can_apply_rule


def regex_rules():
    global Rules
    with open(f"{LOCAL_DIR}/rules/regex.txt") as f:
        for line in f:
            line = line.strip()
            if len(line) == 0:
                continue
            if line[0] == "#":
                continue
            regex_str, rightside, ratio = parse_regex_rule(line)
            regex = re.compile(regex_str)
            Rules.append(
                Rule(
                    create_regex_rule_can_apply(regex),
                    create_regex_rule_apply(regex, rightside),
                    (regex_str, rightside),
                    ratio,
                )
            )


FLIP_CHANCE = 0.5


def accent_flip():
    global Rules

    more_regex = re.compile("[aeiouyAEIOUY]")

    def flip_more_accents(w):
        def maybe_flip(match):
            ret = match.group(0)
            # Don't flip everything
            if coinflip(1 - FLIP_CHANCE):
                return ret
            ret = ret.replace("a", "á")
            ret = ret.replace("e", "é")
            ret = ret.replace("i", "í")
            ret = ret.replace("o", "ó")
            ret = ret.replace("u", "ú")
            ret = ret.replace("y", "ý")
            ret = ret.replace("A", "Á")
            ret = ret.replace("E", "É")
            ret = ret.replace("I", "Í")
            ret = ret.replace("O", "Ó")
            ret = ret.replace("U", "Ú")
            ret = ret.replace("Y", "Ý")
            return ret

        return more_regex.sub(maybe_flip, w)

    Rules.append(Rule(lambda x: more_regex.search(x), flip_more_accents, "flip_more_accents"))

    less_regex = re.compile("[áéíóúýÁÉÍÓÚÝ]")

    def flip_less_accents(w):
        def maybe_flip(match):
            ret = match.group(0)
            # Don't flip everything
            if coinflip(1 - FLIP_CHANCE):
                return ret
            ret = ret.replace("á", "a")
            ret = ret.replace("é", "e")
            ret = ret.replace("í", "i")
            ret = ret.replace("ó", "o")
            ret = ret.replace("ú", "u")
            ret = ret.replace("ý", "y")
            ret = ret.replace("Á", "A")
            ret = ret.replace("É", "E")
            ret = ret.replace("Í", "I")
            ret = ret.replace("Ó", "O")
            ret = ret.replace("Ú", "U")
            ret = ret.replace("Ý", "Y")
            return ret

        return less_regex.sub(maybe_flip, w)

    Rules.append(Rule(lambda x: less_regex.search(x), flip_less_accents, "flip_less_accents"))


def char_noise():
    global Rules

    noise_chars = string.ascii_lowercase + string.digits + "þæöð´'°.,'"

    def add_random_char(w):
        if coinflip(1 - FLIP_CHANCE):
            # Index to change
            i = random() % len(w)
            # Don't change numbers
            if not w[i].isdigit():
                rand_char_int = random() % len(noise_chars)
                return w[0:i:] + noise_chars[rand_char_int] + w[i::]
        return w

    Rules.append(Rule(lambda x: True, add_random_char, "add_random_char"))

    def replace_random_char(w):
        if coinflip(1 - FLIP_CHANCE):
            # Index to change
            i = random() % len(w)
            # Don't change numbers
            if not w[i].isdigit():
                rand_char_int = random() % len(noise_chars)
                return w[0:i:] + noise_chars[rand_char_int] + w[i + 1 : :]
        return w

    Rules.append(Rule(lambda x: True, replace_random_char, "replace_random_char"))

    def drop_char(w):
        if coinflip(1 - FLIP_CHANCE):
            # don't drop from single letter words
            if len(w) > 1:
                # Letter to drop
                i = random() % len(w)
                # Don't change numbers
                if w[i].isdigit():
                    return w
                else:
                    return w[0:i:] + w[i + 1 : :]
        return w

    Rules.append(Rule(lambda x: True, drop_char, "drop_char"))


def looong():
    global Rules

    def longify(w):
        reps = exp_len(0.3, 2, 3)
        # Letter to repeat
        if len(w) > 0:
            i = random() % len(w)
            # Don't change numbers
            if not w[i].isdigit():
                return w[:i] + w[i] * reps + w[i + 1 :]
        return w

    Rules.append(Rule(lambda x: True, longify, "longify"))


def initialize_rules():
    simple_rules()
    word_substitution_rules()
    regex_rules()
    accent_flip()
    looong()
    char_noise()


initialize_rules()
