#!/usr/bin/env python

import argparse
import sys

from tqdm import tqdm

from .fixed_random import coinflip, exp_len, random, set_random_state
from .rules import Rules

TARGET_USE_RATIO = 0.1

CORRUPT_WORDS = 0
TOTAL_WORDS = 0

ERROR_RATE = 0.5


def should_error(error_rate=ERROR_RATE):
    return coinflip(error_rate)


def errorify_word(word):
    global TOTAL_WORDS

    TOTAL_WORDS += 1

    rule_candidates = []
    for rule in Rules:
        if rule.can_apply(word):
            rule.could_use_count += 1
            if rule.use_ratio() < TARGET_USE_RATIO * rule.relative_use_ratio:
                rule_candidates.append(rule)

    if len(rule_candidates) == 0:
        return word

    global CORRUPT_WORDS
    CORRUPT_WORDS += 1

    # Apply many corruptions in some cases
    for i in range(exp_len(0.2, max_len=len(rule_candidates))):
        chosen_rule = rule_candidates.pop(random() % len(rule_candidates))
        chosen_rule.used_count += 1
        if chosen_rule.can_apply(word):
            # Occasionally rules can no longer be applied after other rules have been applied
            word = chosen_rule.apply(word)

    return word


def errorify_line(line, word_error_rate=ERROR_RATE):

    errorified_words = []
    for word in line.split():
        if should_error(word_error_rate):
            errorified_words.append(errorify_word(word))
        else:
            errorified_words.append(word)
    return " ".join(errorified_words)


def errorify_file(filename):
    with open(filename) as f:
        for line in tqdm(f, total=10000):
            # for line in f:
            print(errorify_line(line))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="File to errorify", required=True)
    parser.add_argument("-s", "--seed", help="Random seed", default=1)
    parser.add_argument(
        "-c",
        "--error-rate",
        help="Probability of a word getting corrupted - not exact",
        type=float,
        default=0.3,
    )
    args = parser.parse_args()

    global ERROR_RATE
    ERROR_RATE = args.error_rate
    global RANDOM_STATE
    set_random_state(args.seed)

    errorify_file(args.input)
    for r in Rules:
        print(
            r,
            "use ratio: ",
            r.used_count / r.could_use_count if r.could_use_count > 0 else float("nan"),
            file=sys.stderr,
        )

    print(f"Corruption ratio: {CORRUPT_WORDS/TOTAL_WORDS}", file=sys.stderr)


if __name__ == "__main__":
    main()
