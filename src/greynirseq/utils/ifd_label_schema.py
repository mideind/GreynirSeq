# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

# flake8: noqa

# defined in two places for backwards compatibility / stable canonical ordering
GROUP_NAMES = [
    "gender",
    "number",
    "case",
    "def",
    "proper",
    "adj_c",
    "deg",
    "voice",
    "person",
    "number",
    "tense",
]


CATEGORIES = [
    "n",
    "g",
    "x",
    "e",
    "v",
    "l",
    # pronouns
    "fa",
    "fb",
    "fe",
    "fo",
    "fp",
    "fs",
    "ft",
    # Deprecated but needs to be here for backwards compatibility.
    # numerals
    "tf",
    "ta",
    "tp",
    "to",
    # verbs+
    "sn",
    "sb",
    "sf",
    "sv",
    "ss",
    "sl",
    "sþ",
    # conjunctions
    "cn",
    "ct",
    "c",
    # adverbs, adpositions, interjections, misc
    "aa",
    "au",
    "ao",
    "aþ",
    "ae",
    "as",
    # punctuation
    "p",
]

# from mim_tagset_files
CATEGORY_TO_CATEGORY_NAME = {
    "n": "noun",
    "g": "article",
    "x": "unspecified",
    "e": "foreign",
    "v": "url",
    "l": "adjective",
    "g": "def article",
    # pronouns
    "fa": "demonstrative pron",
    "fb": "indef demonstrative",
    "fe": "possessive pron",
    "fo": "indef pron",
    "fp": "personal pron",
    "fs": "interr pron",
    "ft": "relative pron",
    # Deprecated but needs to be here for backwards compatibility.
    # numerals
    "tf": "cardinal num",
    "ta": "date and misc. num",
    "tp": "percentage",
    "to": "number prec. numeral",  # e.g. one third
    # verbs+
    "sn": "infinitive",
    "sb": "imperative",
    "sf": "indicative",
    "sv": "subjunctive",
    "ss": "supine",
    "sl": "present part.",
    "sþ": "past part.",
    # conjunctions
    "cn": "infinitive symbol",
    "ct": "relative junction",
    "c": "conjunction",
    # adverbs, adpositions, interjections, misc
    "aa": "adv governs none",
    "au": "exclamation",
    "ao": "adv governs acc",
    "aþ": "adv governs dat",
    "ae": "adv governs gen",
    "as": "---",  # missing from doc
    # punctuation
    "p": "punctuation",
}

CATEGORY_TO_GROUP_NAMES = {
    "n": ["gender", "number", "case", "def", "proper"],
    "g": ["gender", "number", "case"],
    "l": ["gender", "number", "case", "adj_c", "deg"],
    # pronouns
    "fa": ["gender", "number", "case"],
    "fb": ["gender", "number", "case"],
    "fe": ["gender", "number", "case"],
    "fs": ["gender", "number", "case"],
    "ft": ["gender", "number", "case"],
    # note: fo and fp are special, here number and person are actually mutually exclusive
    "fo": ["gender", "person", "number", "case"],
    "fp": ["gender", "person", "number", "case"],
    # numerals
    "tf": ["gender", "number", "case"],
    # verbs
    "sn": ["voice"],
    "sb": ["voice", "person", "number", "tense"],  # imp
    "sf": ["voice", "person", "number", "tense"],  # indic.
    "sv": ["voice", "person", "number", "tense"],  # subjunct.
    "ss": ["voice"],  # supine
    "sl": ["voice", "person", "number", "tense"],  # present part.
    "sþ": ["voice", "gender", "number", "case"],  # past part.
    # adverbs, adpositions, interjections, misc
    "aa": ["deg"],  # no case governing
    "au": ["deg"],  # interjections
    "ao": ["deg"],  # govern acc
    "aþ": ["deg"],  # govern dat
    "ae": ["deg"],  # govern gen
    "as": ["deg"],  # abbreviation
}


GENDER = {"k": "masc", "v": "fem", "h": "neut", "x": "unspec", "": "gend-empty"}
NUMBER = {"e": "sing", "f": "plur"}
PERSON = {"1": "p1", "2": "p2", "3": "p3", "": "per-empty"}
CASE = {"n": "nom", "o": "acc", "þ": "dat", "e": "gen"}
DEGREE = {"f": "pos", "m": "cmp", "e": "superl"}
VOICE = {"g": "act", "m": "mid"}
TENSE = {"n": "pres", "þ": "past"}
ADJ_CLASS = {"s": "strong", "v": "weak", "o": "fixed"}
# DEFINITE = {"g": "def", "-": "indef"}
DEFINITE = {"g": "def", "-": "", "": ""}
PROPER = {"": "", "s": "proper"}


GROUP_NAME_TO_IFD_SUBLABEL_TO_NAME = {
    "gender": GENDER,
    "number": NUMBER,
    "person": PERSON,
    "case": CASE,
    "degree": DEGREE,
    "voice": VOICE,
    "tense": TENSE,
    "def": DEFINITE,
    "proper": PROPER,
}


SUBLABELS = []
GROUP_NAME_TO_LABELS = dict()
for group_name, label_to_name in GROUP_NAME_TO_IFD_SUBLABEL_TO_NAME.items():
    label_list = []
    for label, name in label_to_name.items():
        if name:
            label_list.append(f"{name}")
    GROUP_NAME_TO_LABELS[group_name] = label_list
    SUBLABELS.extend(label_list)


ALL_LABELS = CATEGORIES + SUBLABELS

seen = set()
collisions = []
for lbl in ALL_LABELS:
    if lbl not in seen:
        seen.add(lbl)
    else:
        collisions.append(lbl)
assert not collisions, f"label collision in ifd_label_schema: {collisions}"


IFD_TAGSET = {
    "n": [
        GENDER,
        NUMBER,
        CASE,
        # {"g": "def", "-": "indef", " ": ""},
        {"g": "def", "-": "", " ": ""},
        {"": "", "s": "proper"},
    ],
    "l": [GENDER, NUMBER, CASE, ADJ_CLASS, DEGREE],
    # we skip PRONTYPE, as thats part of fine categories
    "f": [{**GENDER, **PERSON}, NUMBER, CASE],
    "g": [GENDER, NUMBER, CASE],
    # we skip NUMTYPE, as thats part of fine categories, this only applies to 'tf'
    "t": [GENDER, NUMBER, CASE],
    # NOTE:
    "sþ": [VOICE, GENDER, NUMBER, CASE],
    # we skip MOOD, as thats part of fine categories
    "s": [VOICE, PERSON, NUMBER, TENSE],
    # we skip subcategories as thats part of fine categories
    "a": [DEGREE],
}


def ifd_tag_to_schema(tag):
    if not tag[0].isalpha():
        return ["p"]

    tagset_key = tag[0]
    category = tag[0]
    subfields = tag[1:]
    if tag[0] in "csfta":
        category = tag[:2]
        subfields = tag[2:]
    if tag.startswith("sþ"):
        tagset_key = "sþ" if tag.startswith("sþ") else tagset_key

    labels = []
    labels.append(category)

    if tagset_key not in IFD_TAGSET:
        return labels

    for feature, code in zip(IFD_TAGSET[tagset_key], subfields):
        sublabel = feature[code]
        if not sublabel:
            continue
        labels.append(sublabel)

    if category in ("fo", "fb") and set(labels).intersection(set("kvh")):
        labels.append("per-empty")
    elif category in ("fo", "fb"):
        # one of [p1, p2, p3] is in labels
        labels.append("gend-empty")

    return labels


def ifd_label_schema():
    return {
        "label_categories": CATEGORIES,
        "category_to_group_names": CATEGORY_TO_GROUP_NAMES,
        "group_names": GROUP_NAMES,
        "group_name_to_labels": GROUP_NAME_TO_LABELS,
        "labels": ALL_LABELS,
    }


if __name__ == "__main__":
    import argparse
    import json
    import sys
    from pathlib import Path

    try:
        import argcomplete
    except ImportError as e:
        pass
    parser = argparse.ArgumentParser("Description")

    parser.add_argument(
        "output",
        type=str,
        metavar="FILE",
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="Force overwrite",
    )

    args = parser.parse_args()
    path = Path(args.output)
    if path.exists() and not args.force:
        print("Error: output file already exists")
        sys.exit(0)

    obj = ifd_label_schema()
    with path.open("w") as handle:
        json.dump(obj, handle, indent=4, ensure_ascii=False)
