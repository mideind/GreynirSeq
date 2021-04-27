#!/usr/bin/env python3
# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

# flake8: noqa

import enum
import itertools

import numpy as np

GENDER = list(range(4))
PER = list(range(4, 7))
GENDER_OR_PER = GENDER + PER
NUMBER = list(range(7, 9))
CASE = list(range(9, 13))
DEF = list(range(13, 14))
PROP = list(range(14, 15))
ADJ_C = list(range(15, 18))
DEG = list(range(18, 21))
A_DEG = list(range(19, 21))  # adjective not pos
TEN = list(range(21, 24))
VOI = list(range(24, 26))


LABEL_GROUPS = [
    GENDER,
    PER,
    GENDER_OR_PER,
    NUMBER,
    CASE,
    DEF,
    PROP,
    ADJ_C,
    DEG,
    TEN,
    VOI,
]


def groups_to_label(groups):
    return [LABEL_GROUPS.index(grp) for grp in groups]


CATS_W_LABELS = [
    ("n", [GENDER, NUMBER, CASE, DEF, PROP]),
    ("g", [GENDER, NUMBER, CASE]),
    ("x", []),
    ("e", []),
    ("v", []),
    ("l", [GENDER, NUMBER, CASE, ADJ_C, DEG]),
    # pronouns
    ("fa", [GENDER, NUMBER, CASE]),
    ("fb", [GENDER, NUMBER, CASE]),
    ("fe", [GENDER, NUMBER, CASE]),
    ("fo", [GENDER_OR_PER, NUMBER, CASE]),
    ("fp", [GENDER_OR_PER, NUMBER, CASE]),
    ("fs", [GENDER, NUMBER, CASE]),
    (
        "ft",
        [GENDER, NUMBER, CASE],
    ),  # Deprecated but needs to be here for backwards compatibility.
    # numerals
    ("tf", [GENDER, NUMBER, CASE]),
    ("ta", []),
    ("tp", []),
    ("to", []),
    # verbs+
    ("sn", [VOI]),
    ("sb", [VOI, PER, NUMBER, TEN]),  # imp
    ("sf", [VOI, PER, NUMBER, TEN]),  # indic.
    ("sv", [VOI, PER, NUMBER, TEN]),  # subjunct.
    ("ss", [VOI]),  # supine
    ("sl", [VOI, PER, NUMBER, TEN]),  # present part.
    ("sþ", [VOI, GENDER, NUMBER, CASE]),  # past part.
    # conjunctions
    ("cn", []),
    ("ct", []),
    ("c", []),
    # adverbs, adpositions, interjections, misc
    ("aa", [DEG]),  # no case governing
    ("af", [DEG]),  #
    ("au", [DEG]),  # interjections
    ("ao", [DEG]),  # govern acc
    ("aþ", [DEG]),  # govern dat
    ("ae", [DEG]),  # govern gen
    ("as", [DEG]),  # abbreviation
    ("ks", []),
    ("kt", []),
    # punctuation
    ("p", []),
    ("pl", []),
    ("pk", []),
    ("pg", []),
    ("pa", []),
    ("ns", []),
    ("m", []),
]
CATS = [c[0] for c in CATS_W_LABELS]
CAT_GROUPS = [groups_to_label(c[1]) for c in CATS_W_LABELS]

FEATS = [
    "masc",
    "fem",
    "neut",
    "gender_x",
    "1",  # person
    "2",
    "3",
    "sing",
    "plur",
    "nom",
    "acc",
    "dat",
    "gen",
    "definite",  # noun with clitic article
    "proper",
    "strong",
    "weak",
    "equiinflected",  # "óbeygt"
    "pos",  # positive degree
    "cmp",
    "superl",
    "past",
    "pres",
    "pass",  # Not used but needs to be here for backwards compatibility.
    "act",
    "mid",
    # "supine",
    # NOTE: moods and verbforms are part of in the cat list above
    # "part",
    # "ind",
    # "sub",
    # "imp",
    # "inf",
]

FEATS_MUTEX = {
    "gen": [
        "masc",
        "fem",
        "neut",
        "gender_undet",
    ],
    "per": [
        "1",
        "2",
        "3",
    ],  # person
    "num": [
        "sing",
        "plur",
    ],
    "cas": [
        "nom",
        "acc",
        "dat",
        "gen",
    ],
    "def": [
        "definite",
    ],
    "pro": ["proper"],
    "dec": [
        "strong",
        "weak",
        "equiinflected",
    ],  # "óbeygt"
    "deg": [
        "pos",
        "cmp",
        "superl",
    ],  # positive degree
    "ten": [
        "past",
        "pres",
        "pass",
    ],  # Not used but needs to be here for backwards compatibility.
    "voi": [
        "act",
        "mid",
    ],
}
FEATS_MUTEX_MAP = {}
FEATS_MUTEX_MAP_IDX = {}
for feat, i in enumerate(FEATS):
    for feat_map in FEATS_MUTEX:
        if feat in FEATS_MUTEX[feat_map]:
            FEATS_MUTEX_MAP[feat] = FEATS_MUTEX[feat_map]
            FEATS_MUTEX_MAP_IDX[FEATS.index(feat)] = [FEATS.index(f) for f in FEATS_MUTEX[feat_map]]


LABELS = CATS + FEATS
LABEL_TO_IDX = {label: idx for (idx, label) in enumerate(LABELS)}
DIM = len(CATS) + len(FEATS)

assert DIM == len(set(LABELS)), "tag collision"


GENDER = {"k": "masc", "v": "fem", "h": "neut", "-": "gender_x"}
NUMBER = {"e": "sing", "f": "plur"}
PERSON = {"1": "1", "2": "2", "3": "3"}
CASE = {"n": "nom", "o": "acc", "þ": "dat", "e": "gen"}
DEGREE = {"f": "pos", "m": "cmp", "e": "superl"}
VOICE = {"g": "act", "m": "mid"}
TENSE = {"n": "pres", "þ": "past"}
# MOOD = {"l":"part", "f":"ind", "v":"sub", "b":"imp", "n":"inf", "þ":"part"}
# PRONTYPE = {
#     # "name":"PronType",
#     "a":"dem",  # ábendingarfornafn
# ic(pred_idxs)
# ic(target_idxs)
# assert n_pred_idxs == len(target_idxs)
#     "p":"prs",  # persónufornafn
#     "s":"int",  # spurnarfornafn
#     "t": None,  # tilvísunarfornafn  # er þetta enn í notkun?
# }
# }
ADJ_CLASS = {"s": "strong", "v": "weak", "o": "equiinflected"}
DEFINITE = {"g": "definite", " ": "indefinite"}

TAGSET = {
    "n": [
        GENDER,
        NUMBER,
        CASE,
        {"g": "definite", "-": "", " ": ""},
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


def one_hot(idx, dim=DIM):
    vec = np.zeros(idx)
    vec[idx] = 1
    return vec


def get_cat_idx(tag):
    raise ValueError(f"Invalid tag '{tag}'")


def ifd2fine(tag):
    if not tag[0].isalpha():
        return "p"
    if tag.startswith("l"):
        assert tag[-1] in "fme"
        return "l" + tag[-1]  # degree
    elif tag.startswith("g"):
        return tag[0]
    return tag[:2]


def ifd2coarse(tag):
    if not tag[0].isalpha():
        return "p"
    elif tag.startswith("sþ"):
        return "sþ"
    return tag[0]


foreign_name = "n----s"


def ifd2labels(tag):
    if not tag[0].isalpha():
        return ["p"]

    if tag == foreign_name:
        return ["ns"]

    tagset_key = tag[0]
    cat = tag[0]
    tagset_key = tag[0]
    rest = tag[1:]
    if tag[0] in "csftapk":
        cat = tag[:2]
        rest = tag[2:]
        tagset_key = "sþ" if tag.startswith("sþ") else tagset_key

    labels = []
    labels.append(cat)

    if tagset_key not in TAGSET:
        return labels

    if tagset_key == "a" and not rest:
        label = "pos"
        labels.append(label)

    for feature, code in zip(TAGSET[tagset_key], rest):
        try:
            label = feature[code]
        except:
            import pdb

            pdb.set_trace()
        if not label:
            continue
        labels.append(label)

    return labels


def ifd2labelsNew(tag):
    # Move to separete file
    GENDER = {"k": "masc", "v": "fem", "h": "neut"}
    NUMBER = {"e": "sing", "f": "plur"}
    PERSON = {"1": "1", "2": "2", "3": "3"}
    CASE = {"n": "nom", "o": "acc", "þ": "dat", "e": "gen"}

    DEGREE = {"f": "pos", "m": "cmp", "e": "superl"}

    VOICE = {"g": "act", "m": "mid"}
    TENSE = {"n": "pres", "þ": "past"}
    ADJ_CLASS = {"s": "strong", "v": "weak", "o": "equiinflected"}
    DEFINITE = {"g": "definite", " ": "indefinite"}

    TAGSET = {
        "n": [
            GENDER,
            NUMBER,
            CASE,
            {"g": "definite", "-": "", " ": ""},
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
    tagset_key = tag[0]
    cat = tag[0]
    tagset_key = tag[0]
    rest = tag[1:]
    if tag[0] in "csfta":
        cat = tag[:2]
        rest = tag[2:]
        tagset_key = "sþ" if tag.startswith("sþ") else tagset_key

    labels = []
    labels.append(cat)

    if tagset_key not in TAGSET:
        return labels

    if tagset_key == "a" and not rest:
        label = "pos"
        labels.append(label)

    for feature, code in zip(TAGSET[tagset_key], rest):
        label = feature[code]

        if not label:
            continue
        labels.append(label)

    return labels


def ifd2vec(tag):
    vec = np.zeros(DIM)
    labels = ifd2labels(tag)
    for label in labels:
        vec[LABEL_TO_IDX[label]] = 1
    return vec, labels


def vec2ifd(vec):
    cat_idx = np.argmax(vec[: len(CATS)])
    cat = CATS[cat_idx]
    idxs = list(np.where(vec == 1)[0])
    features = [LABELS[int(idx)] for idx in idxs if int(idx) >= len(CATS)]
    if not features:
        return cat
    ret = []
    codes = []
    tagset_key = cat[0]
    tagset_key = "sþ" if cat.startswith("sþ") else tagset_key
    for feature in TAGSET[tagset_key]:
        for code, val in feature.items():
            if val in features:
                ret.append(val)
                codes.append(code)
    tag = "".join([cat] + codes)

    if cat == "n" and "proper" in features and "definite" not in features:
        tag = tag[:-1] + "-" + tag[-1]
    return tag


# def test_ifd2vec():
#     import prep_mim

#     for tag in prep_mim.GOLD_TAGS:
#         ifd2vec(tag)
#         assert True


# def test_vec2idf():
#     import prep_mim

#     tag = "fahee"
#     vec, labels = ifd2vec(tag)
#     ret = vec2idf(vec)
#     assert ret == tag
#     for tag in prep_mim.GOLD_TAGS:
#         vec, labels = ifd2vec(tag)
#         ret = vec2idf(vec)
#         assert ret == tag

# def test_ifd2coarse():
#     found = set(ifd2coarse(tag) for tag in GOLD_TAGS)
#     assert "p" not in found
#     found.add("p")  # punct tag is not in GOLD_TAGS
#     all_ = set(IFD_COARSE_TAGSET)
#     assert len(all_) == len(IFD_COARSE_TAGSET)
#     assert all_ == found, "Missing coarse tags"

# def test_ifd2fine():
#     found = set(ifd2fine(tag) for tag in GOLD_TAGS)
#     assert "p" not in found
#     found.add("p")  # punct tag is not in GOLD_TAGS
#     gold_all = set(IFD_FINE_TAGSET)
#     # from collections import Counter
#     # ctr = Counter()
#     # ctr.update(IFD_FINE_TAGSET)
#     # print(list(ctr.most_common()))
#     assert len(gold_all) == len(IFD_FINE_TAGSET)  # punct is extra
#     assert gold_all == found, "Missing fine tags"
