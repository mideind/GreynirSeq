

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


CATEGORY_TO_GROUP_NAMES = {
    "n": ["gender", "number", "case", "def", "proper"],
    "g": ["gender", "number", "case"],
    # "x": [],
    # "e": [],
    # "v": [],
    "l": ["gender", "number", "case", "adj_c", "deg"],
    # pronouns
    "fa": ["gender", "number", "case"],
    "fb": ["gender", "number", "case"],
    "fe": ["gender", "number", "case"],
    # note: fo and fp are special, here number and person are mutually exclusive
    "fo": ["gender", "person", "number", "case"],
    "fp": ["gender", "person", "number", "case"],
    ##
    "fs": ["gender", "number", "case"],
    "ft": ["gender", "number", "case"],
    # Deprecated but needs to be here for backwards compatibility.
    # numerals
    "tf": ["gender", "number", "case"],
    # "ta": [],
    # "tp": [],
    # "to": [],
    # verbs+
    "sn": ["voice"],
    "sb": ["voice", "person", "number", "tense"],  # imp
    "sf": ["voice", "person", "number", "tense"],  # indic.
    "sv": ["voice", "person", "number", "tense"],  # subjunct.
    "ss": ["voice"],  # supine
    "sl": ["voice", "person", "number", "tense"],  # present part.
    "sþ": ["voice", "gender", "number", "case"],  # past part.
    # conjunctions
    # "cn": [],
    # "ct": [],
    # "c": [],
    # adverbs, adpositions, interjections, misc
    "aa": ["deg"],  # no case governing
    "au": ["deg"],  # interjections
    "ao": ["deg"],  # govern acc
    "aþ": ["deg"],  # govern dat
    "ae": ["deg"],  # govern gen
    "as": ["deg"],  # abbreviation
    # punctuation
    # "p": [],
 }


GENDER = {"k": "masc", "v": "fem", "h": "neut", "x": "gender_x"}
NUMBER = {"e": "sing", "f": "plur"}
PERSON = {"1": "1", "2": "2", "3": "3"}
CASE = {"n": "nom", "o": "acc", "þ": "dat", "e": "gen"}
DEGREE = {"f": "pos", "m": "cmp", "e": "superl"}
VOICE = {"g": "act", "m": "mid"}
TENSE = {"n": "pres", "þ": "past"}


GROUP_NAME_TO_IFD_SUBLABEL_TO_NAME = {
    "gender": GENDER,
    "number": NUMBER,
    "person": PERSON,
    "case" : CASE,
    "degree": DEGREE,
    "voice" : VOICE,
    "tense" : TENSE,
}


SUBLABELS = []
GROUP_NAME_TO_LABELS = dict()
for group_name, label_to_name in GROUP_NAME_TO_IFD_SUBLABEL_TO_NAME.items():
    label_list = []
    for label, name in label_to_name.items():
        # label_list.append(f"{group_name}-{name}")  # prevent collisions
        label_list.append(f"{name}")
    GROUP_NAME_TO_LABELS[group_name] = label_list
    SUBLABELS.extend(label_list)


ALL_LABELS = CATEGORIES + SUBLABELS
assert len(ALL_LABELS) == len(set(ALL_LABELS)), "label collision in ifd_label_schema"


def ifd_label_schema():
    return {
        "label_categories": CATEGORIES,
        "category_to_group_names": CATEGORY_TO_GROUP_NAMES,
        "group_names": GROUP_NAMES,
        "group_name_to_labels": GROUP_NAME_TO_LABELS,
        "labels": ALL_LABELS,
    }


if __name__ == '__main__':
    import argparse, json, sys
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
