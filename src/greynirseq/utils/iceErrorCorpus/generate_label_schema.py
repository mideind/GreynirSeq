#!/usr/bin/env python


"""
Example v1 schema (from readme file)
{
    "label_categories": [
        "verb",
        "adj",
        "noun",
      ],
    "category_to_group_names": {
        "adj": [
            "degree",
            "number"
        ],
        "verb": [
            "tense",
            "number"
        ],
      },
    "group_names": [
        "degree",
        "number",
        "tense",
    ],
    "group_name_to_labels": {
        "number": [
            "obj1-single",
            "obj1-plural",
            "obj1-empty"
        ],
        "degree": [
            "degree-positive",
            "degree-comparate",
            "degree-superlative",
            "degree-empty"
        ],
        "tense": [
            "obj2-present",
            "obj2-past",
            "obj2-empty"
        ],
      },
    "labels": [
        "<sep>",
        "obj1-single",
        "obj1-plural",
        "obj1-empty"
        "degree-positive",
        "degree-comparate",
        "degree-superlative",
        "degree-empty"
        "obj2-present",
        "obj2-past",
        "obj2-empty"
        "verb",
        "adj",
        "noun",
      ],
    "null": null,
    "null_leaf": null,
    "separator": "<sep>"
}
"""


import argparse
import error_definitions as ed
import json


SEPARATOR="<sep>"


def generate_simple_simcategories_schema_v1():
    return generate_simple_from_dict_v1(ed.SIMCATEGORIES)


def generate_simple_supercategories_schema_v1():
    return generate_simple_from_dict_v1(ed.SUPERCATEGORIES)


def generate_simple_from_dict_v1(categories):
    schema_dict = {
        "null": None,
        "null_leaf": None,
        "separator": SEPARATOR,
    }

    schema_dict["label_categories"] = list(categories.keys())
    schema_dict["category_to_group_names"] = {cat: [] for cat in categories.keys()}
    schema_dict["group_names"] = []
    schema_dict["group_name_to_labels"] = {}
    schema_dict["labels"] = [schema_dict["separator"]] + schema_dict["label_categories"]

    return schema_dict


def generate_supercategories_schema_v1():
    return generate_from_categories_dict(ed.SUPERCATEGORIES)


def generate_simcategories_schema_v1():
    return generate_from_categories_dict(ed.SIMCATEGORIES)


def generate_from_categories_dict(categories):
    schema_dict = {
        "null": None,
        "null_leaf": None,
        "separator": SEPARATOR,
    }

    group_postfix = "-group"  # Need to fix names to fit the schema
    schema_dict["label_categories"] = list(categories.keys())
    schema_dict["category_to_group_names"] = {cat: [cat + group_postfix] for cat in categories.keys()}
    schema_dict["group_names"] = [cat + group_postfix for cat in categories.keys()]
    schema_dict["group_name_to_labels"] = {cat + group_postfix: labels for cat, labels in categories.items()}

    leaf_labels = []
    for group, labels in schema_dict["group_name_to_labels"].items():
        for label_name in labels:
            leaf_labels.append(label_name)

    schema_dict["labels"] = [schema_dict["separator"]] + schema_dict["label_categories"] + leaf_labels

    return schema_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Print label schema file for GreynirSeq multilabel span classification"
    )
    parser.add_argument(
        "-o",
        "--output_file",
        type=str,
        help="Name of the output file.",
        default="grammatical-error-detection.json",
    )
    args = parser.parse_args()

    with open(args.output_file, "w") as f:
        # schema = generate_supercategories_schema_v1()
        # schema = generate_simcategories_schema_v1()
        schema = generate_simple_supercategories_schema_v1()
        # schema = generate_simple_simcategories_schema_v1()
        f.write(json.dumps(schema, sort_keys=True, indent=4))
        f.write("\n")

    print(f"Wrote schema to {args.output_file}")
