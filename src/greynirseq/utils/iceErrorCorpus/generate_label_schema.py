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


def generate_on_off_schema_v1():
    return generate_from_categories_dict({"error":None}, False)


def generate_supercategories_schema_v1():
    return generate_from_categories_dict(ed.SUPERCATEGORIES.keys())


def generate_simcategories_schema_v1():
    return generate_from_categories_dict(ed.SIMCATEGORIES.keys())


def generate_from_categories_dict(categories, add_unknown=True):
    if add_unknown:
        categories = list(categories) + ["unknown"]

    schema_dict = {
        "null": None,
        "null_leaf": None,
        "separator": "<sep>",
        "ignore_categories": [],
    }

    group_postfix = "-group"  # Need to fix names to fit the schema
    schema_dict["label_categories"] = ["all"]
    schema_dict["category_to_group_names"] = {"all": [cat + group_postfix for cat in categories]}
    schema_dict["group_names"] = [cat + group_postfix for cat in categories]
    schema_dict["group_name_to_labels"] = {cat + group_postfix: [cat+"-yes", cat+"-no"] for cat in categories}

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
        #schema = generate_supercategories_schema_v1()
        #schema = generate_simcategories_schema_v1()
        schema = generate_on_off_schema_v1()
        f.write(json.dumps(schema, sort_keys=True, indent=4))
        f.write("\n")

    print(f"Wrote schema to {args.output_file}")
