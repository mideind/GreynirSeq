#!/usr/bin/env python


from typing import Iterable, Dict, Any, List
import argparse
from icecream import ic #type:ignore
import pprint
import json


def validate_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": True, "problems": []}

    top_level: List[str] = ["label_categories", "category_to_group_names", "group_names", "group_name_to_labels",
            "labels", "null", "null_leaf", "separator"]

    if sorted(top_level) != sorted(schema.keys()):
        result["ok"] = False
        result["problems"].append({"desc": "Incorrect top level schema keys", "expected":top_level, "got":schema.keys(),
            "extra_keys": set(schema.keys()) - set(top_level), "missing_keys": set(top_level) - set(schema.keys())})
        # Return now since the rest of the tests assume the required schema keys exist
        return result

    for key in ["label_categories", "category_to_group_names", "group_names", "group_name_to_labels", "labels"]:
        if len(schema[key]) == 0:
            result["ok"] = False
            result["problems"].append({"desc": "Top level list/dict empty", "key": key})

    if sorted(schema["label_categories"]) != sorted(schema["category_to_group_names"].keys()):
        result["ok"] = False
        result["problems"].append({"desc": "label_categories does not match category_to_group_names"})

    if sorted(schema["group_names"]) != sorted(schema["group_name_to_labels"].keys()):
        result["ok"] = False
        result["problems"].append({"desc": "group_names does not match group_name_to_labels"})

    labels_in_groups = []
    for g in schema["group_name_to_labels"].values():
        labels_in_groups.extend(g)
    implied_labels = [schema["separator"]] + schema["label_categories"] + labels_in_groups

    if sorted(implied_labels) != sorted(schema["labels"]):
        result["ok"] = False
        result["problems"].append({"desc": "Schema labels list does not match the implied labels list",
            "labels": schema["labels"], "implied": implied_labels})

    return result


def validate_line(input_line: str, label_line: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {"ok": True, "problems": []}

    input_toks = input_line.split(' ')
    label_toks = label_line.split(schema["separator"])

    if len(input_toks) != len(label_toks):
        result["ok"] = False
        result["problems"].append({"desc": "Token count mismatch"})

    # Each word should have exactly one category (this assumption may change)
    for labels_for_one_word in label_toks:
        single_labels = labels_for_one_word.split(' ')
        found_categories = 0
        for cat in schema["label_categories"]:
            if cat in single_labels:
                found_categories += 1
        if found_categories == 0:
            result["ok"] = False
            result["problems"].append({"desc": "Missing category on word", "labels": labels_for_one_word})
        if found_categories > 1:
            result["ok"] = False
            result["problems"].append({"desc": "Too many categories on word", "labels": labels_for_one_word})

    # All labels should be in the schema
    for labels_for_one_word in label_toks:
        for l in labels_for_one_word.split():
            if l not in schema["labels"]:
                result["ok"] = False
                result["problems"].append({"desc": "Label not in schema", "label": l})

    return result


def validate(input_lines: List[str], label_lines: List[str], schema: Dict[str, Any], to_stdout=True) -> Dict[str, Any]:
    results = {}
    results["schema"] = validate_schema(schema)

    results["lines"] = {"ok": True, "problems": []}
    if len(input_lines) != len(label_lines):
        results["lines"]["ok"] = False
        results["lines"]["problems"].append({"desc": "input and labels not of the same length"})
        return results

    print(f"About to check {len(input_lines)} lines")
    count = 1
    for i, l in zip(input_lines, label_lines):
        r = validate_line(i, l, schema)
        if not r["ok"]:
            results["lines"]["ok"] = False
            results["lines"]["problems"].append({"lineno": count, "problems": r["problems"]})
        count += 1
        if count % 100 == 0:
            print(f" > {count}")

    print("Done processing lines")

    return results


def summarize(error_dict: Dict[str, Any]) -> str:
    if error_dict["schema"]["ok"]:
        schema_error = "No schema errors (yay!)"
    else:
        schema_error = "Schema errors:\n"
        schema_error += f'{pprint.pformat(error_dict["schema"]["problems"], indent=4)}'

    line_error_count = 0
    line_error_types = []

    if error_dict["lines"]["ok"]:
        line_error = "No line errors (yay!)"
    else:
        line_error = "Line errors (1-indexed):\n"
        problems = {}
        for problems_in_line in error_dict["lines"]["problems"]:
            for problem in problems_in_line["problems"]:
                prob_type = problem["desc"]
                if not prob_type in problems:
                    problems[prob_type] = {"desc": prob_type, "count": 0, "example": problem, "example_lineno": problems_in_line["lineno"]}
                problems[prob_type]["count"] += 1
        line_error += f"{pprint.pformat(problems, indent=4)}"

    return f"{schema_error}\n{line_error}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Validate a pair of input and label files against a schema",
            )

    parser.add_argument("input0", type=str, help="Path to the input file")
    parser.add_argument("label", type=str, help="Path to the label file")
    parser.add_argument("schema", type=str, help="Path to the schema file")

    args = parser.parse_args()

    input0_file = open(args.input0)
    label_file = open(args.label)
    schema = json.load(open(args.schema))


    print(summarize(validate(input0_file.readlines(), label_file.readlines(), schema)))
