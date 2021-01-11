#!/usr/bin/env python

import data_validator
from icecream import ic


schema_ok = {
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
        "noun": [
            "number"
        ]
      },
    "group_names": [
        "degree",
        "number",
        "tense",
    ],
    "group_name_to_labels": {
        "number": [
            "number-single",
            "number-plural",
            "number-empty"
        ],
        "degree": [
            "degree-positive",
            "degree-comparate",
            "degree-superlative",
            "degree-empty"
        ],
        "tense": [
            "tense-present",
            "tense-past",
            "tense-empty"
        ],
      },
    "labels": [
        "<sep>",
        "number-single",
        "number-plural",
        "number-empty",
        "degree-positive",
        "degree-comparate",
        "degree-superlative",
        "degree-empty",
        "tense-present",
        "tense-past",
        "tense-empty",
        "verb",
        "adj",
        "noun"
      ],
    "null": None,
    "null_leaf": None,
    "separator": "<sep>"
}

schema_empty = {}

line_ok = (
        "I buy boat .",
        "verb tense-present number-single <sep> adj degree-empty number-plural <sep> noun number-empty <sep> noun number-plural"
        )

line_count_mismatch = (
        "I buy boat .",
        "verb tense-present number-single <sep> adj degree-empty number-plural <sep> noun number-empty"
        )

line_missing_category = (
        "I buy boat .",
        "tense-present number-single <sep> adj degree-empty number-plural <sep> noun number-empty <sep> noun number-plural"
        )

line_unknown_label = (
        "I buy boat .",
        "verb wobbly-bobbly number-single <sep> adj degree-empty number-plural <sep> noun number-empty <sep> noun number-plural"
        )

def test_good_schema():
    res = data_validator.validate_schema(schema_ok)
    ic(res)
    assert res["ok"] == True
    assert res["problems"] == []


def test_empty_schema():
    res = data_validator.validate_schema(schema_empty)
    assert res["ok"] == False
    assert len(res["problems"]) > 0


def test_good_line():
    res = data_validator.validate_line(line_ok[0], line_ok[1], schema_ok)
    assert res["ok"] == True
    assert res["problems"] == []


def test_bad_line_count_mismatch():
    res = data_validator.validate_line(line_count_mismatch[0], line_count_mismatch[1], schema_ok)
    assert res["ok"] == False
    assert len(res["problems"]) > 0


def test_bad_line_missing_category():
    res = data_validator.validate_line(line_missing_category[0], line_missing_category[1], schema_ok)
    assert res["ok"] == False
    assert len(res["problems"]) > 0


def test_bad_line_unknown_label():
    res = data_validator.validate_line(line_unknown_label[0], line_unknown_label[1], schema_ok)
    assert res["ok"] == False
    assert len(res["problems"]) > 0

