# Simple generalized label schema
This describes the generalized json label schema that IceBERT uses for many subtasks.

There is a list of all legal labels in the schema, `labels`, this defines the canonical ordering for indices.
There are primary labels called `label_categories`, e.g. noun, adjective.
Each primary label can *optionally* have multiple sublabels. These sublabels are organized into label groups (labels within a group are mutually exclusive).
What groups a label category has is defined in `category_to_group_names`.
If a label group applies to a given primary label it must have exactly one sublabel from each group, each group has a null label for flexibility (post-fixed with `-empty`).
There is a canonical ordering (exhaustive listing) of all label groups in `group_names`.

In an annotation file, a sequence consists of multiple instances of a primary label followed by its sublabels where all labels are separated by a space.
There are a couple of special labels for some subtasks (null and null_leaf), that have to do with constituency parsing and do not yet have general semantics.

```
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
```
