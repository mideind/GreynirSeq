# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.


import json
from collections import namedtuple

import torch
from fairseq import file_utils
from fairseq.data import Dictionary


def parse_label_schema(path):
    LabelSchema = namedtuple(
        "LabelSchema",
        [
            "labels",
            "group_name_to_labels",
            "label_categories",
            "category_to_group_names",
            "separator",
            "group_names",
            "null",
            "null_leaf",
            "ignore_categories",
        ],
    )
    path = file_utils.cached_path(path)
    with open(path, "r") as fp:
        j_obj = json.load(fp)
    return LabelSchema(**j_obj)


def label_schema_as_dictionary(label_schema):
    label_dict = Dictionary()

    labels = list(label_schema.labels)
    assert len(labels) == len(set(labels))

    for label in labels:
        label_dict.add_symbol(label)

    return label_dict


def make_vec_idx_to_dict_idx(dictionary, labels, device="cpu", fill_value=-100):
    vec_idx_to_dict_idx = torch.full((len(dictionary),), device=device, fill_value=fill_value, dtype=torch.long)
    for vec_idx, label in enumerate(labels):
        vec_idx_to_dict_idx[vec_idx] = dictionary.index(label)
    return vec_idx_to_dict_idx


def make_group_masks(dictionary, schema, device="cpu"):
    num_groups = len(schema.group_names)
    offset = dictionary.nspecial
    num_labels = len(dictionary) - offset
    ret_mask = torch.zeros(num_labels, num_groups, dtype=torch.int64, device=device)
    for cat, cat_group_names in schema.category_to_group_names.items():
        cat_label_idx = dictionary.index(cat)
        cat_vec_idx = schema.label_categories.index(cat)
        for group_name in cat_group_names:
            ret_mask[cat_vec_idx, schema.group_names.index(group_name)] = 1
        assert cat_label_idx != dictionary.unk()
    for cat in schema.label_categories:
        cat_label_idx = dictionary.index(cat)
        assert cat_label_idx != dictionary.unk()
    return ret_mask


def make_group_name_to_group_attr_vec_idxs(dict_, schema):
    offset = dict_.nspecial
    group_names = schema.group_name_to_labels.keys()
    name_to_labels = schema.group_name_to_labels
    group_name_to_group_attr_vec_idxs = {
        name: torch.tensor([dict_.index(item) - offset for item in name_to_labels[name]]) for name in group_names
    }
    return group_name_to_group_attr_vec_idxs


def make_dict_idx_to_vec_idx(dictionary, cats, device="cpu", fill_value=-100):
    # NOTE: when target is not in label_categories, the error is silent
    map_tgt = torch.full((len(dictionary),), device=device, fill_value=fill_value, dtype=torch.long)
    for vec_idx, label in enumerate(cats):
        map_tgt[dictionary.index(label)] = vec_idx
    return map_tgt
