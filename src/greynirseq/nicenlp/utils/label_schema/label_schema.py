from collections import namedtuple
import json

from fairseq.data import Dictionary
import torch
import torch.nn.functional as F


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
        ],
    )
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


def make_map_cat_to_dict(ldict, schema, device="cpu"):
    map_cat_to_dict = torch.zeros(len(schema.labels), device=device, dtype=torch.long)
    for vec_idx, lbl in enumerate(schema.labels):
        dict_idx = ldict.index(lbl)
        map_cat_to_dict[vec_idx] = dict_idx
    return map_cat_to_dict


def make_bos_mask(nwords_w_bos):
    zero = torch.tensor(0)
    bos_mask = torch.cat(
        [
            (F.one_hot(zero, seq_nwords_w_bos))
            for seq_nwords_w_bos in nwords_w_bos.tolist()
        ],
        0,
    )
    return bos_mask


def make_vec_idx_to_dict_idx(dictionary, labels, device="cpu", fill_value=-100):
    map_vec_to_dict = torch.full(
        (len(dictionary),), device=device, fill_value=fill_value, dtype=torch.long
    )
    for vec_idx, label in enumerate(labels):
        map_vec_to_dict[vec_idx] = dictionary.index(label)
    return map_vec_to_dict


def make_group_masks(schema, dictionary, device="cpu"):
    num_groups = len(schema.group_names)
    label_shift = dictionary.nspecial
    num_labels = len(dictionary) - label_shift
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


def make_mapped_group_masks(schema, ldict, device="cpu"):
    num_groups = len(schema.group_names)
    num_labels = len(ldict)
    ret_mask = torch.zeros(num_labels, num_groups, dtype=torch.int64, device=device)
    for cat, group_names in schema.category_to_group_names.items():
        cat_lbl_idx = ldict.index(cat)
        for group_name in group_names:
            group_idx = schema.group_names.index(group_name)
            ret_mask[cat_lbl_idx, group_idx] = 1
    return ret_mask


def make_group_name_to_mapped_group_idxs(dictionary, group_name_to_labels):
    group_names = group_name_to_labels.keys()
    lshift = dictionary.nspecial
    group_name_to_map_vec_idxs = {
        gname: torch.tensor(
            [dictionary.index(gitem) - lshift for gitem in group_name_to_labels[gname]]
        )
        for gname in group_names
    }
    return group_name_to_map_vec_idxs


def make_dict_idx_to_vec_idx(dictionary, cats, device="cpu", fill_value=-100):
    # NOTE: when target is not in label_categories, the error is silent
    map_tgt = torch.full(
        (len(dictionary),), device=device, fill_value=fill_value, dtype=torch.long
    )
    for vec_idx, label in enumerate(cats):
        map_tgt[dictionary.index(label)] = vec_idx
    return map_tgt
