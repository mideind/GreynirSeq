#!/usr/bin/env python3
import json
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass

from icecream import ic

from greynirseq.nicenlp.utils.constituency.icepahc_utils import CASES, LEGAL_NT_FLAGS, LEGAL_NT_HEADS, LEGAL_POS_HEADS

# LEGAL_ICEPAHC_POS_PARTS = (
#     # verbs
#     "BE",
#     "DO",
#     "HV",
#     "MD",
#     "RD",
#     "VB",
#     "BEPI",
#     "BEPS",
#     "BEDI",
#     "BEDS",
#     "BEI",
#     "BAG",
#     "BEN",
#     "BAN",
#     "DOPI",
#     "DOPS",
#     "DODI",
#     "DODS",
#     "DOI",
#     "DAG",
#     "DON",
#     "DAN",
#     "HVPI",
#     "HVPS",
#     "HVDI",
#     "HVDS",
#     "HVI",
#     "HAG",
#     "HVN",
#     "HAN",
#     "MDPI",
#     "MDPS",
#     "MDDI",
#     "MDDS",
#     "MDI",
#     "MAG",
#     "MDN",
#     "MAN",
#     "RDPI",
#     "RDPS",
#     "RDDI",
#     "RDDS",
#     "RDI",
#     "RAG",
#     "RDN",
#     "RAN",
#     "VBPI",
#     "VBPS",
#     "VBDI",
#     "VBDS",
#     "VBI",
#     "VAG",
#     "VBN",
#     "VAN",
#     # adverbs
#     "ADV",
#     "WADV",
#     "ADVR",
#     "ADVS",
#     # adjectives
#     "ADJ",
#     "ADJR",  # comparative
#     "ADJS",  # superlative
#     "WADJ",  # wh-adjective
#     "SUCH",
#     #
#     # conjunctions
#     "CONJ",
#     "C",
#     # determiners
#     "D",
#     "WD",  # wh-determiner
#     # nouns
#     "N",
#     "NS",
#     "NPR",
#     "NPRS",
#     "ONE",
#     "ONES",
#     # particles
#     "FP",  # focus particle
#     "RP",  # adverbial particle
#     "RPX",  # adverbial particle
#     # pronouns
#     "PRO",
#     "WPRO",
#     # prepositions
#     "P",
#     # quantifiers
#     "Q",  # 'fáir'
#     "QR",  # 'færri'
#     "QS",  # 'fæstir'
#     "WQ",  # 'hver'
#     # miscellaneous
#     "LS",  # list marker
#     "FW",  # foreign word
#     #####
#     "TO",
#     "NEG",
#     "ALSO",
#     "NUM",
#     "ES",
#     "INTJ",
#     "OTHER",
#     "OTHERS",
#     # shared modifers
# )
# CASES = ("N", "A", "D", "G")
# # MISC = ("TTT", "OB1", "OB2", "LB")  # line break
# # SKIP_PARTS = ("SBJ", "PRD", "IP", "INF", "NP", "WADVP", "ADJP", "CONJP", "RX", "FOREIGN", "VDPI", "PRN", "NRP", "SPE")
# # LEGAL_ICEPAHC_POS_PARTS = set(LEGAL_ICEPAHC_POS_PARTS + CASES + MISC + SKIP_PARTS)

with open("/tmp/icepahc_nt_tags.json", "r") as fh_in:
    nt_list = json.load(fh_in)

with open("/tmp/icepahc_pos_tags.json", "r") as fh_in:
    pos_list = json.load(fh_in)


def maybe_fix_pos(pos_str):
    if pos_str == "ŃUM":
        pos_str = pos_str.replace("ŃUM", "NUM")
    if pos_str == "FÐ":
        return "FP"
    elif pos_str == "D-MSN":
        return "D-N"
    return pos_str


# namedtuple("POS", )
@dataclass
class POS:
    primary: str
    secondary: str = ""
    tertiary: str = ""
    flags: str = ""
    number: str = ""
    case: str = ""
    sequence: str = ""
    coindex: str = ""
    spelling: str = ""
    rest: str = ""
    shared_modifier: str = ""  # NX, ADVX, RX, etc


# def has_plural(primary, tail):
#     if primary == "NS" or primary == "NRPS":
#         return "S"
#     return ""

# (OTHERS+NUM-N aðrirtveggju-aðrirtveggju)
# (NP (ONES+Q-G einhverra-einhver) (NS-G hluta-hlutur))


def is_unitary_sequence(head):
    if not head or head.isnumeric() or len(head) <= 2:
        return False
    return head[-2:].isnumeric()


def parse_pos(pos_str):
    heads, *tail = pos_str.split("-")
    heads = heads.split("+")
    prim = heads[0]
    sec, tert = None, None
    if len(heads) > 1:
        sec = heads[1]
    if len(heads) > 2:
        tert = heads[2]
    if len(heads) > 3:
        breakpoint()
    if not any(c.isalpha() for c in pos_str):
        return POS(primary=pos_str)

    tailcopy = list(tail)
    for part in list(tail):
        if part.isnumeric():
            tailcopy.remove(part)
    for item in [prim, sec, tert]:
        if not item:
            continue
        item = item[:-2] if is_unitary_sequence(item) else item
        tailcopy.append(item[:-2] if is_unitary_sequence(item) else item)

    if any(part not in LEGAL_POS_HEADS for part in tailcopy):
        ic(pos_str)
        breakpoint()

    spelling = ""
    if "TTT" in tail:
        spelling = "TTT"
        tail.remove("TTT")

    sequence = ""
    if len(prim) > 1 and prim[-2:].isnumeric():
        pass


def inspect_pos(pos_list):
    pos_dict = defaultdict(int)
    heads = defaultdict(int)
    primary_heads = defaultdict(int)
    secondary_heads = defaultdict(int)
    tertiary_heads = defaultdict(int)
    tails = defaultdict(int)
    for (pos_str, count) in pos_list:
        tag = maybe_fix_pos(pos_str)
        pos_dict[tag] += count
        if pos_str != "-":
            head, *tail = pos_str.split("-", 1)
        else:
            head = "-"
            tail = []
        heads[head] += count
        if "+" in head:
            primary_heads[head.split("+", 1)[0]] += count
            secondary_heads[head.split("+", 1)[1]] += count
            if head.count("+") > 1:
                tertiary_heads[head.split("+", 2)[2]] += count

        if tail:
            tail = tail[0]
            tails[tail] += count

        parse_pos(tag)

    ic(sorted(list(pos_dict.items()), key=lambda x: x[0]))
    ic(sorted([(t, c) for t, c in pos_dict.items() if "+" in t], key=lambda x: x[0]))
    # ic(heads, tails)
    # ic(set(heads), set(tails))
    # ic(primary_heads, secondary_headsmisericordie-misericordie)
    breakpoint()


def inspect_pos(pos_list):
    pos_dict = defaultdict(int)
    heads = defaultdict(int)
    parts = defaultdict(int)

    head_to_parts = dict()

    queue = []
    queue.extend(pos_list.items())
    while queue:
        item, count = queue.pop()
        head, *pieces = item.split("-") if len(item) > 2 else (item, [])
        pieces = [part for part in pieces if part and not part.isnumeric()]
        # if head in LEGAL_NT_HEADS:
        #     continue
        if head == "" or "" in pieces:
            breakpoint()
        heads[head] += count
        part_counts_by_head = head_to_parts.get(head, defaultdict(int))
        for part in pieces:
            parts[part] += count
            part_counts_by_head[part] += count
        head_to_parts[head] = part_counts_by_head

    clean_head_to_parts = {}
    for head, part_counts_by_head in head_to_parts.items():
        item = dict(part_counts_by_head) if part_counts_by_head else None
        clean_head_to_parts[head] = item

    merged_heads = dict()
    for head in [k for k in set(heads) if "+" in k]:
        merged_heads[head] = heads.pop(head)

    ic(clean_head_to_parts)
    ic(set(heads) & LEGAL_POS_HEADS)
    ic()
    ic("Unused pos heads", LEGAL_POS_HEADS.difference(set(heads)))
    ic("Illegal pos heads", set(heads).difference(LEGAL_POS_HEADS))
    ic(merged_heads)
    ic(parts)


def inspect_nt(nt_list):
    pos_dict = defaultdict(int)
    nt_heads = defaultdict(int)
    parts = defaultdict(int)

    head_to_parts = dict()

    queue = []
    queue.extend(nt_list.items())
    while queue:
        item, count = queue.pop()
        if ">" in item:
            for sub_item in item.split(">"):
                queue.append((sub_item, count))
            continue
        head, *pieces = item.split("-")
        pieces = [part for part in pieces if part]
        # if head in LEGAL_NT_HEADS:
        #     continue
        if head == "" or "" in pieces:
            breakpoint()
        nt_heads[head] += count
        part_counts_by_head = head_to_parts.get(head, defaultdict(int))
        for part in pieces:
            parts[part] += count
            part_counts_by_head[part] += count
        head_to_parts[head] = part_counts_by_head

    clean_head_to_parts = {}
    for head, part_counts_by_head in head_to_parts.items():
        if part_counts_by_head:
            clean_head_to_parts[head] = dict(part_counts_by_head)
        else:
            clean_head_to_parts[head] = None

    from pprint import pprint
    ic(set(nt_heads) & LEGAL_POS_HEADS)
    ic(set(nt_heads) & LEGAL_NT_HEADS)
    ic(set(nt_heads).difference(LEGAL_POS_HEADS))
    ic(set(nt_heads).difference(LEGAL_NT_HEADS).difference(LEGAL_POS_HEADS))
    ic(parts)
    ic(clean_head_to_parts)

inspect_pos(pos_list)
# (PP (P (P undir-undir) (CONJ og-og) (P af$-af)) ())NP (N-D $reisu-reisa))

# inspect_nt(nt_list)
