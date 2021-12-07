# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.


import sys
from collections import OrderedDict, namedtuple

import nltk
import numpy as np
from nltk.tree import Tree as NltkTree
from reynir import simpletree

from greynirseq.nicenlp.utils.constituency import unary_branch_labels


def _simplify_nonterminal(nonterminal):
    if ">" not in nonterminal:
        return nonterminal.split("-")[0]
    labels = [subnonterminal.split("-")[0] for subnonterminal in nonterminal.split(">")]
    return ">".join(labels)


class Node:
    def __init__(self):
        self._span = None
        pass

    @property
    def nonterminal(self):
        pass

    @property
    def children(self):
        pass

    @property
    def text(self):
        if self.terminal:
            return self.text
        return " ".join([node.text for node in self.children])

    @property
    def span(self):
        if self._span is None:
            self._span = self._compute_span()
        return self._span

    def _compute_span(self, offset=0):
        if self.terminal:
            span = offset, offset + 1
            self._span = span
            return span
        left_most, end = offset, offset + 1
        for child in self.children:
            start, end = child._compute_span(offset)
            offset = end
        self._span = (left_most, end)
        return self._span

    @property
    def terminal(self):
        pass

    @property
    def category(self):
        pass

    @property
    def label(self):
        pass

    @property
    def simple_label(self):
        pass

    @property
    def tag(self):
        pass

    @property
    def simple_tag(self):
        pass

    @classmethod
    def from_psd_file_obj(cls, file_obj, ignore_errors=False, limit=-1):
        num_trees_seen = 0
        for tree_str in file_obj.read().split("\n\n"):
            if not tree_str.strip():
                continue
            try:
                reynir_tree = simpletree.AnnoTree(tree_str).as_simple_tree()
                yield cls.from_simple_tree(reynir_tree), tree_str
            except Exception as exc:
                print(f"Could not parse tree: \n{tree_str}")
                if not ignore_errors:
                    raise exc
                yield None, tree_str
            num_trees_seen += 1
            if limit > 0 and num_trees_seen >= limit:
                return

    @classmethod
    def from_simple_tree(cls, obj):
        if isinstance(obj, dict):
            obj = simpletree.SimpleTree([[obj]])
        return cls._from_simple_tree_inner(obj)

    @classmethod
    def _from_simple_tree_inner(cls, obj):
        if obj.is_terminal:
            term = obj.terminal_with_all_variants
            if obj.kind == "PUNCTUATION":
                node = TerminalNode(obj.text, "grm")
                return node

            try:
                node = TerminalNode(obj.text, term)
            except Exception as e:
                sys.stderr.write(f"Could not parse flat terminal: {term}\n")
                raise e
            return node
        node = NonterminalNode(obj.tag)
        for child in obj.children:
            node.children.append(cls._from_simple_tree_inner(child))
        return node

    def binarize(self):
        if self.terminal:
            return self
        new_root = NonterminalNode(self.nonterminal)
        if len(self.children) <= 2:
            new_root._children = [child.binarize() for child in self.children]
            return new_root

        split_idx = np.random.randint(1, len(self.children))  # end is excluded
        left, right = self.children[:split_idx], self.children[split_idx:]

        new_children = []
        for side in [left, right]:
            if len(side) == 1:
                new_children.append(side[0].binarize())
            else:
                new_node = NonterminalNode(NULL_CAT_NONTERM, side).binarize()
                new_children.append(new_node)

        new_root._children = new_children
        return new_root

    def debinarize(self):
        if self.terminal or self.category:
            return self

        children = [child.debinarize() for child in self.children]
        if self.simple_label == NULL_CAT_NONTERM:
            return children

        new_children = []
        stack = list(reversed(children))
        while stack:
            c = stack.pop()
            if isinstance(c, list):
                stack.extend(reversed(c))
            else:
                new_children.append(c)
        new_root = NonterminalNode(self.nonterminal, new_children)
        return new_root

    @classmethod
    def convert_to_nltk_tree(cls, node, simplify_leaves=False, html_escape=False):
        if node.terminal:
            text = node.text.replace("(", r"\(").replace(")", r"\)")
            if html_escape:
                text = node.text.replace(r"\(", "&#40;").replace(r"\)", "&#41;")
            nltk_children = [text]
            if node.lemma is not None:
                lemma = node.lemma.replace("(", r"\(").replace(")", r"\)")
                if html_escape:
                    lemma = lemma.replace(r"\(", "&#40;").replace(r"\)", "&#41;")
                lemma_node = nltk.Tree("lemma", [lemma])
                nltk_children.append(lemma_node)
            tag = node.tag
            if simplify_leaves:
                tag = node.category
            return NltkTree(tag, nltk_children)
        return NltkTree(
            node.tag,
            [
                cls.convert_to_nltk_tree(child, simplify_leaves=simplify_leaves, html_escape=html_escape)
                for child in node.children
            ],
        )

    @classmethod
    def from_nltk_tree(cls, node):
        if not isinstance(node, NltkTree):
            assert isinstance(node, str)
            assert False
        label = node.label()
        if label.islower():
            new_node = TerminalNode(node[0], label)
            assert len(node) == 1
            return new_node
        new_node = NonterminalNode(label, children=[cls.from_nltk_tree(child) for child in node])
        return new_node

    def as_nltk_tree(self, simplify_leaves=False, html_escape=False):
        return self.convert_to_nltk_tree(self, simplify_leaves=simplify_leaves, html_escape=html_escape)

    def pretty_print(self, stream=None, simplify_leaves=False, html_escape=False):
        tree = self.as_nltk_tree(simplify_leaves=simplify_leaves, html_escape=html_escape)
        tree.pretty_print(stream=stream)
        del tree

    def labelled_spans(self, include_null=True, simplify=False):
        return self.tree_to_spans(self, include_null=include_null, simplify=simplify)

    @classmethod
    def _split_multiword_tokens(cls, node):
        if node.terminal:
            return node.split_multiword_tokens()
        new_children = []
        for child in node.children:
            if child.terminal:
                new_children.extend(child.split_multiword_tokens())
                continue
            new_children.append(cls._split_multiword_tokens(child))
        node._children = new_children
        return node

    def merge_multiword_tokens(self):
        if self.terminal:
            return self

        def merge_mw_token(term_nodes):
            new_text = []
            terminal = term_nodes[0].terminal
            terminal = terminal if terminal != "mw" else "x"
            for node in term_nodes:
                new_text.append(node.text)
            new_text = " ".join(new_text)
            new_node = TerminalNode(new_text, terminal)
            return new_node

        if not any(child.terminal == "mw" for child in self.children):
            return NonterminalNode(self.nonterminal, [child.merge_multiword_tokens() for child in self.children])

        new_children = []
        end = None
        for child_idx in range(len(self.children) - 1, -1, -1):
            # traverse in reverse order
            child = self.children[child_idx]
            if child.nonterminal and end is None:
                new_children.append(child.merge_multiword_tokens())
            elif child.nonterminal and end is not None:
                # just in case parser outputs illegal multiword token
                new_node = merge_mw_token(self.children[child_idx + 1 : end + 1])
                new_children.append(new_node)
                new_children.append(child.merge_multiword_tokens())
                end = None

            if child.terminal and child.terminal != "mw" and end is None:
                # regular terminal node
                new_children.append(child)
            elif child.terminal and child.terminal != "mw" and end is not None:
                # multiword token starts here
                new_node = merge_mw_token(self.children[child_idx : end + 1])
                new_children.append(new_node)
                end = None
            elif child.terminal == "mw" and end is None:
                # multiword token ends here
                end = child_idx

        if end is not None:
            # just in case parser outputs illegal multiword token
            start = 0
            new_node = merge_mw_token(self.children[start : end + 1])
            new_children.append(new_node)

        return NonterminalNode(self.nonterminal, list(reversed(new_children)))

    @property
    def leaves(self):
        if self.terminal is not None:
            return [self]
        ret = []
        for child in self.children:
            ret.extend(child.leaves)
        return ret

    @classmethod
    def _add_lemmas(cls, node, lemmas, allow_partial=False):
        # the reason we need this is because multiword tokens need to be merged
        # before lemmas can be added (i.e. in from_labelled_spans_and_terminals)
        # if node.nonterminal and "NP-POSS" in node.nonterminal:
        #     import pdb; pdb.set_trace()
        start, end = node.span  # cache spans
        assert (start < len(lemmas)) or allow_partial
        if node.terminal and start < len(lemmas):
            start, _end = node.span
            lemma = lemmas[start]
            if node.text.count(" ") != lemmas[start].count(" ") and allow_partial:
                lemma = node.text
            new_node = TerminalNode(node.text, node.terminal, lemma=lemma)
            return new_node
        elif node.terminal and start >= len(lemmas):
            new_node = TerminalNode(node.text, node.terminal, lemma=node.text)
            return new_node

        new_children = []
        for child in node.children:
            new_child = cls._add_lemmas(child, lemmas, allow_partial=allow_partial)
            new_children.append(new_child)
        return NonterminalNode(node.nonterminal, new_children)

    def add_lemmas(self, lemmas, allow_partial=False):
        return self._add_lemmas(self, lemmas, allow_partial=allow_partial)

    @classmethod
    def tree_to_spans(cls, tree, include_null=True, simplify=True):
        _ = tree.span  # compute and cache spans
        nterms, terms = [], []

        def traverse(node, depth=0, is_only_child=False):
            if node.terminal:
                node.add_empty_variants()
                assert " " not in node.text, "Split multiword tokens before export"
                if include_null and not is_only_child:
                    nterm_1wide = LabelledSpan(node.span, NULL_LEAF_NONTERM, depth, node, None)
                    nterms.append(nterm_1wide)
                    depth = depth + 1
                lspan = LabelledSpan(node.span, node.category, depth, node, node.flags)
                terms.append(lspan)
                return
            # nonterminal
            tag = node.simple_label if simplify else node.tag
            lspan = LabelledSpan(node.span, tag, depth, node, None)
            nterms.append(lspan)

            has_only_child = len(node.children) == 1
            for child in node.children:
                traverse(child, depth=depth + 1, is_only_child=has_only_child)

        traverse(tree)

        nterms = sorted(nterms, key=lambda x: (x.span[0], -x.span[1]))
        return nterms, terms

    def collapse_unary(self):
        if self.terminal:
            return self
        if len(self.children) > 1:
            new_node = NonterminalNode(self.nonterminal, [child.collapse_unary() for child in self.children])
            return new_node
        elif self.children[0].terminal:
            return NonterminalNode(self.nonterminal, list(self.children))

        merge_list = [self]
        cursor = self
        while True:
            cursor = cursor.children[0]
            if cursor.terminal:
                break
            merge_list.append(cursor)
            if len(cursor.children) > 1:
                break

        labels = []
        for elem in merge_list:
            labels.append(elem.tag)
        label = ">".join(labels)

        new_node = NonterminalNode(label, [child.collapse_unary() for child in merge_list[-1].children])
        return new_node

    def separate_unary(self):
        if self.terminal:
            return self
        if ">" not in self.nonterminal:
            new_children = [child.separate_unary() for child in self.children]
            return NonterminalNode(self.nonterminal, new_children)

        separated_labels = self.nonterminal.split(">")
        new_root = NonterminalNode(separated_labels[0])
        parent = new_root
        for label in separated_labels[1:]:
            new_node = NonterminalNode(label)
            parent.children.append(new_node)
            parent = new_node
        for child in self.children:
            new_node.children.append(child.separate_unary())
        return new_root

    def remove_null_leaves(self):
        cleaned = self._remove_null_leaves_inner()
        if len(cleaned) == 1:
            return cleaned[0]
        return NonterminalNode("S0-X", cleaned)

    def _remove_null_leaves_inner(self):
        if self.terminal:
            return [self]
        new_children = []
        for child in self.children:
            new_children.extend(child._remove_null_leaves_inner())
        if self.nonterminal == NULL_LEAF_NONTERM:
            return new_children
        return [NonterminalNode(self.nonterminal, new_children)]

    @classmethod
    def from_labelled_spans(cls, spans, labels, tokens=None):
        # spans need to be in pre-order
        last_ii, last_jj = None, None
        stack = []
        for idx, ((ii, jj), label) in enumerate(zip(spans, labels)):
            if ii + 1 == jj:
                text = None
                children = []
                if tokens:
                    text = tokens[ii]
                    children.append(TerminalNode(text, NULL_CAT_TERM))
                new_node = NonterminalNode(label, children)
            else:
                new_node = NonterminalNode(label)
            new_node._span = (ii, jj)

            if last_ii is None:
                stack.append(new_node)
            elif last_ii <= ii and jj <= last_jj:
                stack[-1].children.append(new_node)
                stack.append(new_node)
            elif last_jj <= ii:
                while last_jj <= ii:
                    parent = stack.pop()
                    last_ii, last_jj = parent.span
                stack.append(parent)
                parent.children.append(new_node)
                stack.append(new_node)
            else:
                import pdb

                pdb.set_trace()
                _ = 1 + 1

            last_ii, last_jj = ii, jj

        tree = stack.pop(0)
        del stack
        return tree

    @classmethod
    def from_labelled_spans_and_terminals(cls, spans, labels, tokens, terminals):
        # spans need to be in pre-order
        last_ii, last_jj = None, None
        stack = []
        for idx, ((ii, jj), label) in enumerate(zip(spans, labels)):
            if ii + 1 == jj:
                text = None
                children = []
                if tokens:
                    text = tokens[ii]
                    children.append(TerminalNode(text, terminals[ii]))
                new_node = NonterminalNode(label, children)
            else:
                new_node = NonterminalNode(label)
            new_node._span = (ii, jj)
            if last_ii is None:
                stack.append(new_node)
            elif last_ii <= ii and jj <= last_jj:
                stack[-1].children.append(new_node)
                stack.append(new_node)
            elif last_jj <= ii:
                while last_jj <= ii:
                    parent = stack.pop()
                    last_ii, last_jj = parent.span
                stack.append(parent)
                parent.children.append(new_node)
                stack.append(new_node)
            else:
                assert False, "Unreachable"
            last_ii, last_jj = ii, jj
        tree = stack.pop(0)
        del stack
        return tree

    def to_postfix(self, include_terminal=False):
        """Export tree to postfix ordering
        with node-labels as keys"""
        if self.terminal:
            return []
        result = []
        for child in self.children:
            result.extend(child.to_postfix())
        result.append(self.tag)
        return result

    def uniform(self, nonterminal="A", allow_null=True):
        if self.terminal:
            return self
        new_children = [child.uniform(nonterminal, allow_null) for child in self.children]
        new_label = nonterminal
        if allow_null and self.nonterminal == NULL_CAT_NONTERM:
            new_label = NULL_CAT_NONTERM
        elif self.nonterminal == NULL_LEAF_NONTERM:
            new_label = NULL_LEAF_NONTERM
        return NonterminalNode(new_label, new_children)

    def roof(self):
        if self.terminal:
            return None
        elif self.nonterminal == NULL_LEAF_NONTERM:
            return None

        new_children = []
        for child in self.children:
            new_child = child.roof()
            if new_child is None:
                continue
            new_children.append(new_child)
        return NonterminalNode(self.nonterminal, new_children)


class NonterminalNode(Node):
    def __init__(self, nonterminal_label, children=None):
        super(NonterminalNode, self).__init__()
        self._nonterminal = nonterminal_label
        self._children = children if children is not None else []

    @property
    def nonterminal(self):
        return self._nonterminal

    @property
    def simple_label(self):
        tag = self.simple_tag
        if tag == self.nonterminal or tag not in NONTERM_SIMPLIFIED:
            return self.nonterminal
        label = sorted(
            [label for label in NONTERM_SUFFIXES_SIMPLIFIED[tag] if self.nonterminal.startswith(label)],
            key=lambda x: len(x),
        )[-1]
        return label

    @property
    def tag(self):
        return self.nonterminal

    @property
    def simple_tag(self):
        return _simplify_nonterminal(self.nonterminal)

    @property
    def children(self):
        return self._children

    def __repr__(self):
        return "({0})".format(self.nonterminal)

    def split_multiword_tokens(self):
        return self._split_multiword_tokens(self)


class TerminalNode(Node):
    def __init__(self, text, terminal_label, category=None, lemma=None):
        super(TerminalNode, self).__init__()
        self._text = text
        self._terminal = terminal_label
        self._lemma = lemma

        if self._terminal:
            flags = []
            cat, variants = split_flat_terminal(self._terminal)
            assert cat in TERM_CATS or cat == NULL_CAT_TERM, cat
            for var_name, val in variants.items():
                flags.append("{}-{}".format(var_name, val))
            self._flags = flags
            self._category = cat
        else:
            self._flags = []
            self._category = category

    def add_empty_variants(self):
        flags = []
        cat, variants = split_flat_terminal(self._terminal)

        empty = tuple()
        for var_name in VAR:
            if var_name not in CATEGORY_TO_VARIANT.get(cat, empty):
                continue
            if var_name not in variants:
                variants[var_name] = "empty"

        for var_name, val in variants.items():
            flags.append("{}-{}".format(var_name, val))
        self._flags = flags

    @property
    def simple_label(self):
        return self.tag

    @property
    def labels(self):
        labels = [self.category]
        if self.category == "grm":
            return labels
        flat = self.terminal
        _cat, variants = split_flat_terminal(flat)
        for variant_name, subvariant in variants.items():
            if variant_name == "cat":
                continue
            labels.append("{}-{}".format(variant_name, subvariant))
        return labels

    @property
    def text(self):
        return self._text

    @property
    def terminal(self):
        return self._terminal

    @property
    def tag(self):
        return self.terminal

    @property
    def flags(self):
        return self._flags

    @property
    def simple_tag(self):
        return self.category

    @property
    def category(self):
        return self._category

    @property
    def lemma(self):
        return self._lemma

    def __repr__(self):
        return "({0} {1})".format(self.terminal, self.text)

    def split_multiword_tokens(self):
        new_terminals = []
        for idx, token in enumerate(self.text.split(" ")):
            term = self.terminal if idx == 0 else CAT_INSIDE_MW_TOKEN
            cat = self.category if idx == 0 else CAT_INSIDE_MW_TOKEN
            new_node = TerminalNode(
                token,
                term,
                category=cat,
            )
            new_terminals.append(new_node)
        return new_terminals


NULL_CAT_NONTERM = "NULL"
NULL_LEAF_NONTERM = "LEAF"
NULL_CAT_TERM = "null"
CAT_INSIDE_MW_TOKEN = "mw"


class VARIANT:
    ARTICLE = {"gr"}
    CASE = {"nf", "þf", "þgf", "ef"}
    LO_OBJ = {"sþf", "sþgf", "sef"}
    FS_OBJ = {"nf", "þf", "þgf", "ef", "nh"}
    GENDER = {"kk", "kvk", "hk"}
    NUMBER = {"et", "ft"}
    PERSON = {"p1", "p2", "p3"}
    TENSE = {"þt", "nt"}
    DEGREE = {"fst", "mst", "est", "esb", "evb"}
    STRENGTH = {"sb", "vb"}
    VOICE = {"mm", "gm"}
    MOOD = {"fh", "vh", "nh", "bh", "lhnt", "lhþt", "sagnb"}
    CLITIC = {"sn"}
    IMPERSONAL = {"none", "es", "subj"}


TERM_CATS = [
    "abfn",
    "abbrev",
    "amount",
    "ao",
    "ártal",
    # "dags",
    "dagsföst",
    "dagsafs",
    "entity",
    "eo",
    "fn",
    "foreign",
    "fs",
    "fyrirtæki",
    "gata",
    "gr",
    "grm",
    "kennitala",
    "lén",
    "lo",
    "mælieining",
    "myllumerki",
    CAT_INSIDE_MW_TOKEN,
    "nhm",
    "no",
    "notandanafn",
    "person",
    # "p",
    "pfn",
    "prósenta",
    "raðnr",
    "sameind",
    "sérnafn",
    "sequence",
    "símanúmer",
    "so",
    "st",
    "stt",
    "tala",
    "talameðbókstaf",
    "tao",
    "tímapunktur",
    "tímapunkturafs",
    "tímapunkturfast",
    "tími",
    "to",
    "töl",
    "tölvupóstfang",
    "uh",
    "vefslóð",
    "vörunúmer",
    "x",
]

NONTERM_CATS = [
    NULL_CAT_NONTERM,
    NULL_LEAF_NONTERM,
    "ADJP",
    "ADVP",
    "C",
    "CP",
    "FOREIGN",
    "IP",
    "NP",
    "P",
    "PP",
    "S0",
    "S",
    "TO",
    "URL",
    "VP",
]


NONTERM_SUFFIX = {
    # NULL_CAT_NONTERM: [NULL_CAT_NONTERM],
    # NULL_LEAF_NONTERM: [NULL_LEAF_NONTERM],
    # "ADJP": ["ADJP"],
    "ADVP": [
        "ADVP",
        "ADVP-DATE-ABS",
        "ADVP-DATE-REL",
        "ADVP-DIR",
        "ADVP-DUR-ABS",
        "ADVP-DUR-REL",
        "ADVP-DUR-TIME",
        "ADVP-LOC",
        "ADVP-PCL",
        "ADVP-TIMESTAMP-ABS",
        "ADVP-TIMESTAMP-REL",
        "ADVP-TMP-SET",
    ],
    "CP": [
        "CP-ADV-ACK",
        "CP-ADV-CAUSE",
        "CP-ADV-CMP",
        "CP-ADV-COND",
        "CP-ADV-CONS",
        "CP-ADV-PURP",
        "CP-ADV-TEMP",
        "CP-EXPLAIN",
        "CP-QUE",
        "CP-QUE-OBJ",
        "CP-QUE-PRD",
        "CP-QUE-SUBJ",
        "CP-QUOTE",
        "CP-REL",
        "CP-SOURCE",
        "CP-THT",
        "CP-THT-OBJ",
        "CP-THT-PRD",
        "CP-THT-SUBJ",
    ],
    "IP": [
        "IP",
        "IP-INF",
        "IP-INF-IOBJ",
        "IP-INF-OBJ",
        "IP-INF-PRD",
        "IP-INF-SUBJ",
    ],
    "NP": [
        "NP",
        "NP-ADDR",
        "NP-AGE",
        "NP-ADP",
        "NP-COMPANY",
        "NP-DAT",
        "NP-ES",
        "NP-EXCEPT",
        "NP-IOBJ",
        "NP-MEASURE",
        "NP-OBJ",
        "NP-PERSON",
        "NP-POSS",
        "NP-PRD",
        "NP-PREFIX",
        "NP-SOURCE",
        "NP-SUBJ",
        "NP-TITLE",
    ],
    "PP": ["PP", "PP-DIR", "PP-LOC"],
    "S0": ["S0", "S0-X"],
    "S": [
        # "S",
        "S-EXPLAIN",
        "S-HEADING",
        "S-MAIN",
        "S-PREFIX",
        "S-QUE",
        "S-QUOTE",
    ],
    "VP": ["VP", "VP-AUX"],
    # "C": ["C"],
    # "FOREIGN": ["FOREIGN"],
    # "P": ["P"],
    # "TO": ["TO"],
    # "URL": ["URL"],
}


VAR = OrderedDict(
    obj1=["nf", "þf", "þgf", "ef", "empty"],
    obj2=["nf", "þf", "þgf", "ef", "empty"],
    # supine: ["sagnb"],
    impersonal=["subj", "es", "none", "empty"],
    subj=["nf", "þf", "þgf", "ef", "empty"],
    person=["p3", "p2", "p1", "empty"],
    number=["et", "ft", "empty"],
    case=["nf", "þf", "þgf", "ef", "empty"],
    gender=["kk", "kvk", "hk", "empty"],
    article=["gr", "empty"],
    mood=["fh", "vh", "nh", "bh", "lhnt", "lhþt", "sagnb", "empty"],
    tense=["nt", "þt", "empty"],
    voice=["gm", "mm", "empty"],
    degree=["fst", "mst", "est", "esb", "evb", "empty"],
    strength=["sb", "vb", "empty"],
    clitic=["sn", "empty"],  # enclitic for second person
    lo_obj=["sþf", "sþgf", "sef", "empty"],
    fs_obj=["nf", "þf", "þgf", "ef", "nh", "empty"],
)


CATEGORY_TO_VARIANT = {
    "ao": ["degree"],
    "eo": ["degree"],
    "no": ["number", "case", "gender", "article"],
    "person": ["number", "case", "gender", "article"],
    "entity": ["number", "case", "gender", "article"],
    "sérnafn": ["number", "case", "gender", "article"],
    "abfn": ["number", "case", "gender"],
    "fn": ["number", "case", "gender"],
    "pfn": ["number", "case", "gender", "person"],
    "gr": ["number", "case", "gender"],
    "tala": ["number", "case", "gender"],
    "töl": ["number", "case", "gender"],
    "to": ["number", "case", "gender"],
    "lo": ["number", "case", "gender", "degree", "strength", "lo_obj"],
    "so": [
        "obj1",
        "obj2",
        "impersonal",
        "subj",
        "person",
        "number",
        "mood",
        "tense",
        "voice",
        "clitic",
        # hack
        "gender",
        "case",
        "strength",
    ],
    "fs": ["fs_obj"],
    "raðnr": ["case", "gender"],
    "lén": ["case"],
    "prósenta": ["number", "case", "gender"],
    "fyrirtæki": ["number", "case", "gender", "article"],
    "gata": ["number", "case", "gender", "article"],
}


def dedup_list(items):
    seen = set()
    ret_items = []
    for item in items:
        if item in seen:
            continue
        ret_items.append(item)
        seen.add(item)
    return ret_items


def make_nonterm_labels_decl():
    nonterm_labels = list(NONTERM_CATS)

    for prefix, nt_with_suffixes in NONTERM_SUFFIX.items():
        nonterm_labels.extend(nt_with_suffixes)

    for extra in [
        unary_branch_labels.UNARY_BRANCH_LABELS,
    ]:
        for composite_label, freq in extra.items():
            nonterm_labels.append(composite_label)

    nonterm_labels = dedup_list(nonterm_labels)
    return {
        "category_to_group_names": {},
        "group_name_to_labels": {},
        "group_names": [],
        "label_categories": nonterm_labels,
        "labels": nonterm_labels,
        "null": NULL_CAT_NONTERM,
        "null_leaf": NULL_LEAF_NONTERM,
        "separator": None,
    }


def make_simple_nonterm_labels_decl(include_null=True):
    group_name_to_labels = {}
    flags = []
    for prefix, full_labels in NONTERM_SUFFIX.items():
        for full_label in full_labels:
            if "-" not in nt_with_suffix:
                continue
            if "-" in full_label:
                # flag = "-".join(full_label.split("-")[1:])
                # flags.append(flag)
                flags.append("ATTR-{}".format(full_label))
    nonterm_labels = list(NONTERM_CATS) + list(NONTERM_SIMPLIFIED)

    for extra in [unary_branch_labels.UNARY_BRANCH_LABELS]:
        for composite_label, freq in extra.items():
            if composite_label not in nonterm_labels:
                nonterm_labels.append(composite_label)

    flags = dedup_list(flags)
    group_name_to_labels["ALL"] = flags

    nonterm_labels = dedup_list(nonterm_labels)
    return {
        "category_to_group_names": {},
        "group_name_to_labels": group_name_to_labels,
        "group_names": list(group_name_to_labels.keys()),
        "label_categories": nonterm_labels,
        "labels": nonterm_labels + flags,
        "null": NULL_CAT_NONTERM,
        "null_leaf": NULL_LEAF_NONTERM,
        "separator": None,
    }


def make_term_label_decl(sep="<sep>"):
    group_name_to_labels = {}
    label_cats = list(TERM_CATS)
    cat_to_group_names = dict(**CATEGORY_TO_VARIANT)

    term_labels = list(TERM_CATS)
    for gram_cat, subvariants in VAR.items():
        label_group = []
        for subvar in subvariants:
            label = "{}-{}".format(gram_cat, subvar)
            term_labels.append(label)
            label_group.append(label)
        group_name_to_labels[gram_cat] = label_group

    all_labels = [sep]
    all_labels.extend(label_cats)
    for label in term_labels:
        if label not in all_labels:
            all_labels.append(label)
    assert len(all_labels) == len(set(all_labels))

    return {
        "category_to_group_names": cat_to_group_names,
        "group_names": list(group_name_to_labels.keys()),
        "group_name_to_labels": group_name_to_labels,
        "label_categories": label_cats,
        "labels": all_labels,
        "null": None,
        "null_leaf": None,
        "separator": sep,
    }


LABEL_GROUP_NAME_TO_SUBLABELS = {}
LABEL_CATS = TERM_CATS + NONTERM_CATS
LABEL_CAT_TO_LABEL_GROUP_NAMES = dict(**CATEGORY_TO_VARIANT)
LABEL_CAT_TO_LABEL_GROUP_NAMES.update(dict((k, [k]) for k in NONTERM_SUFFIX.keys()))

TERM_LABELS = list(TERM_CATS)
for gram_cat, subvariants in VAR.items():
    label_group = []
    for subvar in subvariants:
        label = "{}-{}".format(gram_cat, subvar)
        TERM_LABELS.append(label)
        label_group.append(label)
    LABEL_GROUP_NAME_TO_SUBLABELS[gram_cat] = label_group

NONTERM_LABELS = list(NONTERM_CATS)
for prefix, nt_with_suffixes in NONTERM_SUFFIX.items():
    label_group = []
    for nt_with_suffix in nt_with_suffixes:
        label = "{}-{}".format(prefix, nt_with_suffix.replace("-", "_"))
        NONTERM_LABELS.append(label)
        label_group.append(label)
    LABEL_GROUP_NAME_TO_SUBLABELS[prefix] = label_group

LABEL_GROUP_NAMES = list(LABEL_GROUP_NAME_TO_SUBLABELS)
ALL_LABELS = list(LABEL_CATS)
for label in TERM_LABELS + NONTERM_LABELS:
    if label not in ALL_LABELS:
        ALL_LABELS.append(label)
assert len(ALL_LABELS) == len(set(ALL_LABELS))

LabelledSpan = namedtuple("LabelledSpan", ["span", "label", "depth", "node", "flags"])


def label_cat_to_label_group_mask(cat):
    mask = [0] * len(LABEL_GROUP_NAMES)
    if cat not in LABEL_CAT_TO_LABEL_GROUP_NAMES:
        return mask
    lbl_grps = LABEL_CAT_TO_LABEL_GROUP_NAMES[cat]
    for grp_name in lbl_grps:
        mask[LABEL_GROUP_NAMES.index(grp_name)] = 1
    return mask


def label_cat_to_label_mask(cat):
    assert cat in LABEL_CATS
    labels = [cat]
    label_group_names = LABEL_CAT_TO_LABEL_GROUP_NAMES.get(cat, [])
    for group_name in label_group_names:
        sublabels = LABEL_GROUP_NAME_TO_SUBLABELS[group_name]
        labels.extend(sublabels)
    mask = [0] * len(ALL_LABELS)
    for label in labels:
        mask[ALL_LABELS.index(label)] = 1
    return mask, labels


LABEL_CAT_TO_LABEL_MASK = {label_cat: label_cat_to_label_mask(label_cat) for label_cat in LABEL_CATS}
LABEL_CAT_TO_GROUP_MASK = {label_cat: label_cat_to_label_group_mask(label_cat) for label_cat in LABEL_CATS}
LABEL_IDX_TO_GROUP_MASK = {LABEL_CATS.index(label): mask for (label, mask) in LABEL_CAT_TO_GROUP_MASK.items()}

for extra in [
    unary_branch_labels.UNARY_BRANCH_LABELS,
]:
    for composite_label, freq in extra.items():
        if composite_label in ALL_LABELS:
            continue
        ALL_LABELS.append(composite_label)
        NONTERM_LABELS.append(composite_label)

NONTERM_SUFFIXES_SIMPLIFIED = {
    "ADVP": ["ADVP"],
    "ADJP": ["ADJP"],
    "NP": ["NP", "NP-SUBJ", "NP-OBJ", "NP-IOBJ", "NP-PRD", "NP-POSS"],
    "PP": ["PP"],
    "VP": ["VP", "VP-AUX"],
    "C": ["C"],
    "IP": ["IP"],
    "CP": ["CP", "CP-ADV", "CP-QUE", "CP-REL", "CP-THT"],
    "S": ["S"],
    "S0": ["S0"],
}
NONTERM_SIMPLIFIED = set([label for group in NONTERM_SUFFIXES_SIMPLIFIED.values() for label in group])


def rebinarize(spans, labels):
    tree = Node.from_labelled_spans(spans, labels)
    tree = tree.debinarize()
    new_tree = tree.binarize()
    nterms, _terms = new_tree.labelled_spans(include_null=True, simplify=False)
    new_spans, new_labels = zip(*[((it.span[0], it.span[1]), it.label) for it in nterms])
    return new_spans, new_labels


def make_flat_terminal_from_attributes(cat, attrs):
    parts = [cat]
    variants = dict([attr.split("-") for attr in attrs if (attr and "empty" not in attr)])
    nobjs = 0
    nobjs = 1 if variants.get("obj1") else 0
    nobjs = 2 if variants.get("obj2") else nobjs
    if cat == "so" or nobjs > 0:
        parts.append(str(nobjs))
    if variants.get("obj1"):
        parts.append(variants["obj1"])
        del variants["obj1"]
    if variants.get("obj2"):
        parts.append(variants["obj2"])
        del variants["obj2"]
    if variants.get("impersonal"):
        parts.append("op")
        parts.append(variants["impersonal"])
        del variants["impersonal"]
    for var_name in VAR:
        if var_name in variants:
            parts.append(variants[var_name])
    return "_".join(parts)


def split_flat_terminal(term):
    parts = term.split("_")
    if len(parts) <= 1:
        cat = term
        if term == "p":
            cat = "grm"
        return cat, {}

    variants = {}
    cat = parts.pop(0)

    # case_control = []
    if cat == "so" and "lhþt" not in term:
        first_var = parts[0]
        if first_var == "0":
            parts.pop(0)
        elif first_var in ("1", "2"):
            parts.pop(0)
            num_control = int(first_var)
            for idx in range(num_control):
                item = parts.pop(0)
                assert item in VARIANT.CASE
                var_name = "obj" + str(idx + 1)  # to 1-based
                variants[var_name] = item

        if "op" in parts and "es" in parts:
            # expletive
            variants["impersonal"] = "es"
            parts.remove("op")
            parts.remove("es")
        elif "none" in parts:
            # no subject
            variants["impersonal"] = "none"
            if "op" in parts:
                parts.remove("op")
            if "none" in parts:
                parts.remove("none")
        elif "op" in parts and "subj" in parts:
            # quirky/oblique subject
            idx = parts.index("subj")
            parts.pop(idx)
            if not any(part in VARIANT.CASE for part in parts):
                print(term, parts, variants)
            if parts[idx] in VARIANT.CASE:
                subj_case = parts.pop(idx)
                variants["subj"] = subj_case
                variants["impersonal"] = "subj"
            elif any(part in VARIANT.CASE for part in parts):
                subj_case = [part for part in parts if part in VARIANT.CASE][0]
                variants["subj"] = subj_case
                variants["impersonal"] = "subj"
                parts.remove(subj_case)
            else:
                # expected to find case
                # assert False, str(parts)
                pass
            parts.remove("op")
        elif "op" in parts:
            # no subject case control and no expletive means no subject
            variants["impersonal"] = "none"
            if "op" in parts:
                parts.remove("op")
    elif cat == "fs":
        attr = parts.pop(0)
        assert attr in VARIANT.FS_OBJ, "Unknown variant for fs: {}".format(attr)
        variants["fs_obj"] = attr
        pass

    done = ("subj", "fs_obj", "obj1", "obj2", "impersonal")
    remaining = set(parts)
    empty = tuple()
    for var_name in VAR.keys():
        if var_name in done:
            continue
        if var_name in CATEGORY_TO_VARIANT.get(cat, empty):
            all_subvariants = getattr(VARIANT, var_name.upper())
            item = all_subvariants & remaining
            if item:
                assert var_name not in variants, var_name
                variants[var_name] = item.pop()

    return cat, variants
