from typing import Optional, Tuple, Any, List, NamedTuple
from dataclasses import dataclass

from icecream import ic

from greynirseq.nicenlp.utils.constituency.greynir_utils import NonterminalNode, TerminalNode, Node


NULL_LABEL = "NULL"
ROOT_LABEL = "ROOT"


@dataclass
class ParseLabel:
    label: str
    span: Tuple[int, int] = None

    def is_null(self):
        return self.label == NULL_LABEL

    @property
    def label_flags(self):
        return NonterminalNode.get_label_flags(self.label)

    @property
    def label_without_flags(self):
        return NonterminalNode.get_label_without_flags(self.label)

    @property
    def label_head(self):
        return NonterminalNode.get_label_head(self.label)


class ParseAction:
    def __init__(
            self,
            parent: Optional[str],
            preterminal: Optional[str],
            depth: int,
            parent_span:Optional[Tuple[int, int]] = None,
            preterminal_span: Optional[Tuple[int, int]] = None,
            right_chain_indices: Optional[List[int]] = None,
            preorder_indices: Optional[List[int]] = None,
            nwords: int = None,
    ):
        self.depth = depth
        self.parent = ParseLabel(parent or NULL_LABEL, parent_span)
        self.preterminal = ParseLabel(preterminal or NULL_LABEL, preterminal_span)
        self.right_chain_indices = right_chain_indices
        self.preorder_indices = preorder_indices
        self.nwords = nwords

    def __eq__(self, other: Any):
        if not isinstance(other, ParseAction):
            return False
        return (self.parent.label == other.parent.label) and (self.preterminal.label == other.preterminal.label)

    def __repr__(self):
        return f"ParseAction(parent='{self.parent.label}', preterminal='{self.preterminal.label}', depth={self.depth}, parent_span={self.parent.span}, preterminal_span={self.preterminal.span}, nwords={self.nwords}, right_chain_indices={self.right_chain_indices}, preorder_indices={self.preorder_indices})"


def get_right_chain(root):
    right_chain = [root]
    cursor = root
    while cursor.children:
        cursor = cursor.children[-1]
        if cursor.nonterminal:
            right_chain.append(cursor)
    return right_chain


def mark_preterminal_and_parent(node, depth=1):
    # we assume a nonterminal either (has one or more nonterminals as children) or (has exactly one terminal)
    # (this means that there are no bare terminals)
    node._depth = depth
    node.parent = node.parent if hasattr(node, "parent") else None
    assert not node.terminal
    assert (
        all(child.nonterminal for child in node.children) or len(node.children) == 1
    ), "Expected no bare terminals, consider wrapping bare terminals"
    node.is_preterminal = len(node.children) == 1 and (node.children[0].terminal is not None)
    for child in node.children:
        child.parent = node
        if child.terminal:
            continue
        mark_preterminal_and_parent(child, depth=depth + 1)


def get_preorder_index_of_right_chain(root, include_terminals=False, preserve_indices=True):
    if not preserve_indices:
        _ = root.preorder_list()  # re-precompute preorder_index for each node
    right_chain = []
    if root.composite_preorder_indices:
        right_chain.extend(root.composite_preorder_indices)
    else:
        right_chain = [root.preorder_index]
    cursor = root
    while cursor.children:
        cursor = cursor.children[-1]
        if cursor.nonterminal and not cursor.composite_preorder_indices:
            right_chain.append(cursor.preorder_index)
        elif cursor.nonterminal:
            right_chain.extend(cursor.composite_preorder_indices)
        elif cursor.terminal and include_terminals:
            right_chain.append(cursor.preorder_index)
    return right_chain


def get_preorder_indices(node, include_terminals=False, preserve_indices=True):
        ret = []
        for desc in node.preorder_list(include_terminals=include_terminals, preserve_indices=preserve_indices):
            if desc.composite_preorder_indices:
                ret.extend(desc.composite_preorder_indices)
            else:
                ret.append(desc.preorder_index)
            if None in ret:
                breakpoint()
        return ret


def get_incremental_parse_actions(node, collapse=True, verbose=False, preorder_index_to_original=True, bp=False):
    ic_enabled = ic.enabled
    if not verbose:
        ic.disable()
    if not node.nonterminal:
        raise ValueError("Expected nonterminal node")

    preorder_list = None
    root = node.clone()
    if collapse:
        root = root.collapse_unary()
        preorder_list = ic(root.preorder_list())
    if not collapse:
        # compute preorder_index before root was collapsed
        root = root.uncollapse_unary()
        preorder_list = ic(root.preorder_list())
        root = root.collapse_unary()

    if verbose:
        root.pretty_print()
    cursor = root
    actions = []

    while root.children:
        if verbose:
            print()
            root.pretty_print()
        _ = root.span
        # breakpoint()
        mark_preterminal_and_parent(root)
        while not collapse and ">" in root.nonterminal:
            ic("root is composite, decomposing")
            top_nt, *rest = root.nonterminal.split(">", 1)
            rest = rest[0]
            actions.append(ParseAction(top_nt, NULL_LABEL, 1, parent_span=root.span, nwords=len(root.leaves)))
            root._nonterminal = rest
            if root.composite_preorder_indices:
                root._composite_preorder_indices.pop(0)
                root._preorder_index = root.composite_preorder_indices[0]
            actions[-1].right_chain_indices = get_preorder_index_of_right_chain(root, include_terminals=False, preserve_indices=preorder_index_to_original)
            actions[-1].preorder_indices = get_preorder_indices(root, include_terminals=False, preserve_indices=preorder_index_to_original)
            ic(actions[-1])

        while not collapse and ">" in cursor.nonterminal:
            ic("cursor is composite, decomposing")
            top_nt, *rest = cursor.nonterminal.split(">", 1)
            rest = rest[0]
            actions.append(ParseAction(top_nt, NULL_LABEL, cursor.depth, parent_span=cursor.span, nwords=len(root.leaves)))
            cursor._nonterminal = rest
            actions[-1].right_chain_indices = get_preorder_index_of_right_chain(root, include_terminals=False, preserve_indices=preorder_index_to_original)
            actions[-1].preorder_indices = get_preorder_indices(root, include_terminals=False, preserve_indices=preorder_index_to_original)
            ic(actions[-1])

        if len(root.children) == 1:
            ic("root has 1 child, finished")
            actions.append(ic(ParseAction(NULL_LABEL, root.nonterminal, 0, preterminal_span=root.span, nwords=len(root.leaves))))
            actions[-1].right_chain_indices = []  # We popped the root, there is no right-chain remaining
            actions[-1].preorder_indices = []
            return ic(actions[::-1]), preorder_list

        # if bp:
        #     ic(root.nonterminal, top_nt, rest, root.preorder_index, root.composite_preorder_indices)
        #     breakpoint()

        cursor_to_child_idx = len(cursor.children) - 1
        child = cursor.children[cursor_to_child_idx]

        if not child.is_preterminal:
            ic("traversing right chain")
            cursor = child
            continue

        if verbose:
            root.pretty_print()
        # found it
        if cursor_to_child_idx > 1:
            ic("cursor_to_child_idx > 1")
            # cursor has 3 or more children, we dont need to contract nodes for now
            # action: append child to cursor
            actions.append(ParseAction(NULL_LABEL, child.nonterminal, cursor.depth, preterminal_span=child.span, nwords=len(root.leaves)))
            child = cursor._children.pop(cursor_to_child_idx)
            actions[-1].right_chain_indices = get_preorder_index_of_right_chain(root, include_terminals=False, preserve_indices=preorder_index_to_original)
            actions[-1].preorder_indices = get_preorder_indices(root, include_terminals=False, preserve_indices=preorder_index_to_original)
            ic(actions[-1])
        elif cursor_to_child_idx == 1:
            # cursor has exactly 2 children
            ic("cursor_to_child_idx == 1")

            # decompose node uncontraction
            while not collapse and ">" in cursor.nonterminal:
                ic("cursor is composite, decomposing cursor")
                ic((cursor, child, child.depth))
                top_nt, rest = cursor.nonterminal.split(">", 1)
                # action: extend leg above cursor (inverse of tree contraction) with top_nt
                actions.append(ParseAction(top_nt, NULL_LABEL, cursor.depth, parent_span=cursor.span, nwords=len(root.leaves)))
                cursor._nonterminal = rest
                actions[-1].right_chain_indices = get_preorder_index_of_right_chain(root, include_terminals=False, preserve_indices=preorder_index_to_original)
                actions[-1].preorder_indices = get_preorder_indices(root, include_terminals=False, preserve_indices=preorder_index_to_original)
                ic(actions[-1])
            while not collapse and ">" in child.nonterminal:
                ic("child is composite, decomposing child")
                ic((cursor, child, child.depth))
                top_nt, rest = child.nonterminal.split(">", 1)
                # action: extend leg above child (inverse of tree contraction) with top_nt
                actions.append(ParseAction(top_nt, NULL_LABEL, child.depth, parent_span=child.span, nwords=len(root.leaves)))
                child._nonterminal = rest
                actions[-1].right_chain_indices = get_preorder_index_of_right_chain(root, include_terminals=False, preserve_indices=preorder_index_to_original)
                actions[-1].preorder_indices = get_preorder_indices(root, include_terminals=False, preserve_indices=preorder_index_to_original)
                ic(actions[-1])

            # action: swap cursor out for a node which has children=[cursor, child]
            actions.append(ParseAction(cursor.nonterminal, child.nonterminal, cursor.depth, parent_span=cursor.span, preterminal_span=child.span, nwords=len(root.leaves)))  # XXX: Do we need two spans here?
            child = cursor._children.pop(cursor_to_child_idx)
            if cursor is root:
                ic("cursor is root, contracting root")
                # have to remove top-level node
                root = root.children[0]
                root.parent = None
            else:
                while len(cursor.children) == 1:
                    # cursor must be contracted
                    ic("contracting cursor")
                    parent = cursor.parent
                    parent.children[parent.children.index(cursor)] = cursor.children[0]
                    cursor = parent
                    if parent is None:
                        break
            actions[-1].right_chain_indices = get_preorder_index_of_right_chain(root, include_terminals=False, preserve_indices=preorder_index_to_original)
            actions[-1].preorder_indices = get_preorder_indices(root, include_terminals=False, preserve_indices=preorder_index_to_original)
            ic(actions[-1])
        else:
            # this should never happen
            # it would mean we didnt contract a node when we should have earlier
            # (and there are no unary chains since we collapsed them)
            assert False
        cursor = root

    if verbose and not ic_enabled:
        ic.disable()


def parse_by_actions(actions, tokens, verbose=False):
    ic_enabled = ic.enabled
    if not verbose:
        ic.disable()
    ic(actions, tokens)
    actions_ = list(actions)
    tokens_ = list(tokens)
    if not actions or not tokens:
        return None
    elif not actions[0].parent == NULL_LABEL or actions[0].depth != 0:
        raise ValueError("Illegal first parse action")
    root = NonterminalNode(ROOT_LABEL)

    while actions_:
        action = actions_.pop(0)
        if verbose:
            root.pretty_print()
        ic(action)
        # breakpoint()
        right_chain = get_right_chain(root)
        assert action.depth < len(right_chain)

        if action.preterminal.is_null():
            ic("uncontracting node")
            if right_chain[action.depth] is root:
                new_parent = NonterminalNode(action.parent.label, [right_chain[action.depth]])
                right_chain[action.depth - 1].children[-1] = new_parent
                ic(action.parent.span)
                continue
            old = right_chain[action.depth]._nonterminal
            right_chain[action.depth]._nonterminal = f"{action.parent.label}>{old}"
            continue

        token = tokens_.pop(0)
        leaf = TerminalNode(token, "x", "x")
        preterminal = NonterminalNode(action.preterminal.label, [leaf])

        if action.parent.is_null():
            # append new preterminal only
            right_chain[action.depth].children.append(preterminal)
        else:
            # extend a leg in the tree graph between action.depth and its parent
            # by inserting a node inbetween that has our new preterminal
            assert 0 < action.depth
            # new_parent = NonterminalNode(action.parent, [old_parent, preterminal])
            old_parent = right_chain[action.depth - 1].children[-1]
            new_parent = NonterminalNode(action.parent.label, [old_parent, preterminal])
            right_chain[action.depth - 1].children[-1] = new_parent

    if verbose and not ic_enabled:
        ic.disable()

    return root.children[0]


def test_incremental_parser():
    mjog = NonterminalNode("ADVP", [TerminalNode("mjög", "x")])
    flotti = NonterminalNode("ADJP", [TerminalNode("flotti", "x")])
    flotti = NonterminalNode("ADJP", [mjog, flotti])
    billinn = NonterminalNode("NP", [TerminalNode("bíllinn", "x")])
    billinn = NonterminalNode("DP", [billinn])
    hafdi = NonterminalNode("VP-AUX", [TerminalNode("hafði", "x")])
    ekki = NonterminalNode("ADVP", [TerminalNode("ekki", "x")])
    runnid = NonterminalNode("VP", [TerminalNode("runnið", "x")])
    hratt = NonterminalNode("ADVP", [TerminalNode("hratt", "x")])
    nidur = NonterminalNode("P", [TerminalNode("niður", "x")])
    brottu = NonterminalNode("ADVP", [TerminalNode("bröttu", "x")])
    gotuna = NonterminalNode("NP", [TerminalNode("götuna", "x")])
    gotuna = NonterminalNode("DP", [gotuna])

    flotti_billinn = NonterminalNode("NP", [flotti, billinn])
    subj = NonterminalNode("NP-SUBJ", [flotti_billinn])
    hafdi_ekki = NonterminalNode("VP", [hafdi, ekki])
    runnid_hratt = NonterminalNode("VP", [runnid, hratt])
    hafdi_ekki_runnid = NonterminalNode("VP", [hafdi, ekki, runnid_hratt])
    hafdi_ekki_runnid = NonterminalNode("VP", [hafdi_ekki_runnid])

    brottu_gotuna = NonterminalNode("NP", [brottu, gotuna])
    nidur_brottu_gotuna = NonterminalNode("PP", [nidur, brottu_gotuna])
    nidur_brottu_gotuna = NonterminalNode("PP-DIR", [nidur_brottu_gotuna])
    nidur_brottu_gotuna = NonterminalNode("PP-ARG", [nidur_brottu_gotuna])

    hafdi_ekki_runnid_nidur_brottu_gotuna = NonterminalNode("VP", [hafdi_ekki_runnid, nidur_brottu_gotuna])
    ip = NonterminalNode("IP", [subj, hafdi_ekki_runnid_nidur_brottu_gotuna])
    sentence = NonterminalNode("S0", [ip]).collapse_unary()

    #                                     S0>IP
    #                _______________________|________________
    #               |                                        VP
    #               |                        ________________|________________
    #           NP-SUBJ>NP                VP>VP                        PP-ARG>PP-DIR>PP
    #        _______|_________        ______|___________            __________|__________
    #      ADJP               |      |      |           VP         |                     NP
    #   ____|_______          |      |      |      _____|____      |           __________|____
    # ADVP         ADJP     DP>NP  VP-AUX  ADVP   VP        ADVP   P         ADVP           DP>NP
    #  |            |         |      |      |     |          |     |          |               |
    #  x            x         x      x      x     x          x     x          x               x
    #  |            |         |      |      |     |          |     |          |               |
    # mjög        flotti   bíllinn hafði   ekki runnið     hratt niður      bröttu          götuna

    tokens = [t.text for t in sentence.leaves]

    collapsed_actions = get_incremental_parse_actions(sentence.clone())
    correct_collapsed_actions = [
        ParseAction(parent=NULL_LABEL, preterminal='ADVP', depth=0, parent_span=None, preterminal_span=(0, 1)),
        ParseAction(parent='ADJP', preterminal='ADJP', depth=1, parent_span=(0, 2), preterminal_span=(1, 2)),
        ParseAction(parent='NP-SUBJ>NP', preterminal='DP>NP', depth=1, parent_span=(0, 3), preterminal_span=(2, 3)),
        ParseAction(parent='S0>IP', preterminal='VP-AUX', depth=1, parent_span=(0, 10), preterminal_span=(3, 4)),
        ParseAction(parent='VP>VP', preterminal='ADVP', depth=2, parent_span=(3, 7), preterminal_span=(4, 5)),
        ParseAction(parent=NULL_LABEL, preterminal='VP', depth=2, parent_span=None, preterminal_span=(5, 6)),
        ParseAction(parent='VP', preterminal='ADVP', depth=3, parent_span=(5, 7), preterminal_span=(6, 7)),
        ParseAction(parent='VP', preterminal='P', depth=2, parent_span=(3, 10), preterminal_span=(7, 8)),
        ParseAction(parent='PP-ARG>PP-DIR>PP', preterminal='ADVP', depth=3, parent_span=(7, 10), preterminal_span=(8, 9)),
        ParseAction(parent='NP', preterminal='DP>NP', depth=4, parent_span=(8, 10), preterminal_span=(9, 10))
    ]
    assert all(a == g for (a, g) in zip(correct_collapsed_actions, collapsed_actions))

    reparsed = parse_by_actions(collapsed_actions, tokens)
    assert all(
        (orig.span == repar.span and orig.label == repar.label)
        for (orig, repar) in zip(sentence.labelled_spans()[0], reparsed.labelled_spans()[0])
    )

    uncollapsed_actions = get_incremental_parse_actions(sentence.clone(), collapse=False)
    correct_uncollapsed_actions = [
        ParseAction(parent=NULL_LABEL, preterminal='ADVP', depth=0, parent_span=None, preterminal_span=(0, 1)),
        ParseAction(parent='ADJP', preterminal='ADJP', depth=1, parent_span=(0, 2), preterminal_span=(1, 2)),
        ParseAction(parent='NP', preterminal='NP', depth=1, parent_span=(0, 3), preterminal_span=(2, 3)),
        ParseAction(parent='DP', preterminal=NULL_LABEL, depth=2, parent_span=(2, 3), preterminal_span=None),
        ParseAction(parent='NP-SUBJ', preterminal=NULL_LABEL, depth=1, parent_span=(0, 3), preterminal_span=None),
        ParseAction(parent='IP', preterminal='VP-AUX', depth=1, parent_span=(0, 10), preterminal_span=(3, 4)),
        ParseAction(parent='VP', preterminal='ADVP', depth=2, parent_span=(3, 7), preterminal_span=(4, 5)),
        ParseAction(parent=NULL_LABEL, preterminal='VP', depth=2, parent_span=None, preterminal_span=(5, 6)),
        ParseAction(parent='VP', preterminal='ADVP', depth=3, parent_span=(5, 7), preterminal_span=(6, 7)),
        ParseAction(parent='VP', preterminal=NULL_LABEL, depth=2, parent_span=(3, 7), preterminal_span=None),
        ParseAction(parent='VP', preterminal='P', depth=2, parent_span=(3, 10), preterminal_span=(7, 8)),
        ParseAction(parent='PP', preterminal='ADVP', depth=3, parent_span=(7, 10), preterminal_span=(8, 9)),
        ParseAction(parent='NP', preterminal='NP', depth=4, parent_span=(8, 10), preterminal_span=(9, 10)),
        ParseAction(parent='DP', preterminal=NULL_LABEL, depth=5, parent_span=(9, 10), preterminal_span=None),
        ParseAction(parent='PP-DIR', preterminal=NULL_LABEL, depth=3, parent_span=(7, 10), preterminal_span=None),
        ParseAction(parent='PP-ARG', preterminal=NULL_LABEL, depth=3, parent_span=(7, 10), preterminal_span=None),
        ParseAction(parent='S0', preterminal=NULL_LABEL, depth=1, parent_span=(0, 10), preterminal_span=None)
    ]
    assert all(a == g for (a, g) in zip(correct_uncollapsed_actions, uncollapsed_actions))

    uncollapsed_reparsed = parse_by_actions(uncollapsed_actions, tokens)
    assert all(
        (orig.span == repar.span and orig.label == repar.label)
        for (orig, repar) in zip(sentence.labelled_spans()[0], uncollapsed_reparsed.labelled_spans()[0])
    )
