from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

from icecream import ic

from greynirseq.nicenlp.utils.constituency.greynir_utils import NonterminalNode, TerminalNode

NULL_LABEL = "NULL"
ROOT_LABEL = "ROOT"
EOS_LABEL = "EOS"


@dataclass
class ParseLabel:
    label: str
    span: Tuple[int, int] = None

    def is_null(self):
        return self.label == NULL_LABEL

    def is_eos(self):
        return self.label == EOS_LABEL

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
        parent_span: Optional[Tuple[int, int]] = None,
        preterminal_span: Optional[Tuple[int, int]] = None,
        right_chain_indices: Optional[List[int]] = None,
        preorder_indices: Optional[List[int]] = None,
        preorder_depths: Optional[List[int]] = None,
        nwords: int = None,
    ):
        self.depth = depth
        self.parent = ParseLabel(parent or NULL_LABEL, parent_span)
        self.preterminal = ParseLabel(preterminal or NULL_LABEL, preterminal_span)
        self.right_chain_indices = right_chain_indices
        self.preorder_indices = preorder_indices
        self.nwords = nwords
        self.preorder_depths = preorder_depths

    def __eq__(self, other: Any):
        if not isinstance(other, ParseAction):
            return False
        return (self.parent.label == other.parent.label) and (
            self.preterminal.label == other.preterminal.label
            and self.nwords == other.nwords
            and self.preorder_depths == other.preorder_depths
            and self.preorder_indices == other.preorder_indices
            and self.right_chain_indices == other.right_chain_indices
        )

    def __repr__(self):
        return (
            f"ParseAction(parent='{self.parent.label}', preterminal='{self.preterminal.label}',"
            f" depth={self.depth}, parent_span={self.parent.span}, preterminal_span={self.preterminal.span},"
            f" nwords={self.nwords}, right_chain_indices={self.right_chain_indices},"
            f" preorder_indices={self.preorder_indices}, preorder_depths={self.preorder_depths})"
        )


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


def get_preorder_index_of_right_chain(root, include_terminals=False, preserve_indices=True, collapse=False):
    if not preserve_indices:
        _ = root.preorder_list()  # re-precompute preorder_index for each node
    right_chain = []

    unary_len = root.nonterminal.count(">") + 1 if not collapse else 1
    right_chain.extend(range(root.preorder_index, root.preorder_index + unary_len))

    cursor = root
    while cursor.children:
        cursor = cursor.children[-1]
        if cursor.terminal and include_terminals:
            right_chain.append(cursor.preorder_index)
        elif cursor.nonterminal:
            unary_len = cursor.nonterminal.count(">") + 1 if not collapse else 1
            right_chain.extend(range(cursor.preorder_index, cursor.preorder_index + unary_len))
    return right_chain


def get_preorder_indices(node, include_terminals=False, preserve_indices=True, collapse=False):
    ret = []
    for desc in node.preorder_list(include_terminals=include_terminals, preserve_indices=preserve_indices):
        unary_len = desc.nonterminal.count(">") + 1 if not collapse else 1
        ret.extend(range(desc.preorder_index, desc.preorder_index + unary_len))
    return ret


def mark_depth(node, depth=0, overwrite=False, collapse=False):
    if not hasattr(node, "__depth"):
        node.__depth = None
    if node.__depth is None or overwrite:
        node.__depth = depth
    if node.terminal:
        return
    child_depth = depth + 1 if collapse else depth + node.nonterminal.count(">") + 1
    for child in node.children:
        mark_depth(child, depth=child_depth, overwrite=overwrite)


def __depth_as_iter(node):
    if node.terminal:
        raise ValueError
    unary_len = node.nonterminal.count(">") + 1
    return range(node.__depth, node.__depth + unary_len)


def get_depths(root):
    mark_depth(root, overwrite=True)
    return [
        d for node in root.preorder_list(include_terminals=False, preserve_indices=True) for d in __depth_as_iter(node)
    ]


def get_banned_attachments(node):
    assert node.nonterminal
    cursor = node.clone().collapse_unary()
    mark_depth(cursor, overwrite=True, collapse=False)
    ret = []
    banned_labels = {}
    while cursor.nonterminal:
        unary_len = cursor.nonterminal.count(">") + 1
        labels = cursor.nonterminal.split(">")
        for d in range(cursor.__depth, cursor.__depth + unary_len + 1):
            banned_labels[d] = labels
        if unary_len >= 3:
            ret.extend(range(cursor.__depth, cursor.__depth + unary_len + 1))
        # traverse right chain
        if cursor.children:
            cursor = cursor.children[-1]
        else:
            break
    return ret, banned_labels


def get_incremental_parse_actions(node, collapse=True, verbose=False, preorder_index_to_original=True, eos=None):
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

    if eos is not None:
        _ = root.span
        mark_preterminal_and_parent(root)
        # we add one to nwords to account for the eos token in src_tokens
        nwords_w_eos = len(root.leaves) + 1
        actions.append(ParseAction(eos, NULL_LABEL, 1, parent_span=root.span, nwords=nwords_w_eos))
        actions[-1].right_chain_indices = get_preorder_index_of_right_chain(
            root, include_terminals=False, preserve_indices=preorder_index_to_original
        )
        actions[-1].preorder_indices = get_preorder_indices(
            root, include_terminals=False, preserve_indices=preorder_index_to_original
        )
        actions[-1].preorder_depths = get_depths(root)

        while not collapse and ">" in root.nonterminal:
            ic("root is composite, decomposing")
            top_nt, *rest = root.nonterminal.split(">", 1)
            rest = rest[0]
            actions.append(ParseAction(top_nt, NULL_LABEL, 1, parent_span=root.span, nwords=nwords_w_eos))
            root._nonterminal = rest
            root._preorder_index += 1
            actions[-1].right_chain_indices = get_preorder_index_of_right_chain(
                root, include_terminals=False, preserve_indices=preorder_index_to_original
            )
            actions[-1].preorder_indices = get_preorder_indices(
                root, include_terminals=False, preserve_indices=preorder_index_to_original
            )
            actions[-1].preorder_depths = get_depths(root)
            ic(actions[-1])

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
            actions.append(ParseAction(top_nt, NULL_LABEL, 1, parent_span=root.span, nwords=len(root.leaves) + 1))
            root._nonterminal = rest
            root._preorder_index += 1
            actions[-1].right_chain_indices = get_preorder_index_of_right_chain(
                root, include_terminals=False, preserve_indices=preorder_index_to_original
            )
            actions[-1].preorder_indices = get_preorder_indices(
                root, include_terminals=False, preserve_indices=preorder_index_to_original
            )
            actions[-1].preorder_depths = get_depths(root)
            ic(actions[-1])

        while not collapse and ">" in cursor.nonterminal:
            ic("cursor is composite, decomposing")
            top_nt, *rest = cursor.nonterminal.split(">", 1)
            rest = rest[0]
            actions.append(
                ParseAction(top_nt, NULL_LABEL, cursor.depth, parent_span=cursor.span, nwords=len(root.leaves) + 1)
            )
            cursor._nonterminal = rest
            actions[-1].right_chain_indices = get_preorder_index_of_right_chain(
                root, include_terminals=False, preserve_indices=preorder_index_to_original
            )
            actions[-1].preorder_indices = get_preorder_indices(
                root, include_terminals=False, preserve_indices=preorder_index_to_original
            )
            actions[-1].preorder_depths = get_depths(root)
            ic(actions[-1])

        if len(root.children) == 1:
            ic("root has 1 child, finished")
            actions.append(
                ic(ParseAction(NULL_LABEL, root.nonterminal, 0, preterminal_span=root.span, nwords=len(root.leaves)))
            )
            # We popped the root, there is no right-chain remaining
            actions[-1].right_chain_indices = []
            actions[-1].preorder_indices = []
            actions[-1].preorder_depths = []
            return ic(actions[::-1]), preorder_list

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
            actions.append(
                ParseAction(
                    NULL_LABEL, child.nonterminal, cursor.depth, preterminal_span=child.span, nwords=len(root.leaves)
                )
            )
            child = cursor._children.pop(cursor_to_child_idx)
            actions[-1].right_chain_indices = get_preorder_index_of_right_chain(
                root, include_terminals=False, preserve_indices=preorder_index_to_original
            )
            actions[-1].preorder_indices = get_preorder_indices(
                root, include_terminals=False, preserve_indices=preorder_index_to_original
            )
            actions[-1].preorder_depths = get_depths(root)
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
                actions.append(
                    ParseAction(top_nt, NULL_LABEL, cursor.depth, parent_span=cursor.span, nwords=len(root.leaves) + 1)
                )
                cursor._nonterminal = rest
                actions[-1].right_chain_indices = get_preorder_index_of_right_chain(
                    root, include_terminals=False, preserve_indices=preorder_index_to_original
                )
                actions[-1].preorder_indices = get_preorder_indices(
                    root, include_terminals=False, preserve_indices=preorder_index_to_original
                )
                actions[-1].preorder_depths = get_depths(root)
                ic(actions[-1])
            while not collapse and ">" in child.nonterminal:
                ic("child is composite, decomposing child")
                ic((cursor, child, child.depth))
                top_nt, rest = child.nonterminal.split(">", 1)
                # action: extend leg above child (inverse of tree contraction) with top_nt
                actions.append(
                    ParseAction(top_nt, NULL_LABEL, child.depth, parent_span=child.span, nwords=len(root.leaves) + 1)
                )
                child._nonterminal = rest
                actions[-1].right_chain_indices = get_preorder_index_of_right_chain(
                    root, include_terminals=False, preserve_indices=preorder_index_to_original
                )
                actions[-1].preorder_indices = get_preorder_indices(
                    root, include_terminals=False, preserve_indices=preorder_index_to_original
                )
                actions[-1].preorder_depths = get_depths(root)
                ic(actions[-1])

            # action: swap cursor out for a node which has children=[cursor, child]
            actions.append(
                ParseAction(
                    cursor.nonterminal,
                    child.nonterminal,
                    cursor.depth,
                    parent_span=cursor.span,
                    preterminal_span=child.span,
                    nwords=len(root.leaves),
                )
            )  # XXX: Do we need two spans here?
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
            actions[-1].right_chain_indices = get_preorder_index_of_right_chain(
                root, include_terminals=False, preserve_indices=preorder_index_to_original
            )
            actions[-1].preorder_indices = get_preorder_indices(
                root, include_terminals=False, preserve_indices=preorder_index_to_original
            )
            actions[-1].preorder_depths = get_depths(root)
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
        ParseAction(parent=NULL_LABEL, preterminal="ADVP", depth=0, parent_span=None, preterminal_span=(0, 1)),
        ParseAction(parent="ADJP", preterminal="ADJP", depth=1, parent_span=(0, 2), preterminal_span=(1, 2)),
        ParseAction(parent="NP-SUBJ>NP", preterminal="DP>NP", depth=1, parent_span=(0, 3), preterminal_span=(2, 3)),
        ParseAction(parent="S0>IP", preterminal="VP-AUX", depth=1, parent_span=(0, 10), preterminal_span=(3, 4)),
        ParseAction(parent="VP>VP", preterminal="ADVP", depth=2, parent_span=(3, 7), preterminal_span=(4, 5)),
        ParseAction(parent=NULL_LABEL, preterminal="VP", depth=2, parent_span=None, preterminal_span=(5, 6)),
        ParseAction(parent="VP", preterminal="ADVP", depth=3, parent_span=(5, 7), preterminal_span=(6, 7)),
        ParseAction(parent="VP", preterminal="P", depth=2, parent_span=(3, 10), preterminal_span=(7, 8)),
        ParseAction(
            parent="PP-ARG>PP-DIR>PP", preterminal="ADVP", depth=3, parent_span=(7, 10), preterminal_span=(8, 9)
        ),
        ParseAction(parent="NP", preterminal="DP>NP", depth=4, parent_span=(8, 10), preterminal_span=(9, 10)),
    ]
    assert all(a == g for (a, g) in zip(correct_collapsed_actions, collapsed_actions))

    reparsed = parse_by_actions(collapsed_actions, tokens)
    assert all(
        (orig.span == repar.span and orig.label == repar.label)
        for (orig, repar) in zip(sentence.labelled_spans()[0], reparsed.labelled_spans()[0])
    )

    uncollapsed_actions = get_incremental_parse_actions(sentence.clone(), collapse=False)
    correct_uncollapsed_actions = [
        ParseAction(parent=NULL_LABEL, preterminal="ADVP", depth=0, parent_span=None, preterminal_span=(0, 1)),
        ParseAction(parent="ADJP", preterminal="ADJP", depth=1, parent_span=(0, 2), preterminal_span=(1, 2)),
        ParseAction(parent="NP", preterminal="NP", depth=1, parent_span=(0, 3), preterminal_span=(2, 3)),
        ParseAction(parent="DP", preterminal=NULL_LABEL, depth=2, parent_span=(2, 3), preterminal_span=None),
        ParseAction(parent="NP-SUBJ", preterminal=NULL_LABEL, depth=1, parent_span=(0, 3), preterminal_span=None),
        ParseAction(parent="IP", preterminal="VP-AUX", depth=1, parent_span=(0, 10), preterminal_span=(3, 4)),
        ParseAction(parent="VP", preterminal="ADVP", depth=2, parent_span=(3, 7), preterminal_span=(4, 5)),
        ParseAction(parent=NULL_LABEL, preterminal="VP", depth=2, parent_span=None, preterminal_span=(5, 6)),
        ParseAction(parent="VP", preterminal="ADVP", depth=3, parent_span=(5, 7), preterminal_span=(6, 7)),
        ParseAction(parent="VP", preterminal=NULL_LABEL, depth=2, parent_span=(3, 7), preterminal_span=None),
        ParseAction(parent="VP", preterminal="P", depth=2, parent_span=(3, 10), preterminal_span=(7, 8)),
        ParseAction(parent="PP", preterminal="ADVP", depth=3, parent_span=(7, 10), preterminal_span=(8, 9)),
        ParseAction(parent="NP", preterminal="NP", depth=4, parent_span=(8, 10), preterminal_span=(9, 10)),
        ParseAction(parent="DP", preterminal=NULL_LABEL, depth=5, parent_span=(9, 10), preterminal_span=None),
        ParseAction(parent="PP-DIR", preterminal=NULL_LABEL, depth=3, parent_span=(7, 10), preterminal_span=None),
        ParseAction(parent="PP-ARG", preterminal=NULL_LABEL, depth=3, parent_span=(7, 10), preterminal_span=None),
        ParseAction(parent="S0", preterminal=NULL_LABEL, depth=1, parent_span=(0, 10), preterminal_span=None),
    ]
    assert all(a == g for (a, g) in zip(correct_uncollapsed_actions, uncollapsed_actions))

    uncollapsed_reparsed = parse_by_actions(uncollapsed_actions, tokens)
    assert all(
        (orig.span == repar.span and orig.label == repar.label)
        for (orig, repar) in zip(sentence.labelled_spans()[0], uncollapsed_reparsed.labelled_spans()[0])
    )


def test_parser_nwords():
    billinn = NonterminalNode("NP1", [TerminalNode("bíllinn", "x")])
    billinn = NonterminalNode("NP2", [billinn])
    rann = NonterminalNode("VP1", [TerminalNode("rann", "x")])
    rann = NonterminalNode("VP2", [rann])
    ekki = NonterminalNode("NEG1", [TerminalNode("ekki", "x")])
    ekki = NonterminalNode("NEG2", [ekki])
    hratt = NonterminalNode("ADVP1", [TerminalNode("hratt", "x")])
    hratt = NonterminalNode("ADVP2", [hratt])
    rann_ekki_hratt = NonterminalNode("ADVP2", [rann, ekki, hratt])
    billinn_rann_hratt = NonterminalNode("IP", [billinn, rann_ekki_hratt])
    billinn_rann_hratt = NonterminalNode("S-MAIN", [billinn_rann_hratt])
    sentence = NonterminalNode("S0", [billinn_rann_hratt])

    collapsed_actions = get_incremental_parse_actions(sentence.clone(), collapse=True)[0]
    # fmt: off
    correct_collapsed_actions = [
        ParseAction(parent='NULL', preterminal='NP2>NP1', depth=0, parent_span=None, preterminal_span=(0, 1), nwords=1, right_chain_indices=[], preorder_indices=[], preorder_depths=[]),
        ParseAction(parent='S0>S-MAIN>IP', preterminal='VP2>VP1', depth=1, parent_span=(0, 4), preterminal_span=(1, 2), nwords=2, right_chain_indices=[1, 2], preorder_indices=[1, 2], preorder_depths=[0, 1]),
        ParseAction(parent='ADVP2', preterminal='NEG2>NEG1', depth=2, parent_span=(1, 4), preterminal_span=(2, 3), nwords=3, right_chain_indices=[0, 1, 2, 3, 4], preorder_indices=[0, 1, 2, 1, 2, 3, 4], preorder_depths=[0, 1, 2, 3, 4, 3, 4]),
        ParseAction(parent='NULL', preterminal='ADVP2>ADVP1', depth=2, parent_span=None, preterminal_span=(3, 4), nwords=4, right_chain_indices=[0, 1, 2, 2, 4, 5], preorder_indices=[0, 1, 2, 1, 2, 2, 3, 4, 4, 5], preorder_depths=[0, 1, 2, 3, 4, 3, 4, 5, 4, 5])
    ]
    # fmt: on
    assert all(a == g for (a, g) in zip(correct_collapsed_actions, collapsed_actions))

    # fmt: off
    correct_uncollapsed_actions = [
        ParseAction(parent='NULL', preterminal='NP1', depth=0, parent_span=None, preterminal_span=(0, 1), nwords=1, right_chain_indices=[], preorder_indices=[], preorder_depths=[]),
        ParseAction(parent='NP2', preterminal='NULL', depth=1, parent_span=(0, 1), preterminal_span=None, nwords=2, right_chain_indices=[4], preorder_indices=[4], preorder_depths=[0]),
        ParseAction(parent='IP', preterminal='VP1', depth=1, parent_span=(0, 3), preterminal_span=(1, 2), nwords=2, right_chain_indices=[3, 4], preorder_indices=[3, 4], preorder_depths=[0, 1]),
        ParseAction(parent='VP2', preterminal='NULL', depth=2, parent_span=(1, 2), preterminal_span=None, nwords=3, right_chain_indices=[2, 6], preorder_indices=[2, 3, 4, 6], preorder_depths=[0, 1, 2, 1]),
        ParseAction(parent='ADVP2', preterminal='NEG1', depth=2, parent_span=(1, 4), preterminal_span=(2, 3), nwords=3, right_chain_indices=[2, 6, 7], preorder_indices=[2, 3, 4, 6, 7], preorder_depths=[0, 1, 2, 1, 2]),
        ParseAction(parent='NEG2', preterminal='NULL', depth=3, parent_span=(2, 3), preterminal_span=None, nwords=4, right_chain_indices=[2, 5, 8], preorder_indices=[2, 3, 4, 5, 6, 7, 8], preorder_depths=[0, 1, 2, 1, 2, 3, 2]),
        ParseAction(parent='NULL', preterminal='ADVP2>ADVP1', depth=2, parent_span=None, preterminal_span=(3, 4), nwords=4, right_chain_indices=[2, 5, 8, 9], preorder_indices=[2, 3, 4, 5, 6, 7, 8, 9], preorder_depths=[0, 1, 2, 1, 2, 3, 2, 3]),
        ParseAction(parent='S-MAIN', preterminal='NULL', depth=1, parent_span=(0, 4), preterminal_span=None, nwords=5, right_chain_indices=[2, 5, 10, 11], preorder_indices=[2, 3, 4, 5, 6, 7, 8, 9, 10, 11], preorder_depths=[0, 1, 2, 1, 2, 3, 2, 3, 2, 3]),
        ParseAction(parent='S0', preterminal='NULL', depth=1, parent_span=(0, 4), preterminal_span=None, nwords=5, right_chain_indices=[1, 2, 5, 10, 11], preorder_indices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], preorder_depths=[0, 1, 2, 3, 2, 3, 4, 3, 4, 3, 4]),
    ]
    # fmt: on
    uncollapsed_actions = get_incremental_parse_actions(sentence.clone(), collapse=False)[0]
    assert all(a == g for (a, g) in zip(correct_uncollapsed_actions, uncollapsed_actions))


test_parser_nwords()
