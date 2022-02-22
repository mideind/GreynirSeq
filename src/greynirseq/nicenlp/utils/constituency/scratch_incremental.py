from icecream import ic

from greynirseq.nicenlp.utils.constituency.greynir_utils import NonterminalNode, TerminalNode
from greynirseq.nicenlp.utils.constituency.incremental_parsing import (
    NULL_LABEL,
    ROOT_LABEL,
    ParseAction,
    get_incremental_parse_actions,
    get_right_chain,
)


class ParseError(Exception):
    def __init__(self, tree, action, reason):
        tree_str = tree.as_nltk_tree().pformat()
        super(ParseError, self).__init__(f"{reason}; caused by action = {action} in tree:\n{tree_str}\n")


def mark_frozen(node, children_only=False):
    if not children_only:
        node.frozen = True
    if node.terminal:
        return
    for child in node.children:
        mark_frozen(child)


class IncrementalParser:
    def __init__(self, tokens):
        """docstring"""
        if not tokens:
            raise ValueError("Must provide tokens or number of tokens")
        self.tokens = tokens
        self.num_tokens = len(tokens)
        self.root = NonterminalNode(ROOT_LABEL)
        self.root.frozen = False
        self.actions = []
        self.token_index = 0

    def is_illegal_action(self, action, as_str=False, strict=True):
        """docstring"""
        if not self.actions:
            # First parse action must an append to pseudo-root
            msg = ""
            if (action.depth == 0) and (not action.parent.is_null()):
                msg = "First parse action must an append to pseudo-root"
            return msg if as_str else bool(msg)
        elif action.depth == 0:
            # Only the first action can add to the pseudo-root
            msg = "Cannot modify pseudo-root"
            return msg if as_str else bool(msg)

        msg = ""
        right_chain = self.get_right_chain()
        if action.depth >= len(right_chain):
            msg = "Action depth exceeds right-chain depth"
        elif strict and right_chain[action.depth].frozen:
            # the last allowable action in a subtree is an uncontraction
            # afterwards, no modification of the subtree is allowed
            msg = "Cannot modify subtree after node uncontraction"
        return msg if as_str else bool(msg)

    def get_tree(self):
        if self.root.children:
            return self.root.children[0]
        return None

    def add_many(self, actions, verbose=False, strict=True):
        for action in actions:
            self.add(action, verbose=verbose, strict=strict)

    def get_right_chain(self):
        return get_right_chain(self.root)

    def add(self, action, verbose=False, strict=True):
        """docstring"""
        if verbose:
            self.root.pretty_print()
        ic(action)
        maybe_error_str = self.is_illegal_action(action, as_str=True, strict=strict)
        if maybe_error_str:
            raise ParseError(self.root, action, reason=maybe_error_str)
        self.actions.append(action)
        right_chain = self.get_right_chain()

        if action.preterminal.is_null():
            ic("incremental_parser: uncontracting non-root node")
            old_nt = right_chain[action.depth].nonterminal
            right_chain[action.depth]._nonterminal = f"{action.parent.label}>{old_nt}"
            mark_frozen(right_chain[action.depth], children_only=True)
            return

        leaf = TerminalNode(self.tokens[self.token_index], "x", "x")
        self.token_index += 1
        preterminal = NonterminalNode(action.preterminal.label, [leaf])
        preterminal.frozen = False

        if action.parent.is_null():
            # append new preterminal only
            right_chain[action.depth].children.append(preterminal)
        else:
            # replace a node X in the right_chain with a node that has X and the new preterminal as its children
            assert 0 < action.depth
            old_parent = right_chain[action.depth - 1].children[-1]
            new_parent = NonterminalNode(action.parent.label, [old_parent, preterminal])
            new_parent.frozen = False
            right_chain[action.depth - 1].children[-1] = new_parent


def test_parser():
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

    # The construction above is equivalent to this tree:
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

    collapsed_actions = get_incremental_parse_actions(sentence.clone())[0]
    correct_collapsed_actions = [
        ParseAction(parent=None, preterminal="ADVP", depth=0, parent_span=None, preterminal_span=(0, 1)),
        ParseAction(parent="ADJP", preterminal="ADJP", depth=1, parent_span=(0, 2), preterminal_span=(1, 2)),
        ParseAction(parent="NP-SUBJ>NP", preterminal="DP>NP", depth=1, parent_span=(0, 3), preterminal_span=(2, 3)),
        ParseAction(parent="S0>IP", preterminal="VP-AUX", depth=1, parent_span=(0, 10), preterminal_span=(3, 4)),
        ParseAction(parent="VP>VP", preterminal="ADVP", depth=2, parent_span=(3, 7), preterminal_span=(4, 5)),
        ParseAction(parent=None, preterminal="VP", depth=2, parent_span=None, preterminal_span=(5, 6)),
        ParseAction(parent="VP", preterminal="ADVP", depth=3, parent_span=(5, 7), preterminal_span=(6, 7)),
        ParseAction(parent="VP", preterminal="P", depth=2, parent_span=(3, 10), preterminal_span=(7, 8)),
        ParseAction(parent="PP-ARG>PP-DIR>PP", preterminal="ADVP", depth=3, parent_span=(7, 10), preterminal_span=(8, 9)),
        ParseAction(parent="NP", preterminal="DP>NP", depth=4, parent_span=(8, 10), preterminal_span=(9, 10))
    ]
    assert all(a == g for (a, g) in zip(correct_collapsed_actions, collapsed_actions))
    # breakpoint()

    parser = IncrementalParser(tokens=ic(tokens))
    parser.add_many(collapsed_actions)
    reparsed = parser.get_tree()
    assert all(
        (orig.span == repar.span and orig.label == repar.label)
        for (orig, repar) in zip(sentence.labelled_spans()[0], reparsed.labelled_spans()[0])
    )

    correct_uncollapsed_actions = [
        ParseAction(parent=None, preterminal="ADVP", depth=0, parent_span=None, preterminal_span=(0, 1)),
        ParseAction(parent="ADJP", preterminal="ADJP", depth=1, parent_span=(0, 2), preterminal_span=(1, 2)),
        ParseAction(parent="NP", preterminal="NP", depth=1, parent_span=(0, 3), preterminal_span=(2, 3)),
        ParseAction(parent="DP", preterminal=None, depth=2, parent_span=(2, 3), preterminal_span=None),
        ParseAction(parent="NP-SUBJ", preterminal=None, depth=1, parent_span=(0, 3), preterminal_span=None),
        ParseAction(parent="IP", preterminal="VP-AUX", depth=1, parent_span=(0, 10), preterminal_span=(3, 4)),
        ParseAction(parent="VP", preterminal="ADVP", depth=2, parent_span=(3, 7), preterminal_span=(4, 5)),
        ParseAction(parent=None, preterminal="VP", depth=2, parent_span=None, preterminal_span=(5, 6)),
        ParseAction(parent="VP", preterminal="ADVP", depth=3, parent_span=(5, 7), preterminal_span=(6, 7)),
        ParseAction(parent="VP", preterminal=None, depth=2, parent_span=(3, 7), preterminal_span=None),
        ParseAction(parent="VP", preterminal="P", depth=2, parent_span=(3, 10), preterminal_span=(7, 8)),
        ParseAction(parent="PP", preterminal="ADVP", depth=3, parent_span=(7, 10), preterminal_span=(8, 9)),
        ParseAction(parent="NP", preterminal="NP", depth=4, parent_span=(8, 10), preterminal_span=(9, 10)),
        ParseAction(parent="DP", preterminal=None, depth=5, parent_span=(9, 10), preterminal_span=None),
        ParseAction(parent="PP-DIR", preterminal=None, depth=3, parent_span=(7, 10), preterminal_span=None),
        ParseAction(parent="PP-ARG", preterminal=None, depth=3, parent_span=(7, 10), preterminal_span=None),
        ParseAction(parent="S0", preterminal=None, depth=1, parent_span=(0, 10), preterminal_span=None)

    ]
    uncollapsed_actions = get_incremental_parse_actions(sentence.clone(), collapse=False)[0]
    assert all(a == g for (a, g) in zip(correct_uncollapsed_actions, uncollapsed_actions))

    parser = IncrementalParser(tokens=tokens)
    parser.add_many(uncollapsed_actions)
    uncollapsed_reparsed = parser.get_tree()
    assert all(
        (orig.span == repar.span and orig.label == repar.label)
        for (orig, repar) in zip(sentence.labelled_spans()[0], uncollapsed_reparsed.labelled_spans()[0])
    )


    # def preorderlist(node):
    #     if node.terminal:
    #         return [node]
    #     ret = [node]
    #     for child in node.children:
    #         ret.extend(preorderlist(child))
    #     return ret
    # uncollapsed_reparsed.pretty_print()
    # preord = preorderlist(uncollapsed_reparsed)
    # for idx, node in enumerate(preorderlist(uncollapsed_reparsed)):
    #     node.preorder_index = idx
    #     print(idx, node)
    # ic.enable()
    # ic()
    # print()

    # leaves = sentence.leaves
    # tok_idx = 0
    # vec = ["ROOT", leaves[tok_idx].text]
    # # vec.append("-------")
    # tok_idx += 1
    # for action in uncollapsed_actions:
    #     vec.append(action)
    #     if action.parent != NULL:
    #         vec.append(action.parent)
    #     if action.preterminal != NULL:
    #         vec.append(action.preterminal)
    #         if tok_idx < len(leaves):
    #             add_tok = True
    #     # vec.append("-------")
    #     if add_tok:
    #         vec.append(leaves[tok_idx].text)
    #         tok_idx += 1
    #     add_tok = False

    # ic(vec)
    # breakpoint()


test_parser()


def test_incremental_vector():
    # [mjög, flotti, bíllinn, hafði, ekki, ...]
    # embed ROOT as ROOT0, mjög is bert0
    # optionally add positional embeddings
    # input state:   [root0, bert0]
    #                 p0+pn?  p0
    # classif inp.:  [root1 bert1]
    # classif outp.: (NULL, ADVP, 0)
    #
    # (since this is first action, only append is available)
    #     append ADVP to ROOT
    #     embed ADVP as advp0
    #     add span positional embedding to advp0 (startpos + endpos)/2
    #
    # input state:   [root1, mjog1, advp0, flotti0]
    #                        p0+p1  p0+p1   p1+p2
    # output:        [root2, mjog2, advp1, flotti1]
    # classif outp.: (ADJP, ADJP, 1)
    #                []
    #
    # input state:   [root2, mjog2, advp1, flotti1, adjp0, adjp0, billinn0]
    #                                               p1+p2  p0+p2  p2+p3
    # output:        [root3, mjog3, advp2, flotti2, adjp1, adjp1, billinn1]
    # classif outp.: (NPSBJ, NP, 1)
    #
    #
    # how would the uncontraction of np0 into dp>np0 come about?
    #
    # we could alternatively
    # input state:   [root3, mjog3, advp2, flotti2, adjp1, adjp1, billinn1]
    #
    # how would the uncontraction of np0 into dp>np0 come about?
    # input state:   [root3, mjog3, advp2, flotti2, adjp1, adjp1, billinn1, npsbj0, np0, hafdi0]
    #                                                                       p0+p3  p2+p3  p3+p4
    # if PARENT==NULL
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    #
    pass
