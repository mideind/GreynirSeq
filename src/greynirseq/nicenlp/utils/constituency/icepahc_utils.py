import nltk
from icecream import ic

import greynirseq.nicenlp.utils.constituency.greynir_utils as greynir_utils


UNHANDLED = "UNHANDLED"
ICEPAHC_SKIP_TAGS = ("CODE", "CPDE", "ID", "META")
UNWRAP = ("REF",)  # or discard


class DiscardTreeException(Exception):
    pass


# what to do with this?
#                   ____________________|___________________________
#                 CODE                                              |
#      ____________|__________________________________              |
#     |                       NP                      |             |
#     |             __________|_________              |             |
#     |            |                  NP-POS          |             |
#     |            |                    |             |             |
#    CODE         N-N                  NS-G          CODE           ID
#     |            |                    |             |             |
# <heading> Þjónusta-þjónust     kennimanna-kenni </heading> 1150.HOMILIUBOK.
#                  a                  maður                     REL-SER,.2


# ( (IP-MAT (PP (P 0)
#               (NP (PRO-D Því-það)))
#           (NP-SBJ (Q-N allar-allur) (ADJS-N bestu-góður) (NS-N húsfreyjur$-húsfreyja) (D-N $nar-hinn) (ADV þar-þar))
#           (BEDI voru-vera)
#           (PP (P í-í)
#               (NP (ADJ-D góðum-góður) (NS-D efnum-efni))))
#   (ID 1902.FOSSAR.NAR-FIC,.15))
# ( (IP-MAT (CONJ og-og)
#           (NP-SBJ *con*)
#           (HVDI höfðu-hafa)
#           (NP-OB1 (N-A gnótt-gnótt))
#           (PP (P í-í)
#               (NP (N-D búi-bú)))
#           (. .-.))
#   (ID 1902.FOSSAR.NAR-FIC,.16))

# (TOP (IP*MAT (PP (NP (PRO*D Því)))
#              (NP*SBJ (Q*N allar)
#                      (ADJS*N bestu)
#                      (NS*N húsfreyjurnar)
#                      (ADV þar-þar))
#              (BEDI voru)
#              (PP (P í)
#                  (NP (ADJ*D góðum)
#                      (NS*D efnum)))))
# (TOP (IP*MAT (CONJ og)
#              (HVDI höfðu)
#              (NP*OB1 (N*A gnótt))
#              (PP (P í)
#                  (NP (N*D búi)))
#           (. .-.)))


# SKEMAUPPLÝSINGAR
# https://linguist.is/icelandic_treebank/Splitting_and_joining_words#Items_treated_as_unitary
# some tokens have been split into two and are analyzed as separate words (with different POS tags)
#   (VBPI veis$-vita) (NP-SBJ (PRO-N $tu-þú))
# whereas some tokens have been split into two (or more) parts of one whole where they use a different markup
#  (NP-SBJ (N-N (N21-G manns) (N22-N bani))
#  (N-D (N31-D hús-hús) (N32-D bónda-bóndi) (N33-D veislu-veisla))
# This can also be nested such as:
#  (NS-A (NS21 (NS21 morð<dash>-morð) (CONJ og-og) (NS21 mann-maður)) (NS22 drápsbréf-drápsbréf))


# Nonterminals with -TTT are for spelling errors such as:
#     (ADVP-TTT (ADV meðaukunarlega-meðaukunarlega))


# Sharing modifiers:
# ADVX: means this shares a modifier (amod) with another token
#       (áin skar svo [skýrt og greinilega] úr landamerkjum hvorratveggja jarðanna) - greinilega is ADVX
# NX: similarly a noun can for example share a NP-POSS relationship
#       (sumir samhljóðendur hafa sín [líkneski og nafn og jartein]) - nafn and jarteinn are NX
# RPX
# https://linguist.is/icelandic_treebank/Conjunction#Shared_modifiers
# does not apply to PP


# See this for information on REP, QTP, FRAG, META, LATIN, CODE
# https://linguist.is/icelandic_treebank/Nonstructural_labels


# See this for information on (suffixes) -LFD, -RSP, -SPE, -SPR
# https://linguist.is/icelandic_treebank/Phrase_Types
# https://linguist.is/icelandic_treebank/Clause_level_constituents


# See this for information on *exp*, *arb*, *con*
# https://linguist.is/icelandic_treebank/Empty_categories


# See this for information on tags like IP-MAT=1 and NP-PRN-1
# See https://www.ling.upenn.edu/~beatrice/annotation/
# "-#" (a hyphen followed by a numeric index) is used to coindex antecedents and their traces, as well as expletives (overt or empty) that are associated with a clause or noun phrase.
# "=#" (an equals sign followed by a numeric index) is used to coindex gapped clauses with full clauses. See Gapping, Right-node raising.

LEGAL_POS_HEADS = set(
    (
        # verbs
        "BAG",
        "BAN",
        "BE",
        "BEDI",
        "BEDS",
        "BEI",
        "BEN",
        "BEPI",
        "BEPS",
        "DAG",
        "DAN",
        "DO",
        "DODI",
        "DODS",
        "DOI",
        "DON",
        "DOPI",
        "DOPS",
        "HAG",
        "HAN",
        "HV",
        "HVDI",
        "HVDS",
        "HVI",
        "HVN",
        "HVPI",
        "HVPS",
        "MAG",
        "MAN",
        "MD",
        "MDDI",
        "MDDS",
        "MDI",
        "MDN",
        "MDPI",
        "MDPS",
        "RAG",
        "RAN",
        "RD",
        "RDDI",
        "RDDS",
        "RDI",
        "RDN",
        "RDPI",
        "RDPS",
        "VAG",
        "VAN",
        "VB",
        "VBDI",
        "VBDS",
        "VBI",
        "VBN",
        "VBPI",
        "VBPS",
        # adverbs
        "ADV",
        "WADV",
        "ADVR",
        "ADVS",
        # adjectives
        "ADJ",
        "ADJR",  # comparative
        "ADJS",  # superlative
        "WADJ",  # wh-adjective
        "SUCH",
        #
        # conjunctions
        "CONJ",
        "C",
        # determiners
        "D",
        "WD",  # wh-determiner
        # nouns
        "N",
        "NS",
        "NPR",
        "NPRS",
        "NX",
        "ONE",
        "ONES",
        # particles
        "FP",  # focus particle
        "RP",  # adverbial particle
        "RPX",  # adverbial particle
        # pronouns
        "PRO",
        "WPRO",
        # prepositions
        "P",
        # quantifiers
        "Q",  # 'fáir'
        "QR",  # 'færri'
        "QS",  # 'fæstir'
        "WQ",  # 'hver'
        # miscellaneous
        "LS",  # list marker
        "FW",  # foreign word
        #####
        "TO",
        "NEG",
        "ALSO",
        "NUM",
        "ES",
        "INTJ",
        "OTHER",
        "OTHERS",
        # shared modifers
    )
)

CASES = ("N", "A", "D", "G")


def split_wordform_lemma(word_str, pos_str):
    # hard-coded exceptions
    if pos_str == '"':
        return '"', '"'

    if should_skip_by_word_str(word_str):
        return None

    # http://linguist.is/icelandic_treebank/PP#Stranded_prepositions
    braces = ("{með}", "{um}", "{eftir}", "{fyrir}")
    dashes = ("<dash/>", "<dash/>-<dash/>", "<dash>", "</dash>")
    if word_str in dashes:
        return "-", "-"
    elif word_str == r'"-"-"':
        return '"', '"'
    elif word_str == "---" or len(set(word_str)) == 1 and "-" in word_str:
        return "-", "-"
    elif word_str == "-":
        return "-", "-"
    elif word_str == "víst-vís/viss":
        return "víst", "viss"
    elif word_str == "sylgju$":
        return "sylgju$", "sylgja"
    elif word_str == "mynda--myndafloti":
        return "myndafloti", "myndafloti"
    elif word_str == "<dash/>ferð":
        # the original text was:  "flug"-ferð
        return "-ferð", "-ferð"
    elif word_str == "<dash/>sálarflokkur$":
        # the original text was:  Líkama- og -sálarflokkurinn
        return "-sálarflokkur", "-sálarflokkur"
    elif word_str in braces:
        return UNHANDLED, UNHANDLED

    # or word_str == "different_editions_GJÖR_GJÖRÐ"
    # or word_str == "{með}"

    hyphen_count = word_str.count("-")

    if hyphen_count != 1:
        parts = word_str.split("-")
        if len(set(parts[1:])) == 1:
            # the extra items are duplicate such as: minni-lítill-lítill
            return parts[0], parts[1]
        print(word_str, pos_str)
        # ic(word_str, pos_str)
        breakpoint()
    wordform, lemma = word_str.split("-", 1)
    wordform = wordform.replace("<dash/>", "-")
    lemma = lemma.replace("<dash/>", "-")
    if word_str.count("/") == 1 and word_str.count("-") == 1:
        wordform = word_str.split("-")[0].split("/")[0]
        lemma = word_str.split("-")[1].split("/")[0]
        return wordform, lemma
    if "/" in lemma and "/" != lemma:
        # ic(word_str, pos_str)
        print(word_str, pos_str)
        breakpoint()

    return wordform, lemma


def should_skip_by_word_str(word_str):
    word_str_lower = word_str.lower()
    if (
        word_str == "0"
        or word_str_lower.startswith("*t*")
        or word_str_lower.startswith("*ich*")
        or word_str_lower.startswith("*pro*")
        or word_str_lower.startswith("*con*")
        or word_str_lower.startswith("*exp*")
        or word_str_lower.startswith("*arb*")
        or word_str == "*"
        or word_str.startswith("*-")  # *-2
        or word_str == "<slash/>p92"
    ):
        # word_idx==1082  different_editions_GJÖR_GJÖRÐ
        return True
    return False


def skip_leaf_by_pos(pos_str):
    return pos_str.startswith("CODE")


def maybe_fix_pos(pos_str):
    pos_str = REMAP_POS.get(pos_str, pos_str)
    if pos_str == "FÐ":
        return "FP"
    return pos_str


def maybe_fix_by_pos_str_and_wordlemma(wordlemma, pos_str):
    if wordlemma == "þar-þar" and pos_str == "WADVP":
        return wordlemma, "WADVP"

    if pos_str == "ADJP":
        fixed = "ADJ"
        if wordlemma == "tólfti-tólf":
            return wordlemma, fixed
        if wordlemma == "limaður-limaður":
            return wordlemma, fixed

    if pos_str == "CONJP":
        fixed = "CONJ"
        if wordlemma in ("og-og", "eður-eður", "eða-eða"):
            return wordlemma, fixed

    if pos_str == "RX":
        fixed = "RPX"
        if wordlemma in ("á-á", "í-í", "að-að"):
            return wordlemma, fixed

    if pos_str == "FOREIGN":
        fixed = "FW"
        if wordlemma in ("cantori-cantori", "misericordie-misericordie"):
            return wordlemma, fixed

    if wordlemma == "þar-þar" and pos_str == "VDPI":
        return wordlemma, "WADVP"

    if pos_str == "VDPI":
        fixed = "VBPI"
        if wordlemma in ("vilt-vilja", "vill-vilja"):
            return wordlemma, fixed

    if pos_str == "NRP":
        fixed = "NPR"
        return wordlemma, fixed

    if pos_str == "ADVP":
        fixed = "ADV"
        # ofan-ofan
        return wordlemma, fixed

    if pos_str == "PRON":
        fixed = "PRO"
        # það-það
        return wordlemma, fixed

    if pos_str == "NP-ADT":
        if wordlemma == "járnum-járna":
            return "járnum-járn", "NS-D"

    if pos_str == "POR-D":
        # sér-sig
        return wordlemma, "PRO-D"
    if pos_str == "POR-A":
        return wordlemma, "PRO-A"

    if pos_str == "RPO-D":
        # sér-sig
        return wordlemma, "PRO-D"

    if pos_str == "VBDP":
        # sér-sig
        return wordlemma, "VBDS"

    # if pos_str == "N-S":
    #     return  wordlemma, "NS"

    if pos_str == "N-S":
        # hryggjar-hrygg
        return wordlemma, "NS-G"

    if pos_str == "NPR-S":
        # Óspakur-óspakur
        return wordlemma, "NPRS-N"

    if pos_str == "DVBN":
        return wordlemma, "VBN"

    if pos_str == "MS-N":
        return wordlemma, "QS-N"

    if pos_str == "RBN":
        return wordlemma, "RDN"

    if pos_str == "WRPO-N":
        return wordlemma, "WPRO-N"

    if pos_str == "QDJ-A":
        return wordlemma, "ADJ-A"

    if pos_str == "POS-D":
        return wordlemma, "PRO-D"

    if pos_str == "NPRO-A":
        return wordlemma, "NPR-A"

    if pos_str == "HDDI":
        return wordlemma, "HVDI"

    if pos_str == "DV":
        return wordlemma, "ADV"

    # (NPR meistara-meistari) og fleiri
    # (WNP-N-1 hver-hver)
    # (NPR Botna-botna)
    # REP
    return wordlemma, pos_str


def convert_icepahc_nltk_tree_to_node_tree(tree, lowercase_pos=False, dummy_preterminal=None):
    if not isinstance(tree, nltk.Tree):
        raise ValueError(f"Unexpected tree: {str(tree)}")

    def _is_icepahc_leaf(node):
        if not isinstance(node, nltk.Tree):
            raise ValueError(f"Expected nltk.Tree instance, got {node}")
        if isinstance(node[0], str):
            return True
        return False

    def visitor(node):
        children = []
        if _is_icepahc_leaf(node):
            # ic(f"Found leaf: {str(node)}")
            if node.label() == "ID":
                return None, node[0]
            # if "{" in node[0]:
            #     breakpoint()
            if node.label() in ICEPAHC_SKIP_TAGS:
                # ic(f"Skipped: {node.label()}")
                return None, None
            # NonterminalNode()
            word_lemma_str = node[0]
            pos_str = node.label()
            pos_str = maybe_fix_pos(node.label())
            # if "*" == ic(word_lemma_str):
            #     ic(word_lemma_str)
            result = split_wordform_lemma(word_lemma_str, pos_str)
            if result is None:
                return None, None
            # wordform, lemma  = maybe_fix_by_pos_str_and_wordlemma(word_lemma_str, pos_str)
            wordform, lemma = result
            reformatted_pos_str = pos_str
            if lowercase_pos:
                reformatted_pos_str = pos_str.lower().replace("-", "_")
            new_node = greynir_utils.TerminalNode(
                wordform, reformatted_pos_str, category=reformatted_pos_str, skip_terminal_check=True
            )
            if dummy_preterminal is not None:
                new_node = greynir_utils.NonterminalNode(dummy_preterminal, [new_node])
            return new_node, None

        tree_id = None
        if node.label() in ICEPAHC_SKIP_TAGS:
            return None
        for child in node:
            result = visitor(child)
            if result is None:
                continue
            maybe_child, maybe_id = result
            if maybe_child is None and maybe_id is None:
                pass
            if maybe_id is not None:
                tree_id = maybe_id
                continue
            if maybe_child is None:
                continue
            children.append(maybe_child)
        if not children:
            return None, maybe_id
        label = node.label()
        if label:
            label = parse_nt(label)
        new_node = greynir_utils.NonterminalNode(label, children=children)
        if new_node.nonterminal is None or new_node.nonterminal.strip() == "":
            if new_node.children:
                return new_node.children[0], tree_id
            else:
                return None, tree_id
        return new_node, tree_id

    res, tree_id = visitor(tree)
    if res is not None:
        res = merge_split_leaves(res)
        res = maybe_wrap_terminals_in_tree(res)
    return res, tree_id


def get_nonterminals(node_tree):
    if node_tree.terminal:
        return []
    nts = [node_tree.nonterminal]
    for child in node_tree.children:
        nts.extend(get_nonterminals(child))
    return nts


def parse_nt(nt_str):
    nt_str = maybe_fix_nt(nt_str)
    parts = nt_str.replace("=", "-").split("-")

    head = parts[0]
    parts = parts[1:]
    head, *head_extra = head.split("+")

    # while parts and (parts[-1].isnumeric() or parts[-1] == "NaN"):
    #     parts.pop(-1)
    # discard coindexing
    parts = [part for part in parts if not (part.isnumeric() or part == "NaN" or part == "")]

    illegal_parts = (
        []
        if (head in LEGAL_NT_HEADS or head in NT_DISQUALIFY_BOTH_HEAD_AND_PARTS or head in LEGAL_POS_HEADS)
        else [head]
    )
    if len(parts) > 1:
        illegal_parts = illegal_parts.extend(
            [part for part in parts if not (part in LEGAL_NT_FLAGS or part in NT_DISQUALIFY_BOTH_HEAD_AND_PARTS)]
        )
    if head_extra:
        illegal_parts.extend(
            [part for part in head_extra if not (part in LEGAL_POS_HEADS or part in NT_DISQUALIFY_BOTH_HEAD_AND_PARTS)]
        )
    if illegal_parts:
        ic(nt_str, illegal_parts, head in LEGAL_NT_HEADS)
        breakpoint()

    h_disqualify = head in NT_DISQUALIFY_BOTH_HEAD_AND_PARTS
    disqualify = (NT_DISQUALIFY_BOTH_HEAD_AND_PARTS | LEGAL_NT_HEAD_BUT_DISQUALIFY_AS_PART) & set(parts)
    cleaned = head
    if parts:
        cleaned = head + "-" + "-".join(parts)
    if h_disqualify or disqualify:
        # ic(nt_str, parts, cleaned, h_legal, h_disqualify, legal, disqualify)
        raise DiscardTreeException(f"Could not parse: '{nt_str}'")
    return cleaned


def merge_split_leaves(node_tree):
    # example: (NP (N-G (N21-G konung-konungur) (N22-G dóms$-dómur)) (D-G $ins-hinn)))
    if not any("$" in leaf.text for leaf in node_tree.leaves):
        return node_tree
    if node_tree.terminal:
        return node_tree.terminal

    def delete(node, other):
        if node.terminal:
            raise ValueError("Cannot delete leaf inside leaf")
        for idx in range(len(node.children) - 1, -1, -1):
            child = node.children[idx]
            if child is other:
                node._children.pop(idx)
                break
            elif child.nonterminal:
                should_delete = delete(child, other)
                if should_delete:
                    node._children.pop(idx)
        if not node.children:
            return True
        return False

    def merge(node, left_node, right_node):
        if node.terminal:
            raise ValueError("Leaves have no children")
        for child in node.children:
            if child is left_node:
                # XXX: Do we want to change the leaf POS?
                # XXX: Do we want to modify the label of the governing nonterminal?
                child._text = left_node.text.replace("$", "") + right_node.text.replace("$", "")
                return True
            if child.nonterminal:
                merged = merge(child, left_node, right_node)
                if merged:
                    return True
        return False

    leaves = [leaf for leaf in node_tree.leaves]
    merge_ends = [idx for idx, leaf in enumerate(leaves) if leaf.text.startswith("$") and idx > 0]
    merge_pairs = [(leaves[idx - 1], leaves[idx]) for idx in merge_ends if leaves[idx - 1].text.endswith("$")]
    for left_node, right_node in merge_pairs:
        delete(node_tree, right_node)
        merge(node_tree, left_node, right_node)
    return node_tree


def maybe_unwrap_nonterminals_in_tree(node, unwrap_set=UNWRAP):
    if node.terminal:
        raise ValueError("Cannot unwrap leaf")
    found = False
    for idx in range(len(node.children) - 1, -1, -1):
        child = node.children[idx]
        if child.terminal:
            continue
        if child.nonterminal in unwrap_set:
            node._children.pop(idx)
            maybe_unwrap_nonterminals_in_tree(child, unwrap_set)
            node._children[idx : idx + 1] = child.children
            found = True
    return found


REMAP_POS = {
    "'": "PUNCT",
    ",": "PUNCT",
    "-": "PUNCT",
    ".": "PUNCT",
    ":": "PUNCT",
    ";": "PUNCT",
    "Q+\u0143UM-D": "Q+NUM-D",
    "OTHER+G": "OTHER-G",
}

VPWRAP = "VP"
WRAP_POS_WITH_NT = {
    "bag": VPWRAP,
    "ban": VPWRAP,
    "be": VPWRAP,
    "bedi": VPWRAP,
    "beds": VPWRAP,
    "bei": VPWRAP,
    "ben": VPWRAP,
    "bepi": VPWRAP,
    "beps": VPWRAP,
    "dag": VPWRAP,
    "dan": VPWRAP,
    "do": VPWRAP,
    "dodi": VPWRAP,
    "dods": VPWRAP,
    "doi": VPWRAP,
    "don": VPWRAP,
    "dopi": VPWRAP,
    "dops": VPWRAP,
    "hag": VPWRAP,
    "han": VPWRAP,
    "hv": VPWRAP,
    "hvdi": VPWRAP,
    "hvds": VPWRAP,
    "hvi": VPWRAP,
    "hvn": VPWRAP,
    "hvpi": VPWRAP,
    "hvps": VPWRAP,
    "mag": VPWRAP,
    "man": VPWRAP,
    "md": VPWRAP,
    "mddi": VPWRAP,
    "mdds": VPWRAP,
    "mdi": VPWRAP,
    "mdn": VPWRAP,
    "mdpi": VPWRAP,
    "mdps": VPWRAP,
    "rag": VPWRAP,
    "ran": VPWRAP,
    "rd": VPWRAP,
    "rddi": VPWRAP,
    "rdds": VPWRAP,
    "rdi": VPWRAP,
    "rdn": VPWRAP,
    "rdpi": VPWRAP,
    "rdps": VPWRAP,
    "vag": VPWRAP,
    "van": VPWRAP,
    "vb": VPWRAP,
    "vbdi": VPWRAP,
    "vbds": VPWRAP,
    "vbi": VPWRAP,
    "vbn": VPWRAP,
    "vbpi": VPWRAP,
    "vbps": VPWRAP,
}


def maybe_wrap_terminals_in_tree(node, wrap_map=WRAP_POS_WITH_NT):
    if node.terminal:
        if node.terminal.lower() not in wrap_map:
            return node
        return greynir_utils.NonterminalNode(wrap_map[node.terminal], children=[node])
    new_children = [maybe_wrap_terminals_in_tree(child) for child in node.children]
    return greynir_utils.NonterminalNode(node.nonterminal, children=new_children)


def merged_pos_heads_precedence(merged_heads):
    # stórmikið                  'ADJ+Q': 1,
    # jafnfagurt                 'ADV+ADJ': 3,
    # jafnær                     'ADV+ADVR': 1,
    # héreftir,utanfyrir         'ADV+P': 3,
    # hvergi,ofurlítið           'ADV+Q': 32,   # note, hvergi is wrong and should be Q+ADV
    # velflestir                 'ADV+QS': 1,
    # jafngóð                    'ADVR+ADJ': 52,
    # jafnstærri                 'ADVR+ADJR': 1,
    # jafnvel                    'ADVR+ADV': 107,
    # jafnframt                  'ADVR+P': 4,
    # jafnmargir                 'ADVR+Q': 27,
    # jafnbúnir                  'ADVR+VAN': 1,
    # hinumegin,þessháttar       'D+N': 3,
    # snjámikið                  'N+Q': 2,
    # Hálfdánarheimtur           'NPR+NS': 1,
    # 30um                       'NUM+N': 1,
    # hálf-þrítugur              'NUM+NUM': 1,
    # einusinni,einslags         'ONE+N': 9,
    # eitthvað                   'ONE+Q': 645,
    # einhverskonar              'ONE+Q+N': 7,
    # einhverra                  'ONES+Q': 1,
    # öðrumegin                  'OTHER+N': 13,
    # annarstaðar                'OTHER+NS': 1,
    # annaðtveggja               'OTHER+NUM': 1,
    # annaðhvort                 'OTHER+Q': 2,
    # annaðhvort                 'OTHER+WPRO': 50,
    # aðrirtveggju               'OTHERS+NUM': 1,
    # sitthvað                   'PRO+Q': 1,
    # einskisvert,margfróður     'Q+ADJ': 3,
    # hvergi                     'Q+ADV': 137,
    # alltof                     'Q+ADVR': 1,
    # nokkurskonar,nokkurntíma   'Q+N': 30,
    # allrahanda                 'Q+NS': 2,
    # hvorratveggja              'Q+NUM': 124,
    # Hvort tveggja              'Q+NUM21': 1, 'Q+NUM22': 1,
    # eitthvað                   'Q+ONE': 1,
    # alllítið                   'Q+Q': 3,
    # allraminnsta               'Q+QS': 1,
    # hvergi                     'Q+WADV': 2,
    # einnhver                   'Q+WPRO': 1,
    # jafnháir,meiriháttar       'QR+ADJ': 2,
    # hvergi                     'WADV+Q': 1,  # note, should be Q+WADV
    # annaðhvort                 'WPRO+Q': 1}  # note, should be  OTHER+WPRO
    pass


def maybe_fix_nt(nt):
    fixes = {
        "ADJP-OC": "ADJP-LOC",
        "ADV-RSP": "ADVP-RSP",
        "ADVP-RMP": "ADVP-TMP",
        "ADVP-RSP-RSP": "ADVP-RSP",
        "CONPJ": "CONJP",
        "CONJP\x7f": "CONJP",
        "CONJP-PP": "CONJP",
        "CP-DEG-2-SPE": "CP-DEG-SPE",
        "CP-THT-PRN-NaN": "CP-THT-PRN",
        "CP-THT-SPE1": "CP-THT-SPE",
        "röntgenaugu": "NP",
        "NP\x7f": "NP",
        "IP-INF-SPE-PRN-2-2": "IP-INF-PRN-SPE",
        "IP-SUB-SPE3": "IP-SUB-SPE",
        "NP-SMR": "NP-MSR",
        "ADJP-OC": "ADJP-LOC",
        "MP-PRN": "NP-PRN",
        "NP-AB1": "NP-OB1",
        "NP-PB1": "NP-OB1",
        "CP-AUE-ADV": "CP-QUE-ADV",
        "IP-SUB-SPE-3\x7f": "IP-SUB-SPE",
        "IP-ING": "IP-INF",
        "NP-DIRJ": "NP-DIR",
        "NPX": "NX",
        "PX": "PP",
        "WNX": "NX",
        "IMP-IMP": "IP-IMP",
        "NP-NUM": "NUMP",
    }
    return fixes.get(nt, nt)


LEGAL_NT_HEADS = set(
    (
        "ADJP",
        "ADJX",
        "ADV",  # (ADVP (ADV (ADV21 full-full) (ADJ22 fagurt-fagur)))
        "ADVP",
        "ADVX",
        "CONJP",
        "CP",
        "ENGLISH",
        "FOREIGN",
        "FRAG",  # fragment, not enough to construct an IP
        "FS",  # false start
        "INTJP",
        "IP",
        "LATIN",
        "NEGP",
        "NP",
        "NUMP",
        "PP",
        "QP",
        "QTP",  # quotation phrase
        "QX",
        "REF",  # reference (citation)
        "REP",  # repetition
        "RRC",  # reduced relative clauses
        "TRANSLATION",
        "VP",
        "WADJP",
        "WADJX",
        "WADVP",
        "WNP",
        "WPP",  # pied-piping, https://linguist.is/icelandic_treebank/WPP
        "WQP",
        ## actually pos
    )
)

LEGAL_NT_FLAGS = set(
    (
        "ABS",  # IP-ABS
        "ADT",  # NP-ADT, tækjafall
        "ADV",  # NP-ADV
        "BY",  # PP-BY
        "CAR",  # CP-CAR, clause-adjoined relatives
        "CLF",  # CP-CLF, cleft (it is money that I love)
        "CMP",  # CP-CMP
        "COM",
        "DEG",  # CP-DEG
        "DIR",
        "ELAB",  # https://linguist.is/icelandic_treebank/Disfluencies
        "EOP",  # CP-EOP, purpose/relative infinites
        "EXL",  # CP-EXL, exclamatives, 'en sú óheppni!'
        "FRL",  #  CP-FRL, adverbial free relatives https://linguist.is/icelandic_treebank/CP-FRL
        "IMP",
        "INF",
        "LFD",  # left dislocation
        "LOC",
        "MAT",
        "MSR",  # measure (comparison)
        "OB1",
        "OB2",
        "OB3",
        "POS",
        "PPL",  # participial clause
        "PRD",
        "PRN",
        "PRP",
        "QUE",
        "REL",
        "RSP",  # https://linguist.is/icelandic_treebank/RSP
        "SBJ",
        "SMC",  # small-clause
        "SPE",
        "SPR",  # secondary predicate
        "SUB",
        "THT",
        "TMC",  # CP-TMC, tough movement complements  https://www.ling.upenn.edu/~beatrice/annotation/syn-sub.html#tough_movement
        "TMP",
        "TTT",  # spelling error
        "VOC",  # NP-VOC, vocative (Einar, hvar eru paprikurnar?)
        # Cases, only temporary, remove this later
        # "N", "A", "D", "G"
    )
)

NT_DISQUALIFY_BOTH_HEAD_AND_PARTS = set(
    (
        "2SBJ",
        "BEFORE",
        "BP",
        "DP",
        "FINITE",
        "KOMINN",
        "LLL",
        "MISS",
        "NB",
        "SBP",
        "SENT",
        "SPJ",
        "SUV",
        "UNKNOWN",
        "VERB",
        "WNX",
        "WXP",
        "X",
        "XP",
        "ZZZ",
        "NS21",
    )
)

LEGAL_NT_HEAD_BUT_DISQUALIFY_AS_PART = set(("REP", "CONJP", "NPR", "PRO"))

POS_DISQUALIFY_PARTS = set(("YYY", "S", "INF", "INF"))

POS_LEGAL_PARTS = set(("N", "A", "D", "G", "TTT"))

#  'ADT': 1,
#  'ADV': 1,
#  'C': 1,
#  'CASE': 1,
#  'DEG': 1,
#  'INF': 3,
#  'LOC': 1,
#  'MSA': 1,
#  'MSN': 1,
#  'NPR': 1,
#  'NSNSP': 1,
#  'NUM': 1,
#  'OB1': 3,
#  'OB2': 1,
#  'PRD': 4,
#  'PRN': 2,
#  'Q': 2,
#  'S': 2,
#  'SBJ': 653,
#  'SPE': 2,
#  'THT': 3,
#  'V': 1,
#  'WPRO': 5,  # (OTHER-WPRO annaðhvort) # this is a typo
#  'YYY': 1}


# pos used as phrase unit
# "ADJR-A",  # (ADJP (ADJ-A (ADV21 viður-viður) (ADJ22 lík-lík)))
# "ADV",  # (ADJP (ADJ-A (ADV21 viður-viður) (ADJ22 lík-lík)))
# (VBN (VBN máð) (CONJ og) (VBN ritað))
# BEPI, same as BEN
# (C (C21 sem-sem) (C22 að-að))
# (CONJ (OTHER21 annað-annar) (WPRO22 hvort-hvor))

# example1 = nltk.Tree("NP", [nltk.Tree("N-G", [nltk.Tree("N21-G", ["konung-konungur"]), nltk.Tree("N22-G", ["dóms$-dómur"])]), nltk.Tree("D-G", ["$ins-hinn"])])
# example2 = nltk.Tree("N-G", [nltk.Tree("D-G", ["dóms$-dómur"]), nltk.Tree("N22-G", ["$ins-hinn"])])
# example3 = nltk.Tree("NP", [example1, nltk.Tree("NP", [nltk.Tree("N21-G", ["konung-konungur"])]), example2])
# # example1.pretty_print()
# example3, _id = convert_icepahc_nltk_tree_to_node_tree(example3)
# example3.pretty_print()
# res = merge_split_leaves(example3)
# breakpoint()
