import random

from ieg import b

from .errors import ErrorRule


class NounCaseErrorRule(ErrorRule):
    """Error rule class that inflects nouns in a sentence to another case at random.
       TODO: not inflect every noun but just one in a sentence
    """
    needs_pos = True

    @classmethod
    def _apply(cls, data):
        pos = data["pos"]
        changed_text = []
        for p in pos:
            # not changing capitalized words (or at sentence start)
            if p.category == "no" and not p.text[0].isupper():
                res = cls.change_noun_case(p.text, p)
                changed_text.append(res)
            else:
                changed_text.append(p.text)
        return " ".join(changed_text)

    @classmethod
    def change_noun_case(cls, tok, pos) -> str:
        case_set = set(["nf", "þf", "þgf", "ef"])
        case = case_set.intersection(pos.variants)
        other_cases = case_set - case
        rand_case = other_cases.pop()
        bin_no_result = b.lookup_variants(tok, cat="no", to_inflection=rand_case)

        if len(bin_no_result) > 0:
            # the lookup method adds hyphens to unfound compounds; we're removing those (except for in hyphenated words)
            if not "-" in tok:
                return bin_no_result[0].bmynd.replace("-", "")
        return tok
