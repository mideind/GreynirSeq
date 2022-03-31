import random
from .errors import ErrorRule

from ieg import b


class NounCaseErrorRule(ErrorRule):
    """Error rule class that inflects a noun to another case at random.
    """
    needs_pos = True

    @classmethod
    def _apply(cls, data):
        text, pos = data["text"], data["pos"]
        nps = data["tree"].all_matches("NP")
        np = random.choice(list(nps))
        changed_text = []
        for tok, t_pos in zip(text.split(), pos):
            if t_pos.category == "no":
                changed_text.append(cls.change_noun_case(tok, t_pos))
            else:
                changed_text.append(tok)
        return " ".join(changed_text)

    @classmethod
    def change_noun_case(cls, tok, pos):
        case_set = set(["nf", "þf", "þgf", "ef"])
        case = case_set.intersection(pos.variants)
        other_cases = case_set - case
        rand_case = other_cases.pop()
        
        bin_no_result = b.lookup_variants(tok, cat="no", to_inflection=rand_case)

        if len(bin_no_result) > 0:
            return bin_no_result[0].bmynd
        return tok