from ieg import bin_islenska

from .errors import ErrorRule
import random


class NounCaseErrorRule(ErrorRule):
    """Error rule class that inflects nouns in a sentence to another case at random.
    TODO: not inflect every noun, just one in a sentence.
    """

    needs_pos = True
    CUSTOM_RATIO = 0.5

    @classmethod
    def _apply(cls, data):

        text, pos = data["text"], data["pos"]
        text_list = text.split()

        if not super().tok_and_pos_match(cls, text_list, pos):
            return text
        changed_text = []

        if len(text_list) != len(pos):
            return text
        for idx, p in enumerate(pos):
            # not changing capitalized words (or at sentence start)
            if p.category == "no" and not text_list[idx][0].isupper():
                res = cls.change_noun_case(text_list[idx], p)
                changed_text.append(res)
            else:
                changed_text.append(text_list[idx])
        return " ".join(changed_text)

    @classmethod
    def change_noun_case(cls, tok, pos) -> str:
        case_set = set(["nf", "þf", "þgf", "ef"])
        case = case_set.intersection(pos.variants)
        other_cases = case_set - case
        rand_case = other_cases.pop()
        bin_no_result = bin_islenska.lookup_variants(tok, cat="no", to_inflection=rand_case)

        if len(bin_no_result) > 0:
            # the lookup method adds hyphens to unrecognized compounds; we're
            # removing those (except for in truly hyphenated words)
            if "-" not in tok:
                return bin_no_result[0].bmynd.replace("-", "")
        return tok

    @classmethod
    def random_apply(cls, data) -> bool:
        # lowering the application ratio
        if random.random() < cls.CUSTOM_RATIO:
            return super().random_apply(data)
        return False
