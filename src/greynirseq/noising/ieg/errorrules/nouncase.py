from ieg import b

from .errors import ErrorRule


class NounCaseErrorRule(ErrorRule):
    """Error rule class that inflects nouns in a sentence to another case at random.
    """
    needs_pos = True

    @classmethod
    def _apply(cls, data):
        text, pos = data["text"], data["pos"]
        tok_list = text.split()
        changed_text = []
        for tok, t_pos in zip(tok_list, pos):
            if type(t_pos) == str:
                if pos[0] == "n":
                    changed_text.append(cls.change_noun_case(tok, t_pos))
            elif t_pos.category == "no":
                    changed_text.append(cls.change_noun_case(tok, t_pos))
            else:
                changed_text.append(tok)
        if len(tok_list) > len(pos):
            print(tok_list[-1])
            # no category for the punctuation token
            changed_text.append(tok_list[-1])
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
