from .errors import ErrorRule

from ieg import b


class NounCaseErrorRule(ErrorRule):
    needs_pos = True

    @classmethod
    def _apply(cls, data):
        text, pos = data["text"], data["pos"]
        changed_text = []
        for tok, t_pos in zip(text.split(), pos):
            if t_pos.category == "no":
                changed_text.append(cls.change_noun_case(tok, t_pos))
            else:
                changed_text.append(tok)
        return " ".join(changed_text)

    @classmethod
    def change_noun_case(cls, tok, pos):
        if "ef" in pos.variants or "þf" in pos.variants:
            bin_no_result = b.lookup_variants(tok, cat="no", to_inflection="þgf")
            if len(bin_no_result) > 0:
                return bin_no_result[0].bmynd
        elif "þgf" in pos.variants:
            bin_no_result = b.lookup_variants(tok, cat="no", to_inflection="þf")
            if len(bin_no_result) > 0:
                return bin_no_result[0].bmynd
        return tok