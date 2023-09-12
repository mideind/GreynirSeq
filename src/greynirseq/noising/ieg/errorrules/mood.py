from ieg import bin_islenska

from .errors import ErrorRule


class MoodErrorRule(ErrorRule):
    """Error rule class for changing the mood of verbs between the infinitive and the subjunctive."""

    needs_pos = True

    @classmethod
    def _apply(cls, data) -> str:
        text, pos = data["text"], data["pos"]
        text_list = text.split()
        if not super().tok_and_pos_match(cls, text_list, pos):
            return text
        changed_text = []
        for idx, p in enumerate(pos):
            if p.category == "so" and "vh" in p.variants:
                changed_text.append(cls.change_mood(text_list[idx]))
            else:
                changed_text.append(text_list[idx])
        return " ".join(changed_text)

    @classmethod
    def change_mood(cls, tok) -> str:
        try:
            return bin_islenska.lookup_variants(tok, "so", ("FH"))[0].bmynd
        except Exception:
            return tok
