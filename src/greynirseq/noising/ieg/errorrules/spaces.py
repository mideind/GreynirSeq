import random

from ieg import bin_islenska

from .errors import ErrorRule


class SplitWordsRule(ErrorRule):
    """Error rule class for adding spaces to words. For now only adds spaces to
    compounds not found in BÍN, such as "rauð doppótta", "botn hreinsið" and "fjalla hryggir" (very rare)."""

    needs_pos = True

    @staticmethod
    def _apply(data) -> str:
        sent_tokens = data["text"].split()
        for idx, tok in enumerate(sent_tokens):
            _, m = bin_islenska.lookup(tok)
            if len(m) > 0:
                w_class = m[0][2]
                bin_id = m[0][1]
                # restricted to certain word classes to avoid introducing corrections
                # instead of errors in some phrases (utanað->utan að)
                if bin_id == 0 and w_class in ["no", "lo", "so"]:
                    sent_tokens[idx] = m[0].bmynd.replace("-", " ")
            return " ".join(sent_tokens)

        return data["text"]


class DeleteSpaceErrorRule(ErrorRule):
    """Error rule class that randomly removes the space between words. Best if applied last."""

    needs_pos = False

    @staticmethod
    def _apply(data) -> str:
        text = data["text"]
        sent_tokens = text.split()

        first = random.randint(0, len(sent_tokens) - 2)
        second = first + 1
        sent_tokens[first] = sent_tokens[first] + sent_tokens[second]
        sent_tokens.pop(second)

        return " ".join(sent_tokens)

    @classmethod
    def random_apply(cls, data) -> bool:
        text = data["text"]
        if len(text.split()) <= 2:
            return False
        return super().random_apply(data)
