import random

from torch import randint

from ieg import b
from ieg.spelling.errorify import errorify_line


class ErrorRule:
    """Base class for all ErrorRules to inherit."""

    needs_pos = False

    @classmethod
    def random_apply(cls, data) -> bool:
        if random.random() < data["args"].rule_chance_error_rate:
            return True
        return False

    @staticmethod
    def _apply(data) -> str:
        return data["text"]

    @classmethod
    def apply(cls, data) -> str:
        if cls.random_apply(data):
            return cls._apply(data)
        return data["text"]


class DativitisErrorRule(ErrorRule):
    """Error rule class for applying the dative to nominative or accusative
    subjects, mig vantar -> mér vantar) - the so called "þágufallshneigð".
    """

    @staticmethod
    def _apply(data) -> str:
        try:
            s_tree = data["tree"]
            tok_list = s_tree.text.split()
            if ip := s_tree.all_matches("IP >> { ('langa'|'vanta'|'dreyma') }"):
                for i in ip:
                    np = i.first_match("NP")
                    vp = i.first_match("VP")
                    suggest = np.dative_np
                    if vp is None or np is None:
                        continue
                    so = vp.first_match("so_subj")
                    if so is None:
                        continue
                    variants = set(np.all_variants) - {"þf", np.cat}
                    variants.add("þgf")

                    start, end = np.span[0], np.span[1] + 1

                    tok_list[start:end] = [suggest]
                return " ".join(tok_list)
        except Exception:
            # Sentence does not parse
            return data["text"]


class NoiseErrorRule(ErrorRule):
    """Error rule class that scrambles the spelling of words according to predefined rules.
    Also applies word substitution from a list of common errors (to be abstracted out).
    """

    @staticmethod
    def _apply(data) -> str:
        text = errorify_line(data["text"], word_error_rate=data["args"].word_spelling_error_rate)
        return text


class SwapErrorRule(ErrorRule):
    """Error rule class that randomly swaps adjacent words in a sentence, avoiding the first
    word and last tokens.
    """

    @staticmethod
    def _apply(data) -> str:
        text = data["text"]
        sent_tokens = text.split()
        # Don't want to swap at the first position and -3 is the last item before the period
        first = random.randint(1, len(sent_tokens) - 3)
        second = first + 1
        sent_tokens[first], sent_tokens[second] = sent_tokens[second], sent_tokens[first]
        return " ".join(sent_tokens)

    @classmethod
    def random_apply(cls, data) -> bool:
        text = data["text"]
        if len(text.split()) <= 3:
            return False
        return super().random_apply(data)


class MoodErrorRule(ErrorRule):
    """Error rule class for changing the mood of verbs between the infinitive and the subjunctive."""

    needs_pos = True

    @classmethod
    def _apply(cls, data):
        text, pos, tree = data["text"], data["pos"], data["tree"]
        changed_text = []
        for p in pos:
            if p.category == "so" and "vh" in p.variants:
                changed_text.append(cls.change_mood(p.text))
            else:
                changed_text.append(p.text)
        return " ".join(changed_text)

    @classmethod
    def change_mood(cls, tok):
        try:
            return b.lookup_variants(tok, "so", ("FH"))[0].bmynd
        except:
            return tok


class DuplicateWordsRule(ErrorRule):
    """Error rule class for duplicating a random word in a sentence so that it appears twice in succession"""
    @staticmethod
    def _apply(data):
        try:
            sent_tokens = [p.text for p in data["pos"]]
            rand_int = random.randint(0, len(sent_tokens))
            if not any(sent_tokens[rand_int]).isdigit():
                sent_tokens[rand_int] = sent_tokens[rand_int] + " " + sent_tokens[rand_int]
            return " ".join(sent_tokens)
        except:
            return data["text"]

