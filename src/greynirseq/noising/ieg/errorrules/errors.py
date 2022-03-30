import random
from ieg.spelling.errorify import errorify_line

from ieg import b, g


class ErrorRule:
    """ Base class for all ErrorRules to inherit.
    """

    needs_pos = False

    @classmethod
    def random_apply(cls, data):
        if random.random() < data["args"].rule_chance_error_rate:
            return True
        return False

    @staticmethod
    def _apply(data):
        return data["text"]

    @classmethod
    def apply(cls, data):
        if cls.random_apply(data):
            return cls._apply(data)
        return data["text"]


class DativeSicknessErrorRule(ErrorRule):
    """
    """

    @staticmethod
    def _apply(data):
        text = data["text"]
        try:
            s = g.parse_single(text)
            return " ".join(n.dative for n in s.tree.descendants if n.is_terminal)
        except:
            # Sentence does not parse
            return data["text"]


class NoiseErrorRule(ErrorRule):
    """
    """

    @staticmethod
    def _apply(data):
        text = errorify_line(data["text"], word_error_rate=data["args"].word_spelling_error_rate)
        return text


class SwapErrorRule(ErrorRule):
    """
    """

    @staticmethod
    def _apply(data):
        text = data["text"]
        sent_tokens = text.split()
        # Don't want to swap at the first position and -3 is the last item before the period
        first = random.randint(1, len(sent_tokens) - 3) 
        second = first + 1
        sent_tokens[first], sent_tokens[second] = sent_tokens[second], sent_tokens[first]
        return " ".join(sent_tokens)
        
    @classmethod
    def random_apply(cls, data):
        text = data["text"]
        if len(text.split()) <= 3:
            return False
        return super().random_apply(data)


class MoodErrorRule(ErrorRule):
    """
    """
    needs_pos = True

    @classmethod
    def _apply(cls, data):
        text, pos = data["text"], data["pos"]
        changed_text = []
        for tok, pos in zip(text.split(), pos):
            if pos.category == "so" and "vh" in pos.variants:
                changed_text.append(cls.change_mood(tok, pos))
            else:
                changed_text.append(tok)
        return " ".join(changed_text)
    
    @classmethod
    def change_mood(cls, tok, pos):
        try:
            print(b.lookup_variants(tok, "so", ("FH"))[0].bmynd)
            return b.lookup_variants(tok, "so", ("FH"))[0].bmynd
        except:
            return tok