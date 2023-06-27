import random

from .errors import ErrorRule


class SwapErrorRule(ErrorRule):
    """Error rule class that randomly swaps adjacent words in a sentence, avoiding the first and last tokens."""

    needs_pos = False
    CUSTOM_RATIO = 0.1

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
        # not swapping in very short sentences
        if len(text.split()) <= 3:
            return False
        # lowering the application ratio
        if random.random() < cls.CUSTOM_RATIO:
            return super().random_apply(data)
        return False
