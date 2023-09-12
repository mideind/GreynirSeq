import random

from .errors import ErrorRule


class DropCommaRule(ErrorRule):
    """Error rule class for dropping all commas in a sentence."""

    needs_pos = False
    CUSTOM_RATIO = 0.1

    @classmethod
    def random_apply(cls, data) -> bool:
        text = data["text"]
        if "," not in text.split():
            return False
        # lowering the application ratio
        if random.random() < cls.CUSTOM_RATIO:
            return super().random_apply(data)
        return False

    @classmethod
    def _apply(cls, data) -> str:
        text, _ = data["text"], data["pos"]
        text_list = text.split()
        text_list = list(filter(lambda a: a != ",", text_list))
        return " ".join(text_list)
