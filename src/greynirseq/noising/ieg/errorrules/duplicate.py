import random

from .errors import ErrorRule


class DuplicateWordsRule(ErrorRule):
    """Error rule class for duplicating a random word in a sentence so that it appears twice in succession"""

    needs_pos = True

    @classmethod
    def _apply(cls, data):
        text, pos = data["text"], data["pos"]
        text_list = text.split()
        if not super().tok_and_pos_match(cls, text_list, pos):
            return text
        rand_int = random.randint(0, len(text_list) - 1)
        if not text_list[rand_int].isdigit():
            text_list[rand_int] = text_list[rand_int] + " " + text_list[rand_int]
        return " ".join(text_list)
