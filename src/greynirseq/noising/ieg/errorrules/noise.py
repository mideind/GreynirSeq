from ieg.spelling.errorify import errorify_line

from .errors import ErrorRule


class NoiseErrorRule(ErrorRule):
    """Error rule class that scrambles the spelling of words according to predefined rules.
    Also applies word substitution from a list of common errors (to be abstracted out).
    """

    needs_pos = False

    @staticmethod
    def _apply(data) -> str:
        text = errorify_line(data["text"], word_error_rate=data["args"].word_spelling_error_rate)
        return text
