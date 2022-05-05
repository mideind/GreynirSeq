import random

from ieg import bin_islenska


class ErrorRule:
    """Base class for all ErrorRules to inherit."""

    needs_pos = False

    @classmethod
    def random_apply(cls, data) -> bool:
        if random.random() < data["args"].rule_chance_error_rate:
            return True
        return False

    @staticmethod
    def tok_and_pos_match(cls, tok, pos) -> bool:
        return len(tok) == len(pos)

    @classmethod
    def get_wordform(cls, word, lemma, cat, variants) -> str:
        """Get correct wordform from BinPackage, given a set of variants"""
        wordforms = bin_islenska.lookup_variants(word, cat, tuple(variants), lemma=lemma)
        if not wordforms:
            return ""
        # Can be many possible word forms; we want the first one in most cases
        return wordforms[0].bmynd

    @staticmethod
    def _apply(data) -> str:
        return data["text"]

    @classmethod
    def apply(cls, data) -> str:
        if cls.random_apply(data):
            return cls._apply(data)
        return data["text"]
