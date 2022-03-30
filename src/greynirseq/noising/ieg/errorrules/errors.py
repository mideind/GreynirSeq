import random
from islenska import Bin
from ieg.spelling.errorify import errorify_line
from reynir import Greynir

b = Bin()
g = Greynir()


class ErrorRule:
    APPLICATION_PROBABILITY = 0.9

    @classmethod
    def random_apply(cls, *args):
        if random.random() < cls.APPLICATION_PROBABILITY:
            return True
        return False

    @staticmethod
    def _apply(data):
        return data["text"]

    @classmethod
    def apply(cls, data):
        if cls.random_apply(cls, *data):
            return cls._apply(data)
        return data["text"]


class DativeSicknessErrorRule(ErrorRule):

    @staticmethod
    def _apply(data):
        text, pos = data.values()
        try:
            s = g.parse_single(text)
            return " ".join(n.dative for n in s.tree.descendants if n.is_terminal)
        except:
            pass 

    @classmethod
    def random_apply(cls, *args):

        return super().random_apply()


class NoiseErrorRule(ErrorRule):
    needs_pos = False

    @staticmethod
    def _apply(data):
        return errorify_line(data["text"])

    def random_apply(cls, *args):
        #text, pos = data
        return super().random_apply()


class SwapErrorRule(ErrorRule):
    needs_pos = False

    @staticmethod
    def _apply(data):
        text = data["text"]
        sent_tokens=text.split()
        # Don't want to swap at the first position and -3 is the last item before the period
        first = random.randint(1, len(sent_tokens)-3) 
        second = first + 1
        sent_tokens[first], sent_tokens[second] = sent_tokens[second], sent_tokens[first]
        return " ".join(sent_tokens)
        
    @classmethod
    def random_apply(cls, data):
        print(data)

        text, pos = data
        if len(text.split()) <= 3:
            return False
        return super().random_apply()


class AnnadHvortErrorRule(ErrorRule):
    @classmethod
    def _apply(cls, data):
        text, pos = data.values()
        return text.replace("annaðhvort", "annað hvort")

    @classmethod
    def random_apply(cls, *data):
        if "annaðhvort" in data["text"]:
            return super().random_apply()


class NounCaseErrorRule(ErrorRule):

    @classmethod
    def _apply(cls, data):
        text, pos = data.values()
        changed_text = []
        for tok, pos in zip(text.split(), pos.split()):
            if pos[0] == "n":
                changed_text.append(cls.change_noun_case(tok, pos).upper())
            else:
                changed_text.append(tok)
        return " ".join(changed_text)
    
    @classmethod
    def random_apply(cls, *args):
        return super().random_apply()

    @classmethod
    def change_noun_case(cls, tok, pos):
        if pos[3] == "e" or pos[3] == "o":
            bin_no_result = b.lookup_variants(tok, cat="no", to_inflection="þgf")
            if len(bin_no_result) > 0:
                return bin_no_result[0].bmynd
        elif pos[3] == "þ":
            bin_no_result = b.lookup_variants(tok, cat="no", to_inflection="þf")
            if len(bin_no_result) > 0:
                return bin_no_result[0].bmynd
        return tok


class MoodErrorRule(ErrorRule):
    needs_pos = True

    @classmethod
    def _apply(cls, data):
        text, pos = data.values()
        changed_text = []
        for tok, pos in zip(text.split(), pos.split()):
            if pos[0:2] == "sv":
                changed_text.append(cls.change_mood(tok, pos))
            else:
                changed_text.append(tok)
        return " ".join(changed_text)
    
    @classmethod
    def random_apply(cls, *data):
        return super().random_apply()

    @classmethod
    def change_mood(cls, tok, pos):
        try:
            print(b.lookup_variants(tok, "so", ("FH"))[0].bmynd)
            return b.lookup_variants(tok, "so", ("FH"))[0].bmynd
        except:
            return tok