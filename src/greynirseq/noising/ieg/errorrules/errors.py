import random

from ieg import bin_islenska
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


class DativitisErrorRule(ErrorRule):
    """Error rule class for applying the dative to nominative or accusative
    subjects, mig vantar -> mér vantar) - the so called "þágufallshneigð". Needs to be the first rule to be applied.
    """

    @classmethod
    def _apply(cls, data) -> str:
        sentence = data["text"]
        try:
            s_tree = data["tree"]
            tok_list = sentence.split()

            if kh_ip := s_tree.all_matches("IP >> { ('hlakka') }"):  # TODO: kvíða
                sentence = cls.nom_to_acc_or_dat(tok_list, kh_ip)
            if lvd_ip := s_tree.all_matches("IP >> { ('langa'|'vanta'|'dreyma') }"):
                sentence = cls.acc_to_dat(tok_list, lvd_ip)
        except Exception:
            # Sentence does not parse
            return sentence
        return sentence

    @classmethod
    def acc_to_dat(cls, tok_list, inflectional_phrase) -> str:
        for i in inflectional_phrase:
            np = i.first_match("NP")
            vp = i.first_match("VP")
            suggest = np.dative_np
            if vp is None or np is None:
                continue
            so = vp.first_match("so_subj")
            if so is None:
                continue
            start, end = np.span[0], np.span[1] + 1
            tok_list[start:end] = [suggest]
        return " ".join(tok_list)

    @classmethod
    def nom_to_acc_or_dat(cls, tok_list, ip) -> str:
        for i in ip:
            np = i.first_match("NP")
            vp = i.first_match("VP")
            np_suggest = np.dative_np
            verb = None
            if vp is None or np is None:
                continue
            for v in vp.all_matches("so"):
                if v.lemma == "hlakka":  # or v.lemma == "kvíði":
                    verb = v

            if verb is None:
                continue
            verb_text = verb.text
            # using the third person singular to produce þágufallshneigð
            verb_variants = set(verb.variants) - {"ft", "p1", "p2"}
            verb_variants.add("subj")
            verb_variants.add("p3")
            verb_variants.add("et")

            verb_suggest = super().get_wordform(verb_text, verb.lemma, verb.cat, verb_variants)
            np_start, np_end = np.span[0], np.span[1] + 1
            verb_start, verb_end = verb.span[0], verb.span[1]
            tok_list.pop(verb_start)
            tok_list[verb_start:verb_end] = [verb_suggest]
            tok_list[np_start:np_end] = [np_suggest]

        return " ".join(tok_list)


class NoiseErrorRule(ErrorRule):
    """Error rule class that scrambles the spelling of words according to predefined rules.
    Also applies word substitution from a list of common errors (to be abstracted out).
    """

    @staticmethod
    def _apply(data) -> str:
        text = errorify_line(data["text"], word_error_rate=data["args"].word_spelling_error_rate)
        return text


class SwapErrorRule(ErrorRule):
    """Error rule class that randomly swaps adjacent words in a sentence, avoiding the first and last tokens."""

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


class DuplicateWordsRule(ErrorRule):
    """Error rule class for duplicating a random word in a sentence so that it appears twice in succession"""

    needs_pos = True

    @staticmethod
    def _apply(data):
        text, pos = data["text"], data["pos"]
        text_list = text.split()
        if len(text_list) != len(pos):
            return text
        rand_int = random.randint(0, len(text_list) - 1)
        if not text_list[rand_int].isdigit():
            text_list[rand_int] = text_list[rand_int] + " " + text_list[rand_int]
        return " ".join(text_list)


class SplitWordsRule(ErrorRule):
    """Error rule class for adding spaces to words. For now only adds spaces to
    compounds not found in BÍN, such as "rauð doppótta", "botn hreinsið" and "fjalla hryggir"."""

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
                    sent_tokens[idx] = m[0].bmynd.replace("-", " ").upper()
            return " ".join(sent_tokens)

        return data["text"]


class DeleteSpaceErrorRule(ErrorRule):
    """Error rule class that randomly removes the space between words. Best if applied last."""

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
