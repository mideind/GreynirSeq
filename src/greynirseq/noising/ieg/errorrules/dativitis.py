from .errors import ErrorRule


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
