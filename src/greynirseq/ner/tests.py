import unittest
from greynirseq.ner.patcher import (
    parse_sentence,
    idf2casus,
    decline_np,
    parse_sentence_pair
)


class NERPatchingTests(unittest.TestCase):
    sent_1 = "Um <e:0:nvxo:>Guðrúnu Helgadóttur</e0> hefur <e:1:nkxn:>Einar</e1> ort ."
    sent_2 = "<e:1:nkxn:>Einar</e1> has written about <e:0:nvxo:>Guðrún Helgadóttir</e0> ."

    def test_parse_sentence(self):
        parse = parse_sentence(self.sent_1)
        text = "".join([a["text"] for a in parse])
        ners = [ner for ner in parse if ner["ner"]]
        assert text == "Um Guðrúnu Helgadóttur hefur Einar ort ."
        assert len(parse) == 5
        assert len(ners) == 2
        assert ners[0]["text"] == "Guðrúnu Helgadóttur"
        assert ners[1]["text"] == "Einar"
        assert ners[0]["ner"] == "0"
        assert ners[1]["ner"] == "1"
        assert ners[0]["pos"] == "nvxo"
        assert ners[1]["pos"] == "nkxn"

    def test_parse_sentence_pair(self):
        sent_1, sent_2 = parse_sentence_pair(self.sent_1, self.sent_2)
        ners_1 = [ner for ner in sent_1 if ner["ner"]]
        ners_2 = [ner for ner in sent_2 if ner["ner"]]
        assert ners_1[0]["oidx"] == 2
        assert ners_2[0]["oidx"] == 3

    def test_idf2casus(self):
        gender, case = idf2casus("nvxo")
        assert gender == "v"
        assert case == "o"

    def test_decline_np(self):
        decl = decline_np("Guðrúnu Helgadóttur", "n")
        assert decl == "Guðrún Helgadóttir"

    def test_fill_ner_gap(self):
        pass