from greynirseq.ner.patcher import decline_np, idf2kasus, parse_sentence, parse_sentence_pair


def test_parse_sentence(ner_sentence_pair):
    sent_is, _ = ner_sentence_pair
    parse = parse_sentence(sent_is)
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


def test_parse_sentence_pair(ner_sentence_pair):
    sent_1, sent_2 = parse_sentence_pair(*ner_sentence_pair)
    ners_1 = [ner for ner in sent_1 if ner["ner"]]
    ners_2 = [ner for ner in sent_2 if ner["ner"]]
    assert ners_1[0]["oidx"] == 2
    assert ners_2[0]["oidx"] == 3


def test_idf2casus():
    gender, case = idf2kasus("nvxo")
    assert gender == "v"
    assert case == "o"


def test_decline_np():
    decl = decline_np("Guðrúnu Helgadóttur", "n")
    assert decl == "Guðrún Helgadóttir"


def test_fill_ner_gap():
    pass
