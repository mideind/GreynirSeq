from greynirseq.ner.aligner import NERParser


def test_parse(is_ner, en_ner):
    p = NERParser(is_ner, en_ner)
    parse = p.parse_files_gen(None)
