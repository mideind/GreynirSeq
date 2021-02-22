from greynirseq.ner import postagger
from greynirseq.ner import aligner


def test_add_mark(ner_tagged_sentences_en, ner_tagged_sentences_is, ner_final_simple):
    parser = aligner.NERParser(ner_tagged_sentences_en, ner_tagged_sentences_is)
    for i, (en_parse, is_parse, pair_info) in enumerate(parser.parse_files_gen(None)):
        en_tokens = en_parse.sent.split()
        is_tokens = is_parse.sent.split()
        for idx, alignment in enumerate(pair_info.pair_map):
            en_ner_marker, is_ner_marker, distance = alignment.marker_1, alignment.marker_2, alignment.cost
            # The distance should be small
            assert distance < 0.3
            postagger.add_marker(en_ner_marker, en_tokens, idx, "x")
            postagger.add_marker(is_ner_marker, is_tokens, idx, "x")
        assert " ".join(en_tokens) == ner_final_simple[i][0]
        assert " ".join(is_tokens) == ner_final_simple[i][1]
