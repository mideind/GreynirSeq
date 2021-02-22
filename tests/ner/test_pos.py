from greynirseq.settings import IceBERT_POS_PATH, IceBERT_POS_CONFIG
from greynirseq.nicenlp.models.multilabel import MutliLabelRobertaModel
from greynirseq.ner import aligner


def test_pos_tagging(ner_tagged_sentences_en, ner_tagged_sentences_is, is_pos_tags):
    parser = aligner.NERParser(ner_tagged_sentences_en, ner_tagged_sentences_is)
    pos_model = MutliLabelRobertaModel.from_pretrained(IceBERT_POS_PATH, **IceBERT_POS_CONFIG)
    pos_model.to("cuda")
    pos_model.eval()
    for idx, (p1, p2, pair_info) in enumerate(parser.parse_files_gen(None)):
        pos_tags = pos_model.predict_to_idf(p2.sent, device="cuda")  # type: ignore
        assert is_pos_tags[idx] == pos_tags
