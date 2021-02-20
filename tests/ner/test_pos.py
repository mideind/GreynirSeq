from greynirseq.settings import IceBERT_POS_PATH, IceBERT_POS_CONFIG
from greynirseq.nicenlp.models.icebert import IcebertModel
from greynirseq.ner import aligner, postagger


def test_pos_tagging(ner_tagged_sentences_en, ner_tagged_sentences_is):
    parser = aligner.NERParser(ner_tagged_sentences_en, ner_tagged_sentences_is)
    pos_model = IcebertModel.from_pretrained(IceBERT_POS_PATH, **IceBERT_POS_CONFIG)
    pos_model.to("cuda")
    pos_model.eval()
    for p1, p2, pair_info in parser.parse_files_gen(None):
        pos_tags = postagger.tag_ner_pair(pos_model, p1, p2, pair_info, max_distance=0.9)
        print(pos_tags)
