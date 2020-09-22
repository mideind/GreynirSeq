import os

from greynirseq.ner.filter import ParallelNER
from greynirseq.nicenlp.models.icebert import IcebertModel
from greynirseq.settings import MODEL_DIR, DATASET_DIR


pos_model = IcebertModel.from_pretrained(
    os.path.join(MODEL_DIR, 'icebert_pos'),
    checkpoint_file='checkpoint_last.pt',
    gpt2_encoder_json=os.path.join(MODEL_DIR, 'icebert-base-36k/icebert-bpe-vocab.json'),  
    gpt2_vocab_bpe=os.path.join(MODEL_DIR, 'icebert-base-36k/icebert-bpe-merges.txt'),
)

# TODO accept string literal etc.
def tag_ner_pair(pos_model, p1, p2, pair_info):
    hit = False
    en_sent = p1[0].split()
    is_sent = p2[0].split()

    # Assume p2 is Icelandic
    pos_tags = pos_model.predict_to_idf(p2[0], device="cpu")
    for pair in pair_info['pair_map']:
        is_loc = pair[1][:2]
        en_loc = pair[0][:2]
        tags = pos_tags[is_loc[0]: is_loc[1]]
        if 'e' in tags:
            # Since IDF for some reason uses "e" for foreign names, we ignore those
            continue
        hit = True
        for i in range(en_loc[0], en_loc[1]):
            en_sent[i] = "<p-{}>".format(en_sent[i])
        for i in range(is_loc[0], is_loc[1]):
            is_sent[i] = "<p-{}-{}>".format(pos_tags[i], is_sent[i])
    if hit:
        return en_sent, is_sent
    return None, None  

def main():
    eval_ner = ParallelNER(
        os.path.join(DATASET_DIR, 'parice_ner_mideind_set/en-is.train.en.huggingface.spacy.ner'),
        os.path.join(DATASET_DIR, 'parice_ner_mideind_set/en-is.train.is.ner')
    )
    for p1, p2, pair_info in eval_ner.parse_files():
        if pair_info['pair_map']:
            en_sent, is_sent = tag_ner_pair(pos_model, p1, p2, pair_info)
            if en_sent is not None:
                print("{}\t{}\n".format(" ".join(en_sent), " ".join(is_sent)))

if __name__ == "__main__":
    main()