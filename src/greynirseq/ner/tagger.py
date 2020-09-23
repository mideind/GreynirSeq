import os
import re

from greynirseq.ner.filter import ParallelNER
from greynirseq.nicenlp.models.icebert import IcebertModel
from greynirseq.settings import MODEL_DIR, DATASET_DIR

from reynir import NounPhrase


# TODO accept string literal etc.
def tag_ner_pair(pos_model, p1, p2, pair_info, max_distance=1):
    hit = False
    en_sent = p1[0].split()
    is_sent = p2[0].split()

    # Assume p2 is Icelandic
    pos_tags = pos_model.predict_to_idf(p2[0], device="cuda")
    for idx, pair in enumerate(pair_info['pair_map']):
        is_loc = pair[1][:2]
        en_loc = pair[0][:2]
        tags = pos_tags[is_loc[0]: is_loc[1]]
        if 'e' in tags:
            # Since IDF for some reason uses "e" for foreign names, we ignore those
            continue
        if pair[2] > max_distance:
            continue

        hit = True
        en_sent[en_loc[0]] = '<e:{}:{}:>{}'.format(idx, tags[0], en_sent[en_loc[0]])
        if en_loc[1] - en_loc[0] != 1:
            en_sent[en_loc[1] - 1] = '{}</e{}>'.format(en_sent[en_loc[1]-1], idx)
        else:
            en_sent[en_loc[0]] += '</e{}>'.format(idx)

        is_sent[is_loc[0]] = '<e:{}:{}:>{}'.format(idx, tags[0], is_sent[is_loc[0]])
        if is_loc[1] - is_loc[0] != 1:
            is_sent[is_loc[1] - 1] = '{}</e{}>'.format(is_sent[is_loc[1]-1], idx)
        else:
            is_sent[is_loc[0]] += '</e{}>'.format(idx)

    if hit:
        return en_sent, is_sent
    return None, None


def main():

    pos_model = IcebertModel.from_pretrained(
        os.path.join(MODEL_DIR, 'icebert_pos'),
        checkpoint_file='checkpoint_last.pt',
        gpt2_encoder_json=os.path.join(MODEL_DIR, 'icebert-base-36k/icebert-bpe-vocab.json'),  
        gpt2_vocab_bpe=os.path.join(MODEL_DIR, 'icebert-base-36k/icebert-bpe-merges.txt'),
    )
    pos_model.to('cuda')
    pos_model.eval()

    eval_ner = ParallelNER(
        os.path.join(DATASET_DIR, 'parice_ner_mideind_set/en-is.train.en.huggingface.spacy.ner'),
        os.path.join(DATASET_DIR, 'parice_ner_mideind_set/en-is.train.is.ner')
    )
    with open('/tmp/pos_patch_ner.tsv', 'w') as ofile:
        for p1, p2, pair_info in eval_ner.parse_files():
            if pair_info['pair_map']:
                en_sent, is_sent = tag_ner_pair(pos_model, p1, p2, pair_info, max_distance=0.9)
                if en_sent is not None:
                    ofile.writelines("{}\t{}\n".format(" ".join(en_sent), " ".join(is_sent)))

if __name__ == "__main__":
    main()