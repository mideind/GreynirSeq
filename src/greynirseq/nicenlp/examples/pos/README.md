# POS tagging with IceBERT

This example shows how to train an Icelandic POS tagger with ~97% accuracy on the [Tagged Icelandic Corpus](http://www.malfong.is/index.php?lang=en&pg=mim) (MIM) dataset.

### Preprocessing
See `./prep_mim_pos.sh` which is setup to process all data from the MIM pos set and prepare for crossvalidation.

### Training
See `./train.sh` which trains all ten sets for crossvalidation.

### Simple inference

Point the model class to the checkpoint (any of the splits or an averaged checkpoint) and auxiliary data as e.g.

```python
from greynirseq.nicenlp.models.icebert import IcebertModel
ib = IcebertModel.from_pretrained(
    '/data/models/icebert_pos',
    checkpoint_file='checkpoint_last.pt',
    gpt2_encoder_json="/data/models/icebert-base-36k/icebert-bpe-vocab.json",  
    gpt2_vocab_bpe="/data/models/icebert-base-36k/icebert-bpe-merges.txt",
)
ib.predict_to_idf("Ég veit að þú kemur í kvöld til mín.", device="cpu")
```

which returns `['fp1en', 'sfg1en', 'c', 'fp2en', 'sfg2en', 'ao', 'nheo', 'ae', 'fphee', 'p']`.

