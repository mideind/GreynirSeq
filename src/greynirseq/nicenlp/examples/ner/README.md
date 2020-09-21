# NER tagging with IceBERT

This example shows how to train an Icelandic NER tagger with F1 score 0.918 on the [MIM-GOLD-NER](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/42) named entity recognition corpus.

### Preprocessing

The raw data needs to be mapped from word per line to sentence per line, see `greynirseq.utils.ifd_utils.ifd2labels`. Then bpe encoded and preprocessed. See the README for POS tagging.

### Training

See `./ner_training.sh` for the script used.

### Simple inference

Point the model class to the checkpoint (any of the splits or an averaged checkpoint) and auxiliary data as e.g.

```python
from greynirseq.nicenlp.models.multi_span_model import IcebertConstModel

model = IcebertConstModel.from_pretrained(
    '/media/hd/MIDEIND/data/models/icebert_ner/ner_slset',
    checkpoint_file="checkpoint_last.pt",
    data_name_or_path='/media/hd/MIDEIND/data/models/MIM-GOLD-NER_split/8_entity_types/bin/bin',
    gpt2_encoder_json="/media/hd/MIDEIND/data/models/icebert-base-36k/icebert-bpe-vocab.json",
    gpt2_vocab_bpe="/media/hd/MIDEIND/data/models/icebert-base-36k/icebert-bpe-merges.txt",
    term_schema="/media/hd/MIDEIND/data/models/MIM-GOLD-NER_split/term.json"
)
model.eval()

cat_idx, labels, sentence = model.predict_pos(["Systurnar Guðrún og Monique átu einar um jólin á McDonalds."], device="cpu")
```

which returns the labels `['O', 'B-Person', 'O', 'B-Person', 'O', 'O', 'O', 'O', 'O', 'B-Organization', 'O']`.

