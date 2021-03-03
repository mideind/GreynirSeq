# POS tagging with IceBERT

This example shows how to train an Icelandic POS tagger with ~97% accuracy on the [Tagged Icelandic Corpus](http://www.malfong.is/index.php?lang=en&pg=mim) (MIM) dataset.

### Preprocessing
See `./prep_mim_pos.sh` which is setup to process all data from the MIM pos set and prepare for crossvalidation.

### Training
See `./train.sh` which trains all ten sets for crossvalidation.

### Simple inference

Point the model class to the checkpoint (any of the splits or an averaged checkpoint) and auxiliary data as e.g.

```python
from greynirseq.nicenlp.models.multilabel import MultiLabelRobertaMode
from greynirseq.settings import IceBERT_POS_PATH, IceBERT_POS_CONFIG
model = MultiLabelRobertaModel.from_pretrained(IceBERT_POS_PATH, **IceBERT_POS_CONFIG)

sentence = "Ég veit að þú kemur í kvöld til mín ."
model.predict_labels(sentence)
```

which returns the following

```python
[[('fp', ['1', 'sing', 'nom']),
  ('sf', ['sing', 'act', '1', 'pres']),
  ('c', []),
  ('fp', ['2', 'sing', 'nom']),
  ('sf', ['sing', 'act', '2', 'pres']),
  ('af', ['pos']),
  ('n', ['neut', 'sing', 'acc']),
  ('af', ['pos']),
  ('fp', ['1', 'sing', 'gen']),
  ('pl', [])]]
```

