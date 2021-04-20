# POS tagging with IceBERT

This example shows how to train an Icelandic POS tagger with ~98% accuracy on the [Tagged Icelandic Corpus](http://www.malfong.is/index.php?lang=en&pg=mim) (MIM) dataset.

## Preprocessing
See `./prep_mim_pos.sh` which is setup to process all data from the MIM pos set and prepare for crossvalidation.

## Training
See `./train.sh` which trains all ten sets for crossvalidation.

## Inference

### Using torch hub

This will download the model from our servers and return an instance for inference

```python
import torch
model = torch.hub.load("mideind/GreynirSeq", "icebert.pos")
model.eval()
labels = model.predict_labels(["Systurnar Guðrún og Monique átu einar um jólin á McDonalds ."])
```

which returns

```python
[[('n', ['fem', 'plur', 'nom', 'definite']),
  ('n', ['fem', 'sing', 'nom', 'proper']),
  ('c', []),
  ('ns', []),
  ('sf', ['plur', 'act', '3', 'past']),
  ('l', ['fem', 'plur', 'nom', 'strong', 'pos']),
  ('af', ['pos']),
  ('n', ['neut', 'plur', 'acc', 'definite']),
  ('af', ['pos']),
  ('ns', []),
  ('pl', [])]]
```

### Local inference

Point the model class to the checkpoint (any of the splits or an averaged checkpoint) and auxiliary data as e.g.

```python
from greynirseq.nicenlp.models.multilabel import MultiLabelRobertaMode
from greynirseq.settings import IceBERT_POS_PATH, IceBERT_POS_CONFIG
model = MultiLabelRobertaModel.from_pretrained(IceBERT_POS_PATH, **IceBERT_POS_CONFIG)

sentence = "Ég veit að þú kemur í kvöld til mín ."
model.predict_labels([sentence])
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

