# POS tagging with IceBERT

This example shows how to train an Icelandic POS tagger with ~98.2% accuracy on the [Tagged Icelandic Corpus](http://www.malfong.is/index.php?lang=en&pg=mim) (MIM) dataset.

The output can be configure to use the MIM 2.0 label format which is the default for the CLI. Note that the tags **e** (foreign) and **x** (unknown) tag are excluded from training and labeling since the tagger has no particular problem with labeling those words.

## Preprocessing
See `./prep_mim_pos.sh` which is setup to process all data from the MIM pos set and prepare for cross-validation.

## Training
See `./train.sh` which trains all ten sets for cross-validation.

## Inference

### Using the CLI

Using the CLI is the easiest way of using the tagger, this downloads the necessary files, make sure you have space for around 1GB of data.

``` bash
$ pip install greynirseq
$ echo "Systurnar Guðrún og Monique átu einar um jólin á McDonalds ." | greynirseq pos --input -

nvfng nven-s c n---s sfg3fþ lvfnsf af nhfog af n----s pl
```

It takes a while to load the model so if you need to tag many lines you should provide them all at once.

### Using torch hub

This will download the model from our servers and return an instance for inference

```python
import torch
model = torch.hub.load("mideind/GreynirSeq:main", "icebert-pos")
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

or

``` python

labels = model.predict_ifd_labels(["Systurnar Guðrún og Monique átu einar um jólin á McDonalds ."])
```

which returns

``` python
[['nvfng',
  'nven-s',
  'c',
  'ns',
  'sfg3fþ',
  'lvfnsf',
  'af',
  'nhfog',
  'af',
  'ns',
  'pl']]

```

Note that the length of the sentences has a ceiling set by the used model and direct inference may crash on long sentences. To run the models on GPU simply run `model.to("cuda")`, we refer to the pytorch documentation for further details.

### Local inference

Point the model class to the checkpoint (any of the splits or an averaged checkpoint) and auxiliary data as e.g.

```python
from greynirseq.nicenlp.models.multilabel import MultiLabelRobertaModel
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

