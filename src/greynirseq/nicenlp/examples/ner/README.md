# NER tagging with IceBERT

This example shows how to train an Icelandic NER tagger with F1 score 0.9274 on the [MIM-GOLD-NER](https://repository.clarin.is/repository/xmlui/handle/20.500.12537/42) named entity recognition corpus. The output format is the same as used there.

## Inference

### Using the CLI

Using the CLI is the easiest way of using the tagger, this downloads the necessary files, make sure you have space for around 1GB of data.

``` bash
$ pip install greynirseq
$ echo "Systurnar Guðrún og Monique átu einar um jólin á McDonalds ." | greynirseq ner --input -

O B-Person O B-Person O O O O O B-Organization O
```

It takes a while to load the model so if you need to tag many lines you should provide them all at once.

### From torch hub

This will download the model from our servers and return an instance for inference

```python
import torch
model = torch.hub.load("mideind/GreynirSeq:main", "icebert-ner")
model.eval()
labels = list(model.predict_labels(["Systurnar Guðrún og Monique átu einar um jólin á McDonalds ."])
```
which returns the labels `['O', 'B-Person', 'O', 'B-Person', 'O', 'O', 'O', 'O', 'O', 'B-Organization', 'O']`.

### Local inference

Point the model class to the checkpoint (any of the splits or an averaged checkpoint) and auxiliary data as e.g.

```python
from greynirseq.nicenlp.models.multiclass import MultiClassRobertaMode
from greynirseq.settings import IceBERT_NER_CONFIG, IceBERT_NER_PATH

model = MultiClassRobertaModel.from_pretrained(IceBERT_NER_PATH, **IceBERT_NER_CONFIG)
model.eval()
```

Note that the length of the sentences has a ceiling set by the used model and direct inference may crash on long sentences. To run the models on GPU simply run `model.to("cuda")`, we refer to the pytorch documentation for further details. When running on GPU we recommend using the argument `batch_size` with `predict_labels` to speed up inference.


## Training
 
### Preprocessing

The raw data needs to be mapped from word per line to sentence per line, see `greynirseq.utils.ifd_utils.ifd2labels`. Then bpe encoded and preprocessed. See the README for POS tagging.

### Training

See `./train.sh` for the script used.


