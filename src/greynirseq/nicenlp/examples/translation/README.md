# Translation

This example shows how to use trained translation models.

## Inference

### Using the CLI

Input is accepted from file containing a single **untokenized** sentence per line, or from stdin.

```bash
# For en->is translation
$ echo "This is an awesome test which shows how to use a pretrained translation model." | greynirseq translate --source-lang en --target-lang is

Þetta er frábært próf sem sýnir hvernig nota má forsniðið þýðingarlíkan.

# For is->en translation
$ echo "Þetta er frábært próf sem sýnir hvernig nota má forsniðið þýðingarlíkan." | greynirseq translate --source-lang is --target-lang en

This is a great test that shows how to use a formatted translation model.

```

A different translation model can be selected by supplying the `--model-name` option along with a valid name.

See `greynirseq -h` for available options.

### Using torch hub

This will download the model from our servers and return an instance for inference.

```python
import torch
model = torch.hub.load("mideind/GreynirSeq:main", "mbart25-cont-ees-enis", **{"bpe": "sentencepiece"})
model.eval()
model.sample(["This is an awesome test which shows how to use a pretrained translation model."])
['Þetta er æðislegt próf sem sýnir hvernig nota má formeðhöndlað þýðingarlíkan.[is_IS]']
```

A few things to note about loading this model:

- The model is based on mBART25 and needs to be specifically set to use sentencepiece as a BPE encoder.
- The output contains the target language code (7 characters).
- The model is large (~2.5GB disk space).

```python
import torch
model = torch.hub.load("mideind/GreynirSeq:main", "transformer-enis")
model.eval()
model.sample(["This is an awesome test which shows how to use a pretrained translation model."])
[' Þetta er frábært próf sem sýnir hvernig á að nota fyrirfram þjálfað þýðingalíkan.']
```

Notice here that:

- The output begins with a space and no language code
- This model is smaller (~200MB disk space)

In general:

- You are responsible for making sure that the sequence length input to the model does not exceed what they support and/or the lengths of sequences used during training.
- To run the models on GPU simply run `model.to("cuda")`, we refer to the pytorch documentation for further details.

#### Using the TranslateBART wrapper

Due to a bug in fairseq, the first translated token might be incorrect when using models trained using the BART objective (their names start with "mbart").
To avoid this problem you can use the TranslateBART wrapper.

```
from greynirseq.cli.greynirseq import TranslateBART
model = TranslateBART(device="cpu",
    batch_size=1,
    show_progress=False,
    max_input_words_split=250,
    model_args={"model_name_or_path": "mbart25-cont-ees-enis", "bpe": "sentencepiece"})
for translation in model.run(["This is an awesome test which shows how to use a pretrained translation model."]):
    print(translation)
```

In this wrapper we take care of a few things for you.

- We fix the previously-mentioned fairseq bug.
- We handling batching and longer sequence length splitting.
- We remove the target language code from the output.

### Local inference

Below is an example of how to use a model which is stored locally, or running a specific checkpoint.

```bash
$ echo "This is an awesome test which shows how to load a BART translation model from the file system." | greynirseq translate --additional-arguments mbart25-cont-enis.json

Þetta er frábært próf sem sýnir hvernig hlaða á inn BART þýðingarmódeli úr skráakerfinu.

$ cat mbart25-cont-enis.json
{
    "model_type": "mbart",
    "model_name_or_path": "/data/models/mbart25-cont-enis",
    "checkpoint_file": "checkpoint_9_36000.pt",
    "task": "translation_from_pretrained_bart",
    "bpe": "sentencepiece",
    "langs": "ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,is_IS,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN",
    "source_lang": "en_XX",
    "target_lang": "is_IS",
    "sentencepiece_model": "/data/models/mbart25-cont-enis/sentence.bpe.model"
}
```

And for a Transformer translation model you should specify `"model_type": "transformer"`. Example:

```
{
    "model_type": "transformer",
    "model_name_or_path": "/data/models/eng-isl-base-v6",
    "checkpoint_file": "checkpoint_6_61000.en-is.base-v6.pt",
    "task": "translation_with_backtranslation",
    "bpe": "gpt2",
    "gpt2_encoder_json": "/data/models/eng-isl-vocab/eng-isl-vocab/eng-isl-bbpe-32k/eng-isl-bbpe-32k-vocab.json",
    "gpt2_vocab_bpe": "/data/models/eng-isl-vocab/eng-isl-vocab/eng-isl-bbpe-32k/eng-isl-bbpe-32k-merges.txt",
    "model_uses_prepend_bos": false,
    "source_lang": "en",
    "target_lang": "is"
}
```
