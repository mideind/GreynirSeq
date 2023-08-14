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

## Training

Here is an example script for training a document translation model based on mbart25 which also noises the input and uses backtranslation.

This script requires fairseq version 0.12 or later.

```bash:
#!/usr/bin/env bash
# This is usually the directory which contains this script
WDIR=/path/to/workdir
# This directory needs to contain/import the necessary files for training, i.e. the document_translation_from_pretrained_bart
USER_DIR=$WDIR/GreynirSeq/src/greynirseq
# if using wandb
WANDB_PROJECT=wandb-project-name
# where to save checkpoints and logs
SAVE_DIR=${WDIR}/$EXPERIMENT

# This training setup
ENG=en_XX
ISL=is_IS
SRC_LANG=$ISL
TGT_LANG=$ENG

# Checkpoint to load from
FROM_MODEL_DIR=/some/path
CKPT=$FROM_MODEL_DIR/fairseq_model.pt

# Copy the dicts (src and tgt) from the original model to data
cp $FROM_MODEL_DIR/dict.$SRC_LANG.txt $WDIR/data/dict.$SRC_LANG.txt
cp $FROM_MODEL_DIR/dict.$TGT_LANG.txt $WDIR/data/dict.$TGT_LANG.txt

# mbart25 langs
LANGS="ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,is_IS,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN"

LEARNING_RATE=5e-05  # set this
UPDATE_FREQ=10  # set this
EXPERIMENT=v10  # set this

ALL_TRAIN_DATA="list of all training data files"
BT_DATA="list of backtranslation data files, they need to be included in the all train data as well"
ALIGN_DATA="which of the files in all train data contain alignments"
VALID_DATA="all the validation data files"

# Below are opinionated settings for training
fairseq-train $WDIR/data \
    --fp16 \
    --no-progress-bar \
    --encoder-normalize-before --decoder-normalize-before \
    --bpe 'sentencepiece' --sentencepiece-model $FROM_MODEL_DIR/sentencepiece.bpe.model \
    --arch mbart_large --layernorm-embedding \
    --criterion cross_entropy \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' --clip-norm 1.5 \
    --lr-scheduler polynomial_decay --lr ${LEARNING_RATE} --total-num-update 1500 \
    --best-checkpoint-metric ppl --warmup-updates 150 --max-update 1500 \
    --patience 5 \
    --max-tokens 2500 --update-freq ${UPDATE_FREQ} \
    --num-workers 0 \
    --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.001 \
    --user-dir $USER_DIR \
    --task document_translation_from_pretrained_bart \
    --langs $LANGS \
    --source-lang $SRC_LANG --target-lang $TGT_LANG \
    --num-preprocess-workers 8 \
    --max-sentences 400 \
    --train-subset "$ALL_TRAIN_DATA" \
    --valid-subset "$VALID_DATA" \
    --bt-subset "$BT_DATA" \
    --align-subset "$ALIGN_DATA" \
    --parallel-prob 0.7 \
    --fragment-noise-prob 0.005 \
    --num-preprocess-workers 8 \
    --max-merges 10 \
    --global-skip-noise-prob 0.20 \
    --drop-word-prob 0.005 \
    --max-shift-distance 3 \
    --shift-prob 0.005 \
    --swap-prob 0.005 \
    --delete-prob 0.005 \
    --insert-prob 0.005 \
    --duplicate-prob 0.005 \
    --case-prob 0.005 \
    --substitution-prob 0.005 \
    --seq-lower-prob 0.005 \
    --seq-upper-prob 0.00 \
    --seed 2345 --log-format simple --log-interval 5 \
    --validate-interval-updates 100 \
    --save-interval-updates 200 \
    --keep-interval-updates 4 \
    --save-dir $SAVE_DIR \
    --skip-invalid-size-inputs-valid-test \
    --wandb-project $WANDB_PROJECT \
    --restore-file $CKPT \
    --ddp-backend pytorch_ddp \
    --reset-dataloader --reset-lr-scheduler --reset-meters --reset-optimizer
```bash
