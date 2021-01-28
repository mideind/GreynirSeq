#!/bin/bash

DATA_DIR=$1
DICT=$2

SCRIPT_NAME=$0
usage() {
    echo "$1"
    echo "Usage:"
    echo "$SCRIPT_NAME <data-dir> <dict>"
    exit 1
}

if [[ -z "$DATA_DIR" ]]; then usage "Missing data-dir"; fi
if [[ -z "$DICT" ]]; then usage "Missing dict"; fi


echo "-----------------------------------"
echo "PREPROCESSING $DATA_DIR"
echo "-----------------------------------"

fairseq-preprocess \
    --only-source \
    --trainpref "$DATA_DIR"/train.input0.bpe \
    --validpref "$DATA_DIR"/test.input0.bpe \
    --destdir "$DATA_DIR"/input0 \
    --workers 60 \
    --srcdict "$DICT"

fairseq-preprocess \
    --only-source \
    --trainpref "$DATA_DIR"/train.label \
    --validpref "$DATA_DIR"/test.label \
    --destdir "$DATA_DIR"/label \
    --workers 60

cp "$DATA_DIR"/input0/* "$DATA_DIR"
cp "$DATA_DIR"/label/dict.txt "$DATA_DIR"/dict_term.txt
cp "$DATA_DIR"/label/train.bin "$DATA_DIR"/train.term.bin
cp "$DATA_DIR"/label/train.idx "$DATA_DIR"/train.term.idx
cp "$DATA_DIR"/label/valid.bin "$DATA_DIR"/valid.term.bin
cp "$DATA_DIR"/label/valid.idx "$DATA_DIR"/valid.term.idx

