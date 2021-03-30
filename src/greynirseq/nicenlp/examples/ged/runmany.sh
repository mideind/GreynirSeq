#!/bin/bash

MODELS_DIR=$1
SET_NAME=$2
DATA_DIR=$3
GREYNIRSEQ_PATH=$4
TERM_SCHEMA=$5
ENCODER_JSON=$6
VOCAB_BPE=$7

SCRIPT_NAME=$0
usage() {
    echo "$1"
    echo "Usage:"
    echo "$SCRIPT_NAME <models-dir> <set-name> <data-dir> <greynirseq-path> <term-schema>"
    echo $TERM_SCHEMA
    exit 1
}

if [[ -z "$MODELS_DIR" ]]; then usage "Missing models-dir"; fi
if [[ -z "$SET_NAME" ]]; then usage "Missing set-name"; fi
if [[ -z "$DATA_DIR" ]]; then usage "Missing data-dir"; fi
if [[ -z "$GREYNIRSEQ_PATH" ]]; then usage "Missing greynirseq-path"; fi
if [[ -z "$TERM_SCHEMA" ]]; then usage "Missing term-schema"; fi
if [[ -z "$ENCODER_JSON" ]]; then usage "Missing encoder-json"; fi
if [[ -z "$VOCAB_BPE" ]]; then usage "Missing vocab-bpe"; fi

MODEL_SUFFIX=".pt"
MODELS=$(find "$MODELS_DIR" -name "*$MODEL_SUFFIX")
if [[ ! $? -eq 0 ]]; then usage "Did not find models-dir '$MODELS_DIR'"; fi
if [[ -z "$MODELS" ]]; then usage "No models found in '$MODELS_DIR'. (Do the model filenames end in $MODEL_SUFFIX?)"; fi


for MODEL_FULLPATH in $MODELS;
do
    MODEL=$(basename "$MODEL_FULLPATH")
    bash train.sh "$SET_NAME"/"$MODEL" "$MODEL_FULLPATH" "$DATA_DIR" "$GREYNIRSEQ_PATH" \
        "$TERM_SCHEMA" "$ENCODER_JSON" "$VOCAB_BPE"
done
