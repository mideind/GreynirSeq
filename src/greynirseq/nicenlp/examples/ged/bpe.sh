#!/bin/bash

INPUT_DATA_DIR=$1
OUTPUT_DATA_DIR=$2
ENCODER_JSON=$3
VOCAB_BPE=$4

SCRIPT_NAME=$0
usage() {
    echo "$1"
    echo "Usage:"
    echo "$SCRIPT_NAME <input-data-dir> <output-data-dir> <encoder-json> <vocab-bpe>"
    exit 1
}

if [[ -z "$INPUT_DATA_DIR" ]]; then usage "Missing input-data-dir"; fi
if [[ -z "$OUTPUT_DATA_DIR" ]]; then usage "Missing output-data-dir"; fi
if [[ -z "$ENCODER_JSON" ]]; then usage "Missing encoder-json"; fi
if [[ -z "$VOCAB_BPE" ]]; then usage "Missing vocab-bpe"; fi


echo "-----------------------------------"
echo "BPE-ENCODING $INPUT_DATA_DIR > $OUTPUT_DATA_DIR"
echo "-----------------------------------"

mkdir -p "$OUTPUT_DATA_DIR"

for SPLIT in train test; do
    python -m greynirseq.utils.bpe.multiprocessing_bpe_encoder \
        --encoder-json $ENCODER_JSON \
        --vocab-bpe $VOCAB_BPE \
        --inputs "$INPUT_DATA_DIR/$SPLIT.input0" \
        --outputs "$OUTPUT_DATA_DIR/$SPLIT.input0.bpe" \
        --workers 60 \
        --add-prefix-space \
        --keep-empty

    # Don't need to preprocess the label files
    cp "$INPUT_DATA_DIR/$SPLIT.label" "$OUTPUT_DATA_DIR/"
done
