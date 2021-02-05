#!/bin/bash


TEST_DATA_DIR=/data2/scratch/petur/grammatical-error-detection-2/data/simcategories/
MODEL_DIR=/data2/scratch/petur/grammatical-error-detection-2/data-prepared/simcategories/
CHECKPOINT_FILE=$(pwd)/checkpoints/testrun/icebert-base-36k.pt/checkpoint_best.pt
ENCODER_JSON=/data/models/icebert-base-36k/icebert-bpe-vocab.json
VOCAB_BPE=/data/models/icebert-base-36k/icebert-bpe-merges.txt

./evaluate.py "$TEST_DATA_DIR" "$MODEL_DIR" "$CHECKPOINT_FILE" "$ENCODER_JSON" "$VOCAB_BPE"

