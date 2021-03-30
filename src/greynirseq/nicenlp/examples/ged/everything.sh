#!/bin/bash

echo "-----------------------------------"
echo "STARTING TRAINING RUN"
echo "-----------------------------------"

# Simple logging of run parameters :)
set -x

# INPUT_DATA_DIR should contain these files:
#   test.input0
#   test.label
#   train.input0
#   train.label
SOURCE_DATA_DIR="data/iceErrorCorpus-2classes"
# OUTPUT_DATA_DIR is where processed data files get written
PROCESSED_DATA_DIR="data-prepared/iceErrorCorpus-2classes_icebert-large-v1"
# ENCODER_JSON is often something like *vocab.json
ENCODER_JSON="/data/models/icebert-large-v1/bpe_vocab/vocab.json"
# VOCAB_BPE is often something like *merges.txt
VOCAB_BPE="/data/models/icebert-large-v1/bpe_vocab/merges.txt"
# DICT is the dictionary file for the pretrained models.
DICT="/data/models/icebert-large-v1/dict.txt"
# MODELS_DIR is a directory containing model files.
# Their names should end in .pt
MODELS_DIR="checkpoint-view"
# SET_NAME is a name for this set of runs.
# It is used to create directories for checkpoints and logs.
SET_NAME="icebert-large-v1_1"

set +x # Turn of "logging". Unfortunately goes to stdout


# BPE-encode the training and test data.
#bash bpe.sh "$SOURCE_DATA_DIR" "$PROCESSED_DATA_DIR" "$ENCODER_JSON" "$VOCAB_BPE"

# Process encoded data into binary form that the model likes.
#bash preprocess.sh "$PROCESSED_DATA_DIR" "$DICT"

# Run all the training
bash runmany.sh "$MODELS_DIR" "$SET_NAME" "$PROCESSED_DATA_DIR"

# Evaluate the models to get better scores than just accuracy
#./evaluate.py MISSINGARGS
