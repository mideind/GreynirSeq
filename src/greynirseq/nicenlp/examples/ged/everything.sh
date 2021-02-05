#!/bin/bash

echo "-----------------------------------"
echo "STARTING TRAINING RUN"
echo "-----------------------------------"

# Simple logging of run parameters :)
set -x

# SOURCE_DATA_DIR should contain these files:
#   test.input0
#   test.label
#   train.input0
#   train.label
SOURCE_DATA_DIR="/data2/scratch/petur/grammatical-error-detection-2/data/simcategories/"
# PROCESSED_DATA_DIR is where processed data files get written
PROCESSED_DATA_DIR="/data2/scratch/petur/grammatical-error-detection-2/data-prepared/simcategories/"
# ENCODER_JSON is often something like *vocab.json
ENCODER_JSON="/data/models/icebert-base-36k/icebert-bpe-vocab.json"
# VOCAB_BPE is often something like *merges.txt
VOCAB_BPE="/data/models/icebert-base-36k/icebert-bpe-merges.txt"
# DICT is the dictionary file for the pretrained models.
DICT="/data/models/icebert-base-36k/dict.txt"
# MODELS_DIR is a directory containing model files.
# Their names should end in .pt
MODELS_DIR="/data2/scratch/petur/grammatical-error-detection-2/models/"
# SET_NAME is a name for this set of runs.
# It is used to create directories for checkpoints and logs.
SET_NAME="testrun"
# GREYNIRSEQ_GIT is a path to a checked out GreynirSeq git repo
GREYNIRSEQ_GIT="/home/petur/dev/GreynirSeq/"
GREYNIRSEQ_PATH="$GREYNIRSEQ_GIT/src/greynirseq/"
# TERM_SCHEMA is the label schema for the multilabel task
TERM_SCHEMA="/data2/scratch/petur/grammatical-error-detection-2/ged-simcategories-schema.json"

set +x # Turn of "logging". Unfortunately goes to stdout


# BPE-encode the training and test data.
#bash bpe.sh "$SOURCE_DATA_DIR" "$PROCESSED_DATA_DIR" "$ENCODER_JSON" "$VOCAB_BPE"

# Process encoded data into binary form that the model likes.
#bash preprocess.sh "$PROCESSED_DATA_DIR" "$DICT"

# Run all the training
bash runmany.sh "$MODELS_DIR" "$SET_NAME" "$PROCESSED_DATA_DIR" "$GREYNIRSEQ_PATH" \
    "$TERM_SCHEMA" "$ENCODER_JSON" "$VOCAB_BPE"

# Evaluate the models to get better scores than just accuracy
#./evaluate.py MISSINGARGS
