#!/bin/bash

## DOESN'T WORK
NUM_CLASSES=2           # Number of classes for the classification task.
CHECKPOINT_PATH=./binary-classifier/model.pt

CUDA_VISIBLE_DEVICES=0 fairseq-interactive data-bin/ \
    --max-positions 512 \
    --path $CHECKPOINT_PATH \
    --task sentence_prediction \
    --num-classes $NUM_CLASSES \
