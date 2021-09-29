#!/usr/bin/env bash

# Assumes running inside docker container

MODEL=/model/checkpoint.pt
DATA=/model/data
INPUT_PATH=/data/input.txt
OUTPUT_PATH=/data/output.txt

python /github/greynirseq/src/greynirseq/nicenlp/examples/constituency_parsing/predict_file.py \
    --data ${DATA} \
    --checkpoint ${MODEL} \
    --input-path ${INPUT_PATH} \
    --output-path ${OUTPUT_PATH} \

