#!/usr/bin/env bash

WDIR="/data/scratch/haukur/parser"
DATA="${WDIR}/data/bin-finetune"

python github-greynirseq/src/greynirseq/nicenlp/examples/constituency_parsing/predict_file.py \
    --data "${DATA}" \
    --checkpoint "${WDIR}/checkpoints/pretrain-02/checkpoint30.pt" \
    --input-path "${WDIR}/greynircorpus/testset/txt/greynir_corpus_00441.txt"

