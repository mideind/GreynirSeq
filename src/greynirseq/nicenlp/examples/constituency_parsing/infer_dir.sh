#!/usr/bin/env bash

WDIR="/data/scratch/haukur/parser"
DATA="${WDIR}/data/bin-finetune"
CKPT="${WDIR}/checkpoints/pretrain-02/checkpoint30.pt"
CKPT="/data/scratch/haukur/parser/checkpoints/pretrainsilver.lr5e-05.ngpus1.accum1.bsz50.warmup1000.maxupdate200000.ft/checkpoint253.pt"
CKPT="/data/scratch/haukur/parser/checkpoints/pretrainsilver.lr5e-05.ngpus1.accum1.bsz50.warmup1000.maxupdate200000/checkpoint_2_13000.pt"

python github-greynirseq/src/greynirseq/nicenlp/examples/constituency_parsing/predict_file.py \
    --data "${DATA}" \
    --checkpoint "${CKPT}" \
    --input-dir "${WDIR}/greynircorpus/testset/txt" \
    --input-suffix "txt" \
    --output-dir ./outputs-pretrain \

