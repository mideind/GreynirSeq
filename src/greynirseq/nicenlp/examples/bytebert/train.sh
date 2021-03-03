#!/usr/bin/env bash

set -ex

TOTAL_UPDATES=125000      # Total number of training steps
MAX_UPDATES=125000      # Force stop at this number of training steps
WARMUP_UPDATES=10000    # Warmup the learning rate over this many updates
PEAK_LR=0.0007          # Peak learning rate, adjust as needed
TOKENS_PER_SAMPLE=512   # Max sequence length
# MAX_POSITIONS=512       # Num. positional embeddings (usually same as above)
MAX_SEQUENCES=10        # Number of sequences per batch (batch size)  # deprecated?
UPDATE_FREQ=2          # Accumulate gradients for 50 forward passes on each gpu
SAVE_INTERVAL=1000

DATA_DIR=data
TIMESTAMP=$(date +%s)
RUN_LOGS_DIR="run-logs"
HOSTNAME=$(hostname)
USER_DIR=$(pwd)

CKPT_DIR=/data/scratch/haukur/bytebert/checkpoints
LOG_DIR=/data/scratch/haukur/bytebert/tboard_logs

mkdir -p "$RUN_LOGS_DIR" "$CKPT_DIR"

export CUDA_LAUNCH_BLOCKING=0
fairseq-train --no-progress-bar $DATA_DIR \
                --user-dir "$USER_DIR" \
                --task byte_masked_lm --criterion masked_lm \
                --arch bytebert_small --sample-break-mode complete --tokens-per-sample $TOKENS_PER_SAMPLE \
                --optimizer adam --adam-betas '(0.9,0.98)' --adam-eps 1e-6 --clip-norm 0.0 \
                --lr-scheduler polynomial_decay --lr $PEAK_LR --warmup-updates $WARMUP_UPDATES --total-num-update $TOTAL_UPDATES \
                --dropout 0.1 --attention-dropout 0.1 --weight-decay 0.01 \
                --batch-size $MAX_SEQUENCES \
                --update-freq $UPDATE_FREQ \
                --bpe gpt2 --gpt2-encoder-json vocab.json \
                --gpt2-vocab-bpe merges.txt \
                --max-update $MAX_UPDATES --log-format simple --log-interval 1 \
                --tensorboard-logdir $LOG_DIR --save-interval-updates $SAVE_INTERVAL --keep-interval-updates 8 \
                --save-dir $CKPT_DIR \
                --num-workers 0  \
