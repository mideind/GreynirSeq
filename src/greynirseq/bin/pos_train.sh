#!/bin/sh

TOTAL_NUM_UPDATES=18000  #
WARMUP_UPDATES=500      #  percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=8         # Batch size.
SAVE_INTERVAL=250
LOG_INTERVAL=5

# Pretrained model
# ICEBERT_DIR='/mnt/windows/data/models/icebert-base-36k
ICEBERT_DIR=/data/models/icebert-base-36k
ROBERTA_PATH="$ICEBERT_DIR/model.pt"
ENCODER_JSON_PATH="$ICEBERT_DIR/icebert-bpe-vocab.json"
VOCAB_BPE_PATH="$ICEBERT_DIR/icebert-bpe-merges.txt"
# DATA_DIR=/mnt/windows/data/icebert-data-bin
DATA_DIR=$HOME/github/nicenlp-icebert/data-bin/sym
DATA_FINE_DIR=$HOME/github/nicenlp-icebert/data-bin/sym-fine

USER_DIR="$HOME/github/nicenlp-icebert/nicenlp"
BASE_SAVE_DIR="/data/models/pos_tagger"
NAME=debug_01

# CUDA_VISIBLE_DEVICES=0 fairseq-train "$DATA_DIR" \
export CUDA_VISIBLE_DEVICES=0
# CUDA_VISIBLE_DEVICES=0 gdb --args python $(which fairseq-train) $DATA_DIR \
fairseq-train "$DATA_DIR" \
    --valid-subset valid.cfg,valid.gold \
    --no-progress-bar \
    --log-interval $LOG_INTERVAL \
    --user-dir "$USER_DIR" \
    --num-workers 8 \
    --max-sentences $MAX_SENTENCES \
    --max-positions 512 \
    --term-schema "$DATA_DIR/term_schema.json" \
    --task pos_ice \
    --required-batch-size-multiple 1 \
    --arch icebert_const_base \
    --criterion pos_ice \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --find-unused-parameters \
    --bpe='gpt2' \
    --gpt2-encoder-json "$ENCODER_JSON_PATH" \
    --gpt2-vocab-bpe "$VOCAB_BPE_PATH" \
    --update-freq 1 \
    --max-update $TOTAL_NUM_UPDATES \
    --save-interval-updates $SAVE_INTERVAL \
    --restore-file "$ROBERTA_PATH" \
    --reset-optimizer --reset-dataloader --reset-meters \

# --tensorboard-logdir "$HOME/tensorboard_logdir/$NAME" \
# --save-dir "$BASE_SAVE_DIR/$NAME"

# --criterion multi_span \
# --task multi_span_prediction \
# --restore-file "$ROBERTA_PATH" \
# --tensorboard-logdir ./tensorboard_logdir/$NAME \
# --truncate-sequence \
# --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
