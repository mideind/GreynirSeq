#!/usr/bin/env bash

WDIR=/data/scratch/haukur/parser
DATA_BASE="${WDIR}/data"

SEED=1234
TOTAL_NUM_UPDATES=200000 #
WARMUP_UPDATES=1000     #  percent of the number of updates
LR=5e-05                # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=50         # Batch size.
ACCUM=1

# Pretrained model
RESTORE_PATH=/data/models/icebert/icebert-first-rmh_different_vocab/icebert-base-36k/model.pt
ENCODER_JSON_PATH=/data/models/icebert/icebert-first-rmh_different_vocab/icebert-base-36k/icebert-bpe-vocab.json
VOCAB_BPE_PATH=/data/models/icebert/icebert-first-rmh_different_vocab/icebert-base-36k/icebert-bpe-merges.txt

FAIRSEQ_USER_DIR="${WDIR}/fairseq_user_dir"

NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | grep -oP "[[:digit:]]" | wc -l)
NAME=pretrain-02
NAME="pretrainsilver.lr${LR}.ngpus${NGPUS}.accum${ACCUM}.bsz${MAX_SENTENCES}.warmup${WARMUP_UPDATES}.maxupdate${TOTAL_NUM_UPDATES}"
TENSORBOARD_LOGDIR="${WDIR}/logs/tensorboard/$NAME"
PROGRESS_LOG_DIR="${WDIR}/logs/progress/$NAME"
SLURM_LOG_DIR="${WDIR}/logs/slurm"
CKPT_DIR="${WDIR}/checkpoints/$NAME"

mkdir -p "${TENSORBOARD_LOGDIR}" "${PROGRESS_LOG_DIR}" "${SLURM_LOG_DIR}"

DATA_BIN="${DATA_BASE}/bin-silver"

#export CUDA_LAUNCH_BLOCKING=1
#export NCCL_DEBUG=INFO
echo "Using GPU devices: '$CUDA_VISIBLE_DEVICES'"

#python -m pdb $(which fairseq-train) "${DATA_BIN}" \
fairseq-train "${DATA_BIN}" \
    --fp16 \
    --no-progress-bar \
    --log-interval 5 \
    --user-dir "${FAIRSEQ_USER_DIR}" \
    --num-workers 1 \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 4000 \
    --nonterm-schema "${DATA_BASE}/nonterm_schema.json" \
    --term-schema "${DATA_BASE}/term_schema.json" \
    --task parser \
    --required-batch-size-multiple 1 \
    --arch icebert_base_simple_parser \
    --criterion parser \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --find-unused-parameters \
    --seed $SEED \
    --bpe='gpt2' \
    --gpt2-encoder-json $ENCODER_JSON_PATH \
    --gpt2-vocab-bpe $VOCAB_BPE_PATH \
    --update-freq $ACCUM \
    --max-update $TOTAL_NUM_UPDATES \
    --save-dir "${CKPT_DIR}" \
    --restore-file $RESTORE_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --tensorboard-logdir "${TENSORBOARD_LOGDIR}/$NAME" \
    --save-interval-updates 500 \
    2>&1 | tee "$PROGRESS_LOG_DIR/log-$(date +'%Y-%m-%d-%H-%M').txt"
