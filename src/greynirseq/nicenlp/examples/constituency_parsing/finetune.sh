#!/usr/bin/env bash

WDIR=/data/scratch/haukur/parser
DATA_BASE="${WDIR}/data"

SEED=1234
TOTAL_NUM_UPDATES=50000 #
WARMUP_UPDATES=200 #  percent of the number of updates
LR=1e-05 # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=32 # Batch size.

# Pretrained model
#RESTORE_PATH=/data/models/icebert/icebert-base-igc-ic3-rmhvocab-IceBERT/checkpoint38.pt
#RESTORE_PATH=/data/scratch/haukur/parser/checkpoints/pretrain-02/checkpoint_1_4000.pt
RESTORE_PATH=/data/scratch/haukur/parser/checkpoints/pretrainsilver.lr5e-05.ngpus1.accum1.bsz50.warmup1000.maxupdate200000/checkpoint_2_13000.pt
ENCODER_JSON_PATH=/data/models/icebert/icebert-first-rmh_different_vocab/icebert-base-36k/icebert-bpe-vocab.json
VOCAB_BPE_PATH=/data/models/icebert/icebert-first-rmh_different_vocab/icebert-base-36k/icebert-bpe-merges.txt

FAIRSEQ_USER_DIR="${WDIR}/fairseq_user_dir"

#NAME=pretrain-02
NAME="pretrainsilver.lr5e-05.ngpus1.accum1.bsz50.warmup1000.maxupdate200000"
NAME="pretrainsilver.lr5e-05.ngpus1.accum1.bsz50.warmup1000.maxupdate200000.ft"
TENSORBOARD_LOGDIR="${WDIR}/logs/tensorboard/$NAME"
PROGRESS_LOG_DIR="${WDIR}/logs/progress/$NAME"
SLURM_LOG_DIR="${WDIR}/logs/slurm"
CKPT_DIR="${WDIR}/checkpoints/$NAME"

mkdir -p "${TENSORBOARD_LOGDIR}" "${PROGRESS_LOG_DIR}" "${SLURM_LOG_DIR}"

DATA_BIN="${DATA_BASE}/bin-finetune"

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
    --update-freq 1 \
    --max-update $TOTAL_NUM_UPDATES \
    --save-dir "${CKPT_DIR}" \
    --restore-file $RESTORE_PATH \
    --reset-optimizer --reset-dataloader --reset-meters \
    --tensorboard-logdir "${TENSORBOARD_LOGDIR}/$NAME" \
    --save-interval-updates 100 \
    #2>&1 | tee "$PROGRESS_LOG_DIR/log-$(date +'%Y-%m-%d-%H-%M').txt"
