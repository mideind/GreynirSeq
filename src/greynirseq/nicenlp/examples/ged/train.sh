#!/bin/bash


# XXX TODO accept model arch as parameter

RUN_NAME=$1
MODEL_FILE=$2
DATA_DIR=$3
GREYNIRSEQ_PATH=$4
TERM_SCHEMA=$5
ENCODER_JSON=$6
VOCAB_BPE=$7

SCRIPT_NAME=$0
usage() {
    echo "$1"
    echo "Usage:"
    echo "$SCRIPT_NAME <run-name> <model-file> <data-dir> <greynirseq-path> <term-schema>"
    exit 1
}

if [[ -z "$RUN_NAME" ]]; then usage "Missing run-name"; fi
if [[ -z "$MODEL_FILE" ]]; then usage "Missing model-file"; fi
if [[ -z "$DATA_DIR" ]]; then usage "Missing data-dir"; fi
if [[ -z "$GREYNIRSEQ_PATH" ]]; then usage "Missing greynirseq-path"; fi
if [[ -z "$TERM_SCHEMA" ]]; then usage "Missing term-schema"; fi
if [[ -z "$ENCODER_JSON" ]]; then usage "Missing encoder-json"; fi
if [[ -z "$VOCAB_BPE" ]]; then usage "Missing vocab-bpe"; fi

TOTAL_NUM_UPDATES=7812  # Chosen pretty randomly. Maybe actually ignored in favor of epochs?
EPOCHS=6                # Number of epochs
WARMUP_UPDATES=500      # 
LR=5e-05                # Peak LR for polynomial LR scheduler
BATCH_SIZE=8            # Batch size.
UPDATE_INTERVAL=16      # How many batches between gradient updates. (Yes, the fairseq param name is confusing)

# The hardcoded parameters are mostly directly copied from some example from fairseq


echo "-----------------------------------"
echo "TRAINING $RUN_NAME"
echo " > MODEL FILE $MODEL_FILE"
echo "-----------------------------------"

CUDA_VISIBLE_DEVICES=0 fairseq-train $DATA_DIR \
    --restore-file $MODEL_FILE \
    --max-positions 512 \
    --batch-size $BATCH_SIZE \
    --max-tokens 4400 \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --max-epoch $EPOCHS \
    --best-checkpoint-metric acc_exact --maximize-best-checkpoint-metric \
    --find-unused-parameters \
    --update-freq $UPDATE_INTERVAL \
    --tensorboard-logdir logs/$RUN_NAME \
    --save-dir checkpoints/$RUN_NAME \
    --num-workers 8 \
    --user-dir $GREYNIRSEQ_PATH \
    --task multi_label_token_classification_task \
    --criterion multilabel_token_classification \
    --arch icebert_base_pos \
    --term-schema $TERM_SCHEMA \
    --bpe='gpt2' \
    --gpt2-encoder-json "$ENCODER_JSON" \
    --gpt2-vocab-bpe "$VOCAB_BPE" \
    --no-shuffle \
    --seed 1 \
    --no-progress-bar --log-interval 5 \


    #--max-update $TOTAL_NUM_UPDATES\ TOTAL_NUM_UPDATES=6000  # 
    #--save-interval-updates 2000\
    #--n-trans-layers-to-freeze 0\
    #--freeze-embeddings 1\
    #--no-progress-bar 
