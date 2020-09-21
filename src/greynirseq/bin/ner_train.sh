TOTAL_NUM_UPDATES=18000  #
WARMUP_UPDATES=500      #  percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=8         # Batch size.
SAVE_INTERVAL=250
LOG_INTERVAL=5

# Pretrained model
ROBERTA_PATH=/home/vesteinn/icebert-base-36k/model.pt
ENCODER_JSON_PATH='/home/vesteinn/icebert-base-36k/icebert-bpe-vocab.json'
VOCAB_BPE_PATH='/home/vesteinn/icebert-base-36k/icebert-bpe-merges.txt'

NAME=ner_slset

CUDA_VISIBLE_DEVICES=0 fairseq-train /home/vesteinn/data/MIM-GOLD-NER/8_entity_types/bin/bin \
    --user-dir ../nicenlp \
    --no-progress-bar \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --num-workers 8 \
    --log-interval $LOG_INTERVAL \
    --max-sentences $MAX_SENTENCES \
    --task pos_ice \
    --term-schema /home/vesteinn/data/MIM-GOLD-NER/term.json \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --arch icebert_const_base \
    --criterion pos_ice \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --find-unused-parameters \
    --bpe='gpt2' \
    --gpt2-encoder-json $ENCODER_JSON_PATH \
    --gpt2-vocab-bpe $VOCAB_BPE_PATH \
    --update-freq 1 \
    --tensorboard-logdir ./tensorboard_logdir/$NAME \
    --max-update $TOTAL_NUM_UPDATES\
    --save-interval-updates 500 \
    --save-dir "/data/models/icebert_ner/ner_slset"
