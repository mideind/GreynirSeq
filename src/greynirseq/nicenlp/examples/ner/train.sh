TOTAL_NUM_UPDATES=50000  #
WARMUP_UPDATES=500      #  percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=8         # Batch size.
SAVE_INTERVAL=250
LOG_INTERVAL=5
MAX_TOKENS=3000

# Pretrained model
ROBERTA_PATH=
ENCODER_JSON_PATH=
VOCAB_BPE_PATH=

DATA_PATH=
GREYNIRSEQ_PATH=
ARC=multiclass_roberta_base
CRITERION=multi_class_token_classification
TASK=multi_class_token_classification_task
UPDATE_FREQ=2
OUT_DIR=./ner_out

fairseq-train $DATA_PATH \
    --save-dir $OUT_DIR/chkpts \
    --user-dir $GREYNIRSEQ_PATH \
    --update-freq $UPDATE_FREQ \
    --log-interval 10\
    --no-progress-bar \
    --finetune-from-model $ROBERTA_PATH \
    --max-positions 512 \
    --num-workers 8 \
    --log-interval $LOG_INTERVAL \
    --batch-size $MAX_SENTENCES \
    --max-tokens $MAX_TOKENS \
    --task $TASK \
    --required-batch-size-multiple 1 \
    --arch $ARC \
    --criterion $CRITERION \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --find-unused-parameters \
    --bpe='gpt2' \
    --gpt2-encoder-json $ENCODER_JSON_PATH \
    --gpt2-vocab-bpe $VOCAB_BPE_PATH \
    --update-freq 1 \
    --tensorboard-logdir $OUT_DIR/tensorboard_logdir/$NAME \
    --max-update $TOTAL_NUM_UPDATES\
    --save-interval-updates 5000 \
    --n-trans-layers-to-freeze 0\
    --freeze-embeddings 1
