TOTAL_NUM_UPDATES=5500  # 
WARMUP_UPDATES=200      #  percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
NUM_CLASSES_MUTEX=35    # Number of classes for the classification task.
NUM_CLASSES_BINARY=26   # Number of binary classes, non mutually exclusive.
MAX_SENTENCES=8         # Batch size.

# Pretrained model
ROBERTA_PATH=/home/vesteinn/icebert-base-36k/model.pt
ENCODER_JSON_PATH='/home/vesteinn/icebert-base-36k/icebert-bpe-vocab.json'
VOCAB_BPE_PATH='/home/vesteinn/icebert-base-36k/icebert-bpe-merges.txt'
CRITERION=multi_label_idf

for ITERATION in 06 07 08 09 10
do
    NAME=multilabel_split_$ITERATION
    
    CUDA_VISIBLE_DEVICES=0 fairseq-train /home/vesteinn/work/pytorch_study/data/MIM/MIM-GOLD-1_0_sets_for_training/$ITERATION/bin \
    --save-dir ./checkpoints_$ITERATION\
    --user-dir ../nicenlp \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 4400 \
    --task multi_label_word_classification \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --separator-token 2 \
    --arch icebert \
    --criterion $CRITERION \
    --num-classes-mutex $NUM_CLASSES_MUTEX \
    --num-classes-binary $NUM_CLASSES_BINARY \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --truncate-sequence \
    --find-unused-parameters \
    --bpe='gpt2' \
    --gpt2-encoder-json $ENCODER_JSON_PATH \
    --gpt2-vocab-bpe $VOCAB_BPE_PATH \
    --update-freq 6 \
    --tensorboard-logdir ./tensorboard_logdir/$NAME \
    --max-update $TOTAL_NUM_UPDATES\
    --save-interval-updates 500
done
