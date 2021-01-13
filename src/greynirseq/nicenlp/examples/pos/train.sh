TOTAL_NUM_UPDATES=1000000  # 
WARMUP_UPDATES=200      #  percent of the number of updates
LR=1e-05                # Peak LR for polynomial LR scheduler.
MAX_SENTENCES=32         # Batch size.
MAX_TOKENS=3000

# Pretrained model
ICEBERT_MODEL_DIR=/data/models/icebert/icebert-base
#ICEBERT_MODEL_DIR=/data/models/icebert-base-36k
ICEBERT_MODEL=$ICEBERT_MODEL_DIR/ckpts/checkpoint_70_49000.pt
#ICEBERT_MODEL=model.pt
VOCAB_PATH=/data/models/icebert/bpe_vocab
#VOCAB_PATH=$ICEBERT_MODEL_DIR
#ENCODER_JSON=$VOCAB_PATH/icebert-bpe-vocab.json
#MERGES_TXT=$VOCAB_PATH/icebert-bpe-merges.txt
ENCODER_JSON=$VOCAB_PATH/vocab.json
MERGES_TXT=$VOCAB_PATH/merges.txt
DICT=$VOCAB_PATH/tokenized_dict.txt
LAB_DICT=labdict.txt

DATA_PATH=/data/datasets/MIM-GOLD-1_0_SETS/for_training
#DATA_PATH=~/work/pytorch_study/data/MIM/MIM-GOLD-1_0_sets_for_training
SAVE_DIR=/data/scratch/vesteinn/icebert-base/checkpoints
#SAVE_DIR=/data/scratch/vesteinn/icebert-old-pos-testing/checkpoints
GREYNIRSEQ_PATH=/home/vesteinn/work/GreynirSeq/src/greynirseq

# ARC=icebert_large_pos
ARC=icebert_base_pos
CRITERION=multilabel_token_classification
TASK=multi_label_token_classification_task
TERM_SCHEMA=terms.json
UPDATE_FREQ=2


for ITERATION in $(seq -f "%02g" 1 1)
do
    NAME=multilabel_split_$ITERATION
    IT_DATA_PATH=${DATA_PATH}/$ITERATION/bin
    
    CUDA_VISIBLE_DEVICES=0 fairseq-train $IT_DATA_PATH \
    --save-dir $SAVE_DIR/checkpoints_$ITERATION\
    --user-dir $GREYNIRSEQ_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens $MAX_TOKENS \
    --task $TASK \
    --num-workers 8\
    --term-schema $TERM_SCHEMA \
    --required-batch-size-multiple 1 \
    --arch $ARC \
    --criterion $CRITERION \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --best-checkpoint-metric acc_exact --maximize-best-checkpoint-metric \
    --find-unused-parameters \
    --bpe='gpt2' \
    --gpt2-encoder-json $ENCODER_JSON \
    --gpt2-vocab-bpe $MERGES_TXT \
    --update-freq $UPDATE_FREQ \
    --tensorboard-logdir ./tensorboard_logdir/$NAME \
    --max-update $TOTAL_NUM_UPDATES\
    --save-interval-updates 10000\
    --n-trans-layers-to-freeze 0\
    --freeze-embeddings 1\
    --no-progress-bar --no-shuffle #--restore-file $ICEBERT_MODEL --reset-optimizer --reset-dataloader --reset-meters 
done
#--restore-file $ICEBERT_MODEL \

#--reset-optimizer --reset-dataloader --reset-meters \

