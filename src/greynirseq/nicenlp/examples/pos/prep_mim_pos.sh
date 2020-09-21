# Copyright ...
WDIR=/home/vesteinn/work

MIM_GOLD_PATH=$WDIR/pytorch_study/data/MIM/MIM-GOLD-1_0_SETS
OUTPUT_PATH=$WDIR/pytorch_study/data/MIM/MIM-GOLD-1_0_sets_for_training
SEED=70
VALIDATION_PROPORTION=0.04
ENCODER_JSON_PATH=/home/vesteinn/icebert-base-36k/icebert-bpe-vocab.json
VOCAB_BPE_PATH=/home/vesteinn/icebert-base-36k/icebert-bpe-merges.txt
DICT=/home/vesteinn/icebert-base-36k/dict.txt
LAB_DICT=/home/vesteinn/work/pytorch_study/labdict.txt

MIM_TEST_POSTFIX='PM.plain'
MIM_TRAIN_POSTFIX='TM.plain'

for SPLIT_IDX in $(seq -f "%02g" 1 10)
do
    SPLIT_PATH=$OUTPUT_PATH/$SPLIT_IDX
    mkdir -p $SPLIT_PATH
    TRAIN_FILE=$MIM_GOLD_PATH/$SPLIT_IDX$MIM_TRAIN_POSTFIX
    TEST_FILE=$MIM_GOLD_PATH/$SPLIT_IDX$MIM_TEST_POSTFIX

    python parse_ifd.py --input $TRAIN_FILE --output_folder $SPLIT_PATH --prefix TM
    python parse_ifd.py --input $TEST_FILE --output_folder $SPLIT_PATH --prefix PM

    python split_train_dev.py \
        --seed $SEED \
        -p $VALIDATION_PROPORTION \
        --lines \
        $SPLIT_PATH/TM.input0 $SPLIT_PATH/train.input0 $SPLIT_PATH/valid.input0

    python split_train_dev.py \
        --seed $SEED \
        -p $VALIDATION_PROPORTION \
        --lines \
        $SPLIT_PATH/TM.label0 $SPLIT_PATH/train.label0 $SPLIT_PATH/valid.label0

    
    for SPLIT in train valid
    do
        python -m multiprocessing_bpe_encoder \
            --encoder-json $ENCODER_JSON_PATH \
            --vocab-bpe $VOCAB_BPE_PATH \
	    --inputs $SPLIT_PATH/$SPLIT.input0 \
	    --outputs $SPLIT_PATH/$SPLIT.input0.bpe \
	    --workers 60 \
	    --keep-empty
    done
    
    fairseq-preprocess \
       --only-source \
       --trainpref $SPLIT_PATH/train.input0.bpe \
       --workers 60 \
       --srcdict $DICT \
       --validpref $SPLIT_PATH/valid.input0.bpe \
       --destdir $SPLIT_PATH/bin/input0

    fairseq-preprocess \
       --only-source \
       --trainpref $SPLIT_PATH/train.label0 \
       --validpref $SPLIT_PATH/valid.label0 \
       --workers 60 \
       --destdir $SPLIT_PATH/bin/labels0 \
       --srcdict $LAB_DICT

done
