#!/usr/bin/env bash

MIM_GOLD_PATH=/data/datasets/MIM-GOLD-1_0_SETS
OUTPUT_PATH=$MIM_GOLD_PATH/for_training_noprepend_space_legacy_icebert #for_training
SEED=70
VALIDATION_PROPORTION=0.04

#VOCAB_PATH=/data/models/icebert/bpe_vocab
#ENCODER_JSON=$VOCAB_PATH/vocab.json
#MERGES_TXT=$VOCAB_PATH/merges.txt
#DICT=$VOCAB_PATH/tokenized_dict.txt

VOCAB_PATH=/data/models/icebert-base-36k
ENCODER_JSON=$VOCAB_PATH/icebert-bpe-vocab.json
MERGES_TXT=$VOCAB_PATH/icebert-bpe-merges.txt
DICT=$VOCAB_PATH/dict.txt

LAB_DICT=labdict.txt

MIM_TEST_POSTFIX='PM.plain'
MIM_TRAIN_POSTFIX='TM.plain'

GREYNIRSEQ_PATH=/home/vesteinn/work/GreynirSeq
export PATH="$PATH:${GREYNIRSEQ_PATH}/src/greynirseq/utils:${GREYNIRSEQ_PATH}/src/greynirseq/utils/preprocessing"
mkdir -p $OUTPUT_PATH

for SPLIT_IDX in $(seq -f "%02g" 1 10)
do
    SPLIT_PATH=$OUTPUT_PATH/$SPLIT_IDX
    mkdir -p "$SPLIT_PATH"
    TRAIN_FILE=$MIM_GOLD_PATH/$SPLIT_IDX$MIM_TRAIN_POSTFIX
    TEST_FILE=$MIM_GOLD_PATH/$SPLIT_IDX$MIM_TEST_POSTFIX

    parse_ifd.py --input "$TRAIN_FILE" --output_folder "$SPLIT_PATH" --prefix TM
    parse_ifd.py --input "$TEST_FILE" --output_folder "$SPLIT_PATH" --prefix PM

    split_train_dev.py \
        --seed $SEED \
        -p $VALIDATION_PROPORTION \
        --lines \
        "$SPLIT_PATH"/TM.input0 "$SPLIT_PATH"/train.input0 "$SPLIT_PATH"/valid.input0

    split_train_dev.py \
        --seed $SEED \
        -p $VALIDATION_PROPORTION \
        --lines \
        $SPLIT_PATH/TM.label0 "$SPLIT_PATH"/train.label0 "$SPLIT_PATH"/valid.label0

    
    for SPLIT in train valid
    do
        python -m greynirseq.utils.bpe.multiprocessing_bpe_encoder \
            --encoder-json $ENCODER_JSON \
            --vocab-bpe $MERGES_TXT \
            --inputs "$SPLIT_PATH"/$SPLIT.input0 \
            --outputs "$SPLIT_PATH"/$SPLIT.input0.bpe \
            --workers 60 \
            --keep-empty 
	    #--add-prefix-space
    done
    
    python -m greynirseq.utils.bpe.multiprocessing_bpe_encoder \
        --encoder-json $ENCODER_JSON \
        --vocab-bpe $MERGES_TXT \
	--inputs "$SPLIT_PATH"/PM.input0 \
	--outputs "$SPLIT_PATH"/test.input0.bpe \
        --workers 60 \
        --keep-empty \
	--add-prefix-space

    fairseq-preprocess \
       --only-source \
       --trainpref $SPLIT_PATH/train.input0.bpe \
       --testpref $SPLIT_PATH/test.input0.bpe \
       --workers 60 \
       --srcdict $DICT \
       --validpref $SPLIT_PATH/valid.input0.bpe \
       --destdir $SPLIT_PATH/bin

    fairseq-preprocess \
       --only-source \
       --trainpref $SPLIT_PATH/train.label0 \
       --workers 60 \
       --validpref $SPLIT_PATH/valid.label0 \
       --testpref $SPLIT_PATH/PM.label0 \
       --destdir $SPLIT_PATH/bin/labels0 \
       --srcdict $LAB_DICT



    mv "$SPLIT_PATH"/bin/labels0/train.bin "$SPLIT_PATH"/bin/train.term.bin
    mv "$SPLIT_PATH"/bin/labels0/train.idx "$SPLIT_PATH"/bin/train.term.idx

    mv "$SPLIT_PATH"/bin/labels0/dict.txt "$SPLIT_PATH"/bin/dict_term.txt
    mv "$SPLIT_PATH"/bin/labels0/valid.bin "$SPLIT_PATH"/bin/valid.term.bin
    mv "$SPLIT_PATH"/bin/labels0/valid.idx "$SPLIT_PATH"/bin/valid.term.idx
    mv "$SPLIT_PATH"/bin/labels0/test.bin "$SPLIT_PATH"/bin/test.term.bin
    mv "$SPLIT_PATH"/bin/labels0/test.idx "$SPLIT_PATH"/bin/test.term.idx

done
