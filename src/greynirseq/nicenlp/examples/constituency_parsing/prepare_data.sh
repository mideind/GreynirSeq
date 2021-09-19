#!/usr/bin/env bash

# export parse trees as labelled spans
#python github-greynirseq/src/greynirseq/nicenlp/utils/constituency/prep_greynir_new.py export gc01.gld throwaway/text throwaway/term throwaway/nonterm --binarize-trees
WDIR=/data/scratch/haukur/parser
EXPORTER=${WDIR}/github-greynirseq/src/greynirseq/nicenlp/utils/constituency/prep_greynir_new.py
DATA_BASE=${WDIR}/data
DATA_DEBUG=${DATA_BASE}/debug

DATA_TEST="${DATA_BASE}/test"
DATA_PRETRAIN="${DATA_BASE}/pretrain"
DATA_TRAIN="${DATA_BASE}/train"
DATA_VALID="${DATA_BASE}/valid"
DATA_BIN=${DATA_BASE}/bin
DATA_BIN_PRETRAIN=${DATA_BASE}/bin-pretrain

mkdir -p $DATA_DEBUG $DATA_TEST $DATA_PRETRAIN $DATA_TRAIN $DATA_VALID $DATA_BIN $DATA_BIN_PRETRAIN

DATA_DEBUG_PSD=${WDIR}/greynircorpus/devset/psd/greynir_corpus_00462.gld
DATA_DEBUG_PSD=${WDIR}/greynircorpus/devset/psd/greynir_corpus_00001.gld
DATA_DEBUG_PSD=${WDIR}/greynircorpus/devset/psd/greynir_corpus_00021.gld

DATA_PRETRAIN_PSD=${WDIR}/greynircorpus/psd/silver/silver
DATA_DEV_SILVER_PSD=${DATA_BASE}/devset_silver.psd
DATA_TRAIN_PSD=${DATA_BASE}/train_for_finetune.psd

DATA_TRAIN_GOLD_FILENAMES=${DATA_BASE}/devset_gold.train.txt
DATA_DEV_GOLD_FILENAMES=${DATA_BASE}/devset_gold.dev.txt

### we want a fixed seed for the 'shuf' command
get_fixed_random_byte_stream()
{
  # Obtained from: https://stackoverflow.com/questions/60266215/shuffle-output-of-find-with-fixed-seed?noredirect=1&lq=1
  openssl enc -aes-256-ctr -pass pass:"$1" -nosalt </dev/zero 2>/dev/null
}

set -ex

### data is kept in many separate .gld files, we need to merge them
rm -f ${DATA_DEV_GOLD_FILENAMES}
rm -f ${DATA_TRAIN}/text.txt ${DATA_TRAIN}/term.txt ${DATA_TRAIN}/nonterm.txt
if [ ! -f ${DATA_DEV_GOLD_FILENAMES} ] ; then
    # total number of files 453
    NUM_DEV=100
    NUM_TRAIN=353
    SEED=1
    find ${WDIR}/greynircorpus/devset/psd/ -type f -name "*.gld" | sort | shuf --random-source=<(get_fixed_random_byte_stream ${SEED}) \
        | head -n ${NUM_DEV} | sort > ${DATA_DEV_GOLD_FILENAMES}
    find ${WDIR}/greynircorpus/devset/psd/ -type f -name "*.gld" | sort | shuf --random-source=<(get_fixed_random_byte_stream ${SEED}) \
        | tail -n +$((${NUM_DEV} + 1)) | sort > ${DATA_TRAIN_GOLD_FILENAMES}
fi

### Prepare small dataset for debug purposes
python ${EXPORTER} export ${DATA_DEBUG_PSD} \
    ${DATA_DEBUG}/text.txt ${DATA_DEBUG}/term.txt ${DATA_DEBUG}/nonterm.txt \
    --binarize-trees --ignore-errors --error-log=${WDIR}/data/error_trees_gold.psd --append-errors

### Prepare training (finetuning) set (using part of gold devset data)
while read FILEPATH ; do
    python ${EXPORTER} export ${FILEPATH} \
        ${DATA_TRAIN}/text.txt ${DATA_TRAIN}/term.txt ${DATA_TRAIN}/nonterm.txt \
        --binarize-trees --append --ignore-errors --error-log=${WDIR}/data/error_trees_gold.psd --append-errors
done < ${DATA_TRAIN_GOLD_FILENAMES}

## Prepare gold validation set
while read FILEPATH ; do
    python ${EXPORTER} export ${FILEPATH} \
        ${DATA_VALID}/text.txt ${DATA_VALID}/term.txt ${DATA_VALID}/nonterm.txt \
        --binarize-trees --append --ignore-errors --error-log=${WDIR}/data/error_trees_gold.psd --append-errors
done < ${DATA_DEV_GOLD_FILENAMES}

rm -f ${DATA_TEST}/text ${DATA_TEST}/nonterm ${DATA_TEST}/term
# Prepare testset set (using gold testset)
find ${WDIR}/greynircorpus/testset/psd/ -type f -name "*.gld" | while read FILEPATH ; do
    python ${EXPORTER} export ${FILEPATH} \
        ${DATA_TEST}/text.txt  ${DATA_TEST}/term.txt ${DATA_TEST}/nonterm.txt \
        --binarize-trees --append --ignore-errors --error-log=${WDIR}/data/error_trees_gold.psd --append-errors
done

# Prepare pretraining set (data from cfg parser, ie silver)
python ${EXPORTER} export ${DATA_PRETRAIN_PSD} \
    ${DATA_PRETRAIN}/text.txt ${DATA_PRETRAIN}/term.txt ${DATA_PRETRAIN}/nonterm.txt \
    --binarize-trees --ignore-errors --error-log=${WDIR}/data/error_trees_silver.psd --limit 1000000


python ${EXPORTER} dump-term-schema ${DATA_BASE}/term_schema.json
python ${EXPORTER} dump-nonterm-schema ${DATA_BASE}/nonterm_schema.json

mkdir -p $DATA_BIN/tmp
for SUBSET_NAME in debug 'test' valid pretrain train ; do
    python -m greynirseq.nicenlp.utils.constituency.preprocess_labelled_spans \
        --task parser \
        --destdir $DATA_BIN/tmp  \
        --trainpref "${DATA_BASE}/${SUBSET_NAME}/nonterm" \
        --nonterm_schema ${DATA_BASE}/nonterm_schema.json  \
        --nonterm_suffix txt  \
        --only-source
    mv ${DATA_BIN}/tmp/train.nonterm.bin ${DATA_BIN}/${SUBSET_NAME}.nonterm.bin
    mv ${DATA_BIN}/tmp/train.nonterm.idx ${DATA_BIN}/${SUBSET_NAME}.nonterm.idx
done

## binarize terminals
#python preprocess_labelled_spans.py \
#    --task multi_span_prediction \
#    --trainpref dry/train \
#    --validpref dry/valid \
#    --testpref dry/test \
#    --term_suffix term.txt  \
#    --only-source

TMP_DIR=/tmp/data-bpe
mkdir -p /tmp/data-bpe
ICEBERT=/data/models/icebert/icebert-first-rmh_different_vocab/icebert-base-36k
for SUBSET_NAME in debug 'test' valid train pretrain ; do
    ### encode text files into BPE codes (subwords
    python ${WDIR}/scripts/encode_data.py \
        "${DATA_BASE}/${SUBSET_NAME}/text.txt" \
        "$TMP_DIR/${SUBSET_NAME}.bpe"
    ### binarize BPE text files
    fairseq-preprocess \
        --only-source \
        --srcdict $ICEBERT/icebert-bpe-freqs.txt \
        --trainpref "$TMP_DIR/${SUBSET_NAME}.bpe" \
        --destdir ${DATA_BIN} \
        --workers 30
    mv ${DATA_BIN}/train.bin ${DATA_BIN}/${SUBSET_NAME}.text.bin
    mv ${DATA_BIN}/train.idx ${DATA_BIN}/${SUBSET_NAME}.text.idx
done

# make separate DATA_BIN for pretraining time by link already made files
for SUFFIX in bin idx ; do
    for DATA_TYPE in nonterm text ; do
        for SUBSET_NAME in 'test' valid ; do
            ln -s ${DATA_BIN}/${SUBSET_NAME}.${DATA_TYPE}.${SUFFIX} \
                ${DATA_BIN_PRETRAIN}/${SUBSET_NAME}.${DATA_TYPE}.${SUFFIX}
        done
        ln -s ${DATA_BIN}/pretrain.${DATA_TYPE}.${SUFFIX} \
            ${DATA_BIN_PRETRAIN}//train.${DATA_TYPE}.${SUFFIX}
    done
done
