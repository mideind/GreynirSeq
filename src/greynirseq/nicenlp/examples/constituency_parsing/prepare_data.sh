#!/usr/bin/env bash

# export parse trees as labelled spans
#python github-greynirseq/src/greynirseq/nicenlp/utils/constituency/prep_greynir_new.py export gc01.gld throwaway/text throwaway/term throwaway/nonterm --binarize-trees
WDIR=/data/scratch/haukur/parser
EXPORTER="${WDIR}/github-greynirseq/src/greynirseq/nicenlp/utils/constituency/prep_greynir.py"
SRC_DICT=/data/models/icebert/icebert-first-rmh_different_vocab/icebert-base-36k/dict.txt
DATA_BASE="${WDIR}/data"

DATA_DEBUG="${DATA_BASE}/debug"
DATA_TEST="${DATA_BASE}/test"
DATA_SILVER="${DATA_BASE}/silver"
DATA_COPPER="${DATA_BASE}/copper"
DATA_VALID="${DATA_BASE}/valid"
DATA_FINETUNE="${DATA_BASE}/finetune"
DATA_PRETRAIN="${DATA_BASE}/pretrain"

DATA_BIN="${DATA_BASE}/bin"
DATA_BIN_FINETUNE="${DATA_BASE}/bin-finetune"
DATA_BIN_PRETRAIN="${DATA_BASE}/bin-pretrain"

mkdir -p $DATA_DEBUG $DATA_TEST $DATA_SILVER $DATA_COPPER \
    $DATA_FINETUNE $DATA_VALID $DATA_BIN \
    $DATA_BIN_PRETRAIN $DATA_BIN_FINETUNE

DATA_DEBUG_PSD="${WDIR}/greynircorpus/devset/psd/greynir_corpus_00021.gld"
DATA_SILVER_PSD="${WDIR}/greynircorpus/psd/silver/silver"

DATA_FINETUNE_FILENAMES="${DATA_BASE}/filepaths.devset_gold.finetune.txt"
DATA_VALID_GOLD_FILESNAMES="${DATA_BASE}/filepaths.devset_gold.valid.txt"

### we want a fixed seed for the 'shuf' command
get_fixed_random_byte_stream()
{
  # Obtained from: https://stackoverflow.com/questions/60266215/shuffle-output-of-find-with-fixed-seed?noredirect=1&lq=1
  openssl enc -aes-256-ctr -pass pass:"$1" -nosalt </dev/zero 2>/dev/null
}

set -ex

### data is kept in many separate .gld files, we need to merge them
rm -f "${DATA_VALID_GOLD_FILESNAMES}"
rm -f "${DATA_FINETUNE}/text.txt" "${DATA_FINETUNE}/term.txt" "${DATA_FINETUNE}/nonterm.txt"
if [ ! -f "${DATA_VALID_GOLD_FILESNAMES}" ] ; then
    # total number of files 453
    NUM_DEV=100
    SEED=1
    find "${WDIR}/greynircorpus/devset/psd/" -type f -name "*.gld" | sort | shuf --random-source=<(get_fixed_random_byte_stream $SEED) \
        | head -n "${NUM_DEV}" | sort > "${DATA_VALID_GOLD_FILESNAMES}"
    find "${WDIR}/greynircorpus/devset/psd/" -type f -name "*.gld" | sort | shuf --random-source=<(get_fixed_random_byte_stream $SEED) \
        | tail -n +$((NUM_DEV + 1)) | sort > "${DATA_FINETUNE_FILENAMES}"
fi

##############################################################################################
###############    Prepare text, nonterm and term data from .psd fiels    ####################
##############################################################################################

### Prepare small dataset for debug purposes
python "${EXPORTER}" export "${DATA_DEBUG_PSD}" \
    "${DATA_DEBUG}/text.txt" "${DATA_DEBUG}/term.txt" "${DATA_DEBUG}/nonterm.txt" \
    --binarize-trees --ignore-errors --error-log="${WDIR}/data/error_trees_gold.psd" --append-errors

### Prepare finetuning set (using part of gold devset data)
while read -r FILEPATH ; do
    python "${EXPORTER}" export "${FILEPATH}" \
        "${DATA_FINETUNE}/text.txt" "${DATA_FINETUNE}/term.txt" "${DATA_FINETUNE}/nonterm.txt" \
        --binarize-trees --append --ignore-errors --error-log="${WDIR}/data/error_trees.finetune.psd" --append-errors
done < "${DATA_FINETUNE_FILENAMES}"

## Prepare gold validation set
while read -r FILEPATH ; do
    python "${EXPORTER}" export "${FILEPATH}" \
        "${DATA_VALID}/text.txt" "${DATA_VALID}/term.txt" "${DATA_VALID}/nonterm.txt" \
        --binarize-trees --append --ignore-errors --error-log="${WDIR}/data/error_trees.valid.psd" --append-errors
done < "${DATA_VALID_GOLD_FILESNAMES}"

# Prepare test set (using the gold testset)
rm -f "${DATA_TEST}/text" "${DATA_TEST}/nonterm" "${DATA_TEST}/term"
find "${WDIR}/greynircorpus/testset/psd/" -type f -name "*.gld" | while read -r FILEPATH ; do
    python "${EXPORTER}" export "${FILEPATH}" \
        "${DATA_TEST}/text.txt"  "${DATA_TEST}/term.txt" "${DATA_TEST}/nonterm.txt" \
        --binarize-trees --append --ignore-errors --error-log="${WDIR}/data/error_trees.test.psd" --append-errors
done

# Prepare silver set (selected data from cfg parser)
python "${EXPORTER}" export "${DATA_SILVER_PSD}" \
    "${DATA_SILVER}/text.txt" "${DATA_SILVER}/term.txt" "${DATA_SILVER}/nonterm.txt" \
    --binarize-trees --ignore-errors --error-log="${WDIR}/data/error_trees.silver.psd" --limit 1000000

# Prepare copper set (rest of data from cfg parser)
mkdir -p "${DATA_BASE}/data/copper"
rm -f "${WDIR}/data/copper/nonterm.txt" "${WDIR}/data/copper/term.txt" > "${WDIR}/data/copper/text.txt"
for IDX in {01..10} ; do
    python "${EXPORTER}" export "${WDIR}/greynircorpus/psd/copper/copper${IDX}" \
        "${DATA_COPPER}/text.txt" "${DATA_COPPER}/term.txt" "${DATA_COPPER}/nonterm.txt" \
        --append --ignore-errors --error-log="${DATA_BASE}/error_trees.copper.psd" --append-errors
done

##############################################################################################

# Combine silver and copper for the pretraining phase
for DATA_TYPE in text term nonterm ; do
    rm -f "${DATA_PRETRAIN}/${DATA_TYPE}.txt"
    cat "${DATA_COPPER}/${DATA_TYPE}.txt" >> "${DATA_PRETRAIN}/${DATA_TYPE}.txt"
    cat "${DATA_SILVER}/${DATA_TYPE}.txt" >> "${DATA_PRETRAIN}/${DATA_TYPE}.txt"
done

##############################################################################################
#################    Binarizing text, nonterm and term files    ##############################
##############################################################################################

python "${EXPORTER}" dump-term-schema "${DATA_BASE}/term_schema.json"
python "${EXPORTER}" dump-nonterm-schema "${DATA_BASE}/nonterm_schema.json"

TMP_DIR="${DATA_BIN}/tmp"

### Binarize nonterminal spans
mkdir -p "${TMP_DIR}"
#for SUBSET_NAME in debug 'test' valid finetune pretrain ; do
for SUBSET_NAME in debug 'test' valid finetune ; do
    python -m greynirseq.nicenlp.utils.constituency.preprocess_labelled_spans \
        --task parser \
        --destdir "${TMP_DIR}"  \
        --trainpref "${DATA_BASE}/${SUBSET_NAME}/nonterm" \
        --nonterm_schema "${DATA_BASE}/nonterm_schema.json"  \
        --nonterm_suffix txt  \
        --only-source
    mv "${TMP_DIR}/train.nonterm.bin" "${DATA_BIN}/${SUBSET_NAME}.nonterm.bin"
    mv "${TMP_DIR}/train.nonterm.idx" "${DATA_BIN}/${SUBSET_NAME}.nonterm.idx"
done
rm -f "${TMP_DIR}"/*

## binarize terminals
python preprocess_labelled_spans.py \
    --task multi_span_prediction \
    --trainpref dry/train \
    --validpref dry/valid \
    --testpref dry/test \
    --term_suffix term.txt  \
    --only-source

### Binarize text files
TMP_DIR=/tmp/data-bpe
mkdir -p /tmp/data-bpe
for SUBSET_NAME in debug 'test' valid finetune pretrain ; do
    ### encode text files into BPE codes (subwords
    #python "${WDIR}/scripts/encode_data.py" \
    #    "${DATA_BASE}/${SUBSET_NAME}/text.txt" \
    #    "${TMP_DIR}/${SUBSET_NAME}.bpe"
    ### binarize BPE text files
    fairseq-preprocess \
        --only-source \
        --srcdict "${SRC_DICT}" \
        --trainpref "$TMP_DIR/${SUBSET_NAME}.bpe" \
        --destdir "${TMP_DIR}" \
        --workers 30
    mv "${TMP_DIR}/train.bin" "${DATA_BIN}/${SUBSET_NAME}.text.bin"
    mv "${TMP_DIR}/train.idx" "${DATA_BIN}/${SUBSET_NAME}.text.idx"
done

cp "${SRC_DICT}" "${DATA_BIN}/dict.txt"
ln -s "${DATA_BIN}/dict.txt" "${DATA_BIN_FINETUNE}/dict.txt"
ln -s "${DATA_BIN}/dict.txt" "${DATA_BIN_PRETRAIN}/dict.txt"
# make separate DATA_BIN for pretraining and finetuning by linking already made files
for SUFFIX in bin idx ; do
    for DATA_TYPE in nonterm text ; do
        # test and valid are same for pretraining and finetuning
        for SUBSET_NAME in 'test' valid ; do
            ln -s "${DATA_BIN}/${SUBSET_NAME}.${DATA_TYPE}.${SUFFIX}" \
                "${DATA_BIN_PRETRAIN}/${SUBSET_NAME}.${DATA_TYPE}.${SUFFIX}"
        done
        # link training set for finetuning and pretraining
        ln -s "${DATA_BIN}/finetune.${DATA_TYPE}.${SUFFIX}" \
            "${DATA_BIN_FINETUNE}/train.${DATA_TYPE}.${SUFFIX}"
        ln -s "${DATA_BIN}/pretrain.${DATA_TYPE}.${SUFFIX}" \
            "${DATA_BIN_PRETRAIN}/train.${DATA_TYPE}.${SUFFIX}"
    done
done
