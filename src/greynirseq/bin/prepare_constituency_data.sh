#!/bin/sh

#SCRIPT_NAME=$0
SCRIPT_DIR=$(realpath "$(dirname "$0")")
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
echo "Using project directory: $PROJECT_DIR"
echo ""

# RAW_DIR="$HOME/sandbox/simple_trees/data"
# ENCODED_DIR="$HOME/sandbox/simple_trees/fresh"
# SPLITS_DIR="$HOME/sandbox/simple_trees/splits"
CORPUS_DIR="/data/datasets/greynir_corpus"
# RAW_DIR="/data/datasets/greynir_corpus/jsonl"
ENCODED_DIR="/data/datasets/greynir_corpus/raw"
SPLITS_DIR="/data/datasets/greynir_corpus/splits"
DATA_BIN="$PROJECT_DIR/data-bin"
# ICEBERT=/data/models/icebert-base-36k
# FAIRSEQ=$HOME/github/fairseq
TMP_DIR=/tmp/icebert_tmp

# exit on any error
set -e

# curl https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/25/greynir_corpus.zip \
#      --output $CORPUS_DIR/greynir_corpus.zip --create-dirs
# curl https://repository.clarin.is/repository/xmlui/bitstream/handle/20.500.12537/25/gold.zip \
#      --output $CORPUS_DIR/gold.zip
# unzip $CORPUS_DIR/greynir_corpus.zip -d $CORPUS_DIR/cfg
# unzip $CORPUS_DIR/gold.zip -d $CORPUS_DIR

if [ -f  $CORPUS_DIR/cfg/greynircorpus.psd ]; then
    mv $CORPUS_DIR/cfg/greynircorpus.psd $CORPUS_DIR/cfg/greynir_corpus.psd
fi
if [ ! -f $CORPUS_DIR/gold/all.gld ]; then
    for GLDFILE in "$CORPUS_DIR"/gold/greynir_corpus_0*.gld ; do
        cat "$GLDFILE" >> "$CORPUS_DIR"/gold/all.gld
        echo "" >> "$CORPUS_DIR"/gold/all.gld
    done
fi

# echo "Fixing errors in corpus"
# sed -i 's/"\(janúar\|febrúar\|mars\|apríl\|maí\|júní\|júlí\|ágúst\|september\|október\|nóvember\|desember\)"/dagsafs/' $CORPUS_DIR/cfg/greynir_corpus.psd
# sed -i 's|\(^( (META (ID-CORPUS a4ea75ee-21df-11e7-aa57-04014c605401.44\)|\n \1|' $CORPUS_DIR/gold/all.gld
# sed -i 's|\(^( (META (ID-CORPUS 04e4bde6-7f71-11e6-a2c6-04014c605401.8)\)|\n \1|' $CORPUS_DIR/gold/all.gld

# sed -i 's|\(http://www.mbl.is/frettir/innlent/2016/02/22/kaerdur_fyrir_fjardratt_og_misneytingu/)$\)|\1)|' $CORPUS_DIR/gold/all.gld
# sed -i 's|\(http://www.visir.is/g/2018181018703/umfjollun-fh-selfoss-27-30-selfoss-hafdi-betur-i-toppslagnum)$\)|\1)|' $CORPUS_DIR/gold/all.gld
# sed -i 's|\(http://www.mbl.is/frettir/innlent/2016/01/09/kari_stefansson_i_jotunmod/)$\)|\1)|' $CORPUS_DIR/gold/all.gld
# sed -i 's|\(http://www.ruv.is/frett/viti-a-vellinum-og-utan-hans)$\)|\1)|' $CORPUS_DIR/gold/all.gld
# sed -i 's|\(http://www.ruv.is/frett/arnfridur-ekki-vanhaef)$\)|\1)|' $CORPUS_DIR/gold/all.gld
# sed -i 's|\(http://www.visir.is/g/2017170619146/launakrafa-upp-a-tvaer-milljonir-a-islensk-sjalfbodalidasamtok)$\)|\1)|' $CORPUS_DIR/gold/all.gld

# cat greynir_corpus.psd | grep -C3 '"(janúar|febrúar|mars|apríl|maí|júní|júlí|ágúst|september|október|nóvember|desember)"' | less
# Viðræður um úrgöngu Breta úr Evrópusambandinu hófust fyrir einu og hálfu ári og strax þá var horft til morgundagsins ,  sautjánda október .
# Fundurinn verður haldinn fjórða júní í sumar .
# Núgildandi samningur Útlendingastofnunar við Reykjanesbæ kveður á um að stofnunin greiði Reykjanesbæ daggjald að upphæð 7.500 krónur á sólarhring fyrir hvern hælisleitanda , auk fastagjalds sem nemur um 11,5 milljónum króna , sem á að standa straum af launa- og rekstrarkostnaði .
# Íþróttasamband Íslands hefur kallað eftir því að íþróttafélög og sérsambönd sæki um styrk vegna fjárhagslegs tjóns af völdum COVID - 19 .
# (S0
#  (S-HEADING
#   (NP
#    (no_et_nf_hk frame (lemma frame))
#    (NP-POSS
#     (no_et_ef_hk src (lemma src))
#     (NP-POSS
#      (eight="100%"width="100%"scrolling="no"frameborder="0"seamless"
#             http://visir.is/section/MEDIA&template=iplayer&fileid=CLP47007
#             (lemma
#               http://visir.is/section/MEDIA&template=iplayer&fileid=CLP47007)))))))
#
# (ADVP-TIMESTAMP-REL (P (fs_þf í (lemma í)))
#                     (NP (dagsafs janúar (lemma janúar))))



mkdir -p $TMP_DIR $ENCODED_DIR/gold $ENCODED_DIR/cfg $SPLITS_DIR/gold $SPLITS_DIR/cfg
cd "$PROJECT_DIR"

# cythonize "$PROJECT_DIR/nicenlp/utils/greynir/tree_distance.pyx"

echo "Generating nonterm_schema.json"
python -m nicenlp.utils.greynir.prep_greynir dump-nonterm-schema "$DATA_BIN/nonterm_schema.json"
echo ""
echo "Generating term_schema.json"
python -m nicenlp.utils.greynir.prep_greynir dump-term-schema "$DATA_BIN/term_schema.json"
echo ""


# DEBUG GOLD CORPUS - NEW TREE GENERATOR
# DATASET=gold
# # RAW_PATH="$CORPUS_DIR/gold/all.gld"
# RAW_PATH="$CORPUS_DIR/cfg/greynir_corpus.psd"
# # RAW_PATH="$RAW_DIR/greynir_corpus.shuf.jsonl.5"
# python -m nicenlp.utils.greynir.prep_greynir export "$RAW_PATH" \
#        "$ENCODED_DIR/$DATASET/text.txt" "$ENCODED_DIR/$DATASET/term.txt" "$ENCODED_DIR/$DATASET/nonterm.txt" \
#        --include-null --simplified --anno
# # head -1 "$ENCODED_DIR/$DATASET/text.txt"
# # head -1 "$ENCODED_DIR/$DATASET/nonterm.txt"
# # head -1 "$ENCODED_DIR/$DATASET/term.txt"  && echo ""
# # python "$PROJECT_DIR/bin/gen_constituency_splits.py" \
# #        --valid-size=250 --test-size=250 --train-size=-1 \
# #        "$ENCODED_DIR/$DATASET" "$SPLITS_DIR/$DATASET"
# # echo ""


# # RAW_PATH="$RAW_DIR/greynir_corpus.shuf.jsonl"
# RAW_PATH="$HOME/sandbox/simple_trees/data/greynir_corpus.shuf.jsonl"
# # RAW_PATH="$RAW_DIR/greynir_corpus.shuf.jsonl.5"
# python -m nicenlp.utils.greynir.prep_greynir export "$RAW_PATH" \
#        "$ENCODED_DIR/gold/text.txt" "$ENCODED_DIR/gold/term.txt" "$ENCODED_DIR/gold/nonterm.txt" \
#        --include-null --simplified --anno
# head -1 "$ENCODED_DIR/gold/text.txt"
# head -1 "$ENCODED_DIR/gold/nonterm.txt"
# head -1 "$ENCODED_DIR/gold/term.txt"  && echo ""
# python "$PROJECT_DIR/bin/gen_constituency_splits.py" \
#        --valid-size=250 --test-size=250 --train-size=-1 \
#        "$ENCODED_DIR/gold" "$SPLITS_DIR/gold"
# echo ""

# takes approx 90 minutes single threaded
# # RAW_PATH="$RAW_DIR/sampled_simple_trees_10k_articles.deduped.shuf.jsonl"
# # RAW_PATH="$RAW_DIR/small_sample.jsonl"
# RAW_PATH="$CORPUS_DIR/cfg/greynir_corpus.psd"
# python -m nicenlp.utils.greynir.prep_greynir export "$RAW_PATH" \
#        "$ENCODED_DIR/cfg/text.txt" "$ENCODED_DIR/cfg/term.txt" "$ENCODED_DIR/cfg/nonterm.txt" \
#        --include-null --simplified --use-new
# # --include-null --simplified
# head -1 "$ENCODED_DIR/cfg/text.txt"
# head -1 "$ENCODED_DIR/cfg/nonterm.txt"
# head -1 "$ENCODED_DIR/cfg/term.txt"  && echo ""
# python "$PROJECT_DIR/bin/gen_constituency_splits.py" \
#        --valid-size=1000 --test-size=1000 --train-size=-1 \
#        "$ENCODED_DIR/cfg" "$SPLITS_DIR/cfg"
# echo ""

# # generate dictionary for terminals
# cat "$ENCODED_DIR"/*/term.txt \
#     | tr ' ' '\n' \
#     | sort | uniq -c | sort -rn \
#     | sed 's/^ *\([^ ]\+\) \([^$ ]\+\)/\2 \1/' \
#           > "$DATA_BIN/dict_term.txt"

# # generate label coverage for collapsed unary nonterminal branches
# cat "$ENCODED_DIR/gold/nonterm.txt" "$ENCODED_DIR/cfg/nonterm.txt" \
#     | grep "[^ ]\+>[^ ]\+" -o \
#     | sort | uniq -c > $ENCODED_DIR/unary_branches_01.txt

# for DATASET in gold cfg; do
# # for DATASET in gold gold; do

#     ### encode and binarize nonterminals
#     cd $PROJECT_DIR
#     python -m nicenlp.utils.preprocess_labelled_spans \
#         --task multi_span_prediction \
#         --only-source \
#         --nonterm_suffix nonterm.txt  \
#         --trainpref "$SPLITS_DIR/$DATASET/train" \
#         --validpref "$SPLITS_DIR/$DATASET/valid" \
#         --testpref "$SPLITS_DIR/$DATASET/test" \
#         --nonterm_schema "$DATA_BIN/nonterm_schema.json"  \
#         --destdir "$DATA_BIN/$DATASET/nonterm"
#     echo ""

#     # binarize terminals
#     fairseq-preprocess \
#                --only-source \
#                --srcdict "$DATA_BIN/dict_term.txt" \
#                --trainpref "$SPLITS_DIR/$DATASET/train.term.txt" \
#                --validpref "$SPLITS_DIR/$DATASET/valid.term.txt" \
#                --testpref "$SPLITS_DIR/$DATASET/test.term.txt" \
#                --destdir "$DATA_BIN/$DATASET/term" \
#                --workers 30
#     echo ""

#     # encode text files into BPE codes (subwords)
#     # cd $PROJECT_DIR
#     cd "$FAIRSEQ"
#     for SPLIT in train valid test; do \
#        # symlinking fairseq/examples to local dir is necessary
#        python -m examples.roberta.multiprocessing_bpe_encoder \
#                --encoder-json $ICEBERT/icebert-bpe-vocab.json \
#                --vocab-bpe $ICEBERT/icebert-bpe-merges.txt \
#                --inputs "$SPLITS_DIR/$DATASET/$SPLIT.text.txt" \
#                --outputs "$TMP_DIR/$SPLIT.$DATASET.bpe" \
#                --keep-empty \
#                --workers 30; \
#     done
#     echo ""

#     # binarize text files
#     fairseq-preprocess \
#            --only-source \
#            --srcdict $ICEBERT/icebert-bpe-freqs.txt \
#            --trainpref "$TMP_DIR/train.$DATASET.bpe" \
#            --validpref "$TMP_DIR/valid.$DATASET.bpe" \
#            --testpref "$TMP_DIR/test.$DATASET.bpe" \
#            --destdir "$DATA_BIN/$DATASET/text" \
#            --workers 30
#     echo ""

# done

mkdir -p "$DATA_BIN/sym"
mkdir -p "$DATA_BIN/sym-fine"
rm -f "$DATA_BIN/sym/"*
rm -f "$DATA_BIN/sym-fine/"*
ln -s "$DATA_BIN/nonterm_schema.json" "$DATA_BIN/sym/nonterm_schema.json"
ln -s "$DATA_BIN/term_schema.json" "$DATA_BIN/sym/term_schema.json"
ln -s "$DATA_BIN/dict.txt" "$DATA_BIN/sym/dict.txt"
ln -s "$DATA_BIN/dict_term.txt" "$DATA_BIN/sym/dict_term.txt"

ln -s "$DATA_BIN/nonterm_schema.json" "$DATA_BIN/sym-fine/nonterm_schema.json"
ln -s "$DATA_BIN/term_schema.json" "$DATA_BIN/sym-fine/term_schema.json"
ln -s "$DATA_BIN/dict.txt" "$DATA_BIN/sym-fine/dict.txt"
ln -s "$DATA_BIN/dict_term.txt" "$DATA_BIN/sym-fine/dict_term.txt"
for SUFFIX in bin idx; do
    ln -s "$DATA_BIN/cfg/nonterm/train.nonterm.$SUFFIX"  "$DATA_BIN/sym/train.nonterm.$SUFFIX"
    ln -s "$DATA_BIN/cfg/term/train.$SUFFIX"             "$DATA_BIN/sym/train.term.$SUFFIX"
    ln -s "$DATA_BIN/cfg/text/train.$SUFFIX"             "$DATA_BIN/sym/train.$SUFFIX"

    ln -s "$DATA_BIN/cfg/nonterm/valid.nonterm.$SUFFIX"  "$DATA_BIN/sym/valid.cfg.nonterm.$SUFFIX"
    ln -s "$DATA_BIN/cfg/term/valid.$SUFFIX"             "$DATA_BIN/sym/valid.cfg.term.$SUFFIX"
    ln -s "$DATA_BIN/cfg/text/valid.$SUFFIX"             "$DATA_BIN/sym/valid.cfg.$SUFFIX"

    ln -s "$DATA_BIN/gold/nonterm/valid.nonterm.$SUFFIX" "$DATA_BIN/sym/valid.gold.nonterm.$SUFFIX"
    ln -s "$DATA_BIN/gold/term/valid.$SUFFIX"            "$DATA_BIN/sym/valid.gold.term.$SUFFIX"
    ln -s "$DATA_BIN/gold/text/valid.$SUFFIX"            "$DATA_BIN/sym/valid.gold.$SUFFIX"

    # for fine-tuning on gold
    ln -s "$DATA_BIN/cfg/nonterm/valid.nonterm.$SUFFIX"  "$DATA_BIN/sym-fine/valid.cfg.nonterm.$SUFFIX"
    ln -s "$DATA_BIN/cfg/term/valid.$SUFFIX"             "$DATA_BIN/sym-fine/valid.cfg.term.$SUFFIX"
    ln -s "$DATA_BIN/cfg/text/valid.$SUFFIX"             "$DATA_BIN/sym-fine/valid.cfg.$SUFFIX"

    ln -s "$DATA_BIN/gold/nonterm/valid.nonterm.$SUFFIX" "$DATA_BIN/sym-fine/valid.gold.nonterm.$SUFFIX"
    ln -s "$DATA_BIN/gold/term/valid.$SUFFIX"            "$DATA_BIN/sym-fine/valid.gold.term.$SUFFIX"
    ln -s "$DATA_BIN/gold/text/valid.$SUFFIX"            "$DATA_BIN/sym-fine/valid.gold.$SUFFIX"

    ln -s "$DATA_BIN/gold/nonterm/train.nonterm.$SUFFIX" "$DATA_BIN/sym-fine/train.nonterm.$SUFFIX"
    ln -s "$DATA_BIN/gold/term/train.$SUFFIX"            "$DATA_BIN/sym-fine/train.term.$SUFFIX"
    ln -s "$DATA_BIN/gold/text/train.$SUFFIX"            "$DATA_BIN/sym-fine/train.$SUFFIX"
done
