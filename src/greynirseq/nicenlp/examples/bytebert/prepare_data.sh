#!/usr/bin/env bash

# install branch: feature/nondestructive-tokenizer from github.com/mideind/tokenizer

# train/valid data is arbitrary for debug purpopes /data/datasets/rmh2017/tsv/MIM/jonas/2016.tsv
pushd data/
python ../make_token_offsets.py train.txt train.offsets
python ../make_token_offsets.py valid.txt valid.offsets
popd
