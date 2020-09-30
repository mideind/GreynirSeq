#!/usr/bin/env bash
pwd=$(pwd)

docker kill greynirseq
docker rm greynirseq

# Monkey patch..

docker run \
       -v $pwd/gpt2_bpe_utils.py:/usr/local/lib/python3.7/site-packages/fairseq/data/encoders/gpt2_bpe_utils.py \
       -v /data/models:/data/models \
       -v $pwd/start.sh:/start.sh \
       -v $pwd/../src:/app \
       -p 8001:8001 \
       -t -i --name greynirseq greynirseq bash
