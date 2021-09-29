from pytorch/pytorch
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get install -y g++ python3-dev zlib1g-dev git libxml2-dev libxslt1-dev make cmake libffi-dev
RUN pip install cython icecream tqdm nltk

WORKDIR /github
RUN git clone --single-branch --branch master https://github.com/mideind/greynirpackage greynirpackage
WORKDIR /github/greynirpackage
RUN git checkout 71f811bb9f823c2f977e82312d03e57c21b80f50
RUN pip install .

WORKDIR /github
RUN git clone --single-branch --branch main https://github.com/pytorch/fairseq fairseq
WORKDIR /github/fairseq
RUN git submodule sync
RUN git submodule update --init --recursive
RUN git fetch && git checkout 5a75b079bf8911a327940c28794608e003a9fa52
RUN python setup.py build_ext --inplace
RUN python setup.py install

WORKDIR /github
RUN git clone --single-branch --branch dev/parser https://github.com/mideind/greynirseq greynirseq
WORKDIR /github/greynirseq
RUN git fetch && git checkout 9c415a63d6967844d3b5aa24e84f38b57a9855a6
RUN python setup.py build_ext --inplace
RUN python setup.py install

WORKDIR /data
WORKDIR /model
# Assumes the existance of:
# - /model/infer_file.sh
# - /model/checkpoint.pt
# - /model/data/dict.txt
# - /model/data/icebert-bpe-freqs.txt
# - /model/data/icebert-bpe-merges.txt
# - /model/data/icebert-bpe-vocab.json
# - /model/data/nonterm_schema.json
# - /model/data/term_schema.json

COPY model /model
ENTRYPOINT /model/infer_file.sh
