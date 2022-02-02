FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update \
    && apt-get install --no-install-recommends -y \
        g++=4:9.3.0-1ubuntu2 \
        python3-dev=3.8.2-0ubuntu2 \
        zlib1g-dev=1:1.2.11.dfsg-2ubuntu1.2 \
        git=1:2.25.1-1ubuntu3.2 \
        libxml2-dev=2.9.10+dfsg-5ubuntu0.20.04.1 \
        libxslt1-dev=1.1.34-4 \
        make=4.2.1-1.2 \
        cmake=3.16.3-1ubuntu1 \
        libffi-dev=3.3-4 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir cython==0.29.24 tqdm==4.62.3 nltk==3.6.3

WORKDIR /github
RUN git clone --single-branch --branch master https://github.com/mideind/greynirpackage greynirpackage
WORKDIR /github/greynirpackage
RUN git checkout 71f811bb9f823c2f977e82312d03e57c21b80f50 \
    && pip install --no-cache-dir .

WORKDIR /github
RUN git clone --single-branch --branch main https://github.com/pytorch/fairseq fairseq
WORKDIR /github/fairseq
RUN git submodule sync \
    && git submodule update --init --recursive \
    && git fetch \
    && git checkout 5a75b079bf8911a327940c28794608e003a9fa52 \
    && python setup.py build_ext --inplace \
    && python setup.py install

WORKDIR /github
RUN git clone --single-branch --branch parser-container https://github.com/mideind/greynirseq greynirseq
WORKDIR /github/greynirseq
RUN git fetch \
    && python setup.py build_ext --inplace \
    && python setup.py install

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
ENTRYPOINT ["/bin/bash", "/model/infer_file.sh"]
