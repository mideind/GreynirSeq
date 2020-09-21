FROM tiangolo/uvicorn-gunicorn:python3.7
ENV MODULE_NAME serve
ENV APP_MODULE serve.serve_all:app
COPY ./ /all
RUN cd /all && pip install .
COPY ./src/greynirseq /app
COPY ./requirements.txt /app/requirements.txt
WORKDIR /app
RUN apt update
RUN apt install -y python3-numpy build-essential
ENV CFLAGS="-I /usr/local/lib/python3.7/site-packages/numpy/core/include $CFLAGS"
RUN pip install -r requirements.txt
RUN cython nicenlp/utils/greynir/tree_dist.pyx
