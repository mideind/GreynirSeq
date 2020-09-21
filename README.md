# GreynirSeq

GreynirSeq is a natural language parsing toolkit for Icelandic. It is under active development and is in its early stages.

The modeling part of GreynirSeq is built on top of the excellent [fairseq](https://github.com/pytorch/fairseq) from Facebook (which is built on top of pytorch).

GreynirSeq is licensed under the MIT license unless otherwise stated.

### What's on the horizon?
* Cleanup and configuration of data / model loading -- currently unavailable for download
* More fine tuning tasks for Icelandic
* General cleanup and CI config
* Icelandic - English translation example
* Improved documentation and examples for training and preprocessing pipelines

### What's new?
* This repository!
* An Icelandic RoBERTa model, **IceBERT** with NER, POS tagging and constituency parsing fine tuning options.
* Simple Docker setup to serve models
* NER pre and post processing for NMT corpuses

## Installation

To install GreynirSeq in development mode add the `-e` as shown below

``` bash
pip install -e .
```

## Docker

To build the container

``` bash
docker build -t greynirseq .
```

Assuming you have the models and other files necessary you can then run `serve/prod.sh` or a variation therof. You can serve the models over http using the container or run experiments within it.
