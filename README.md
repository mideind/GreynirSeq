<img src="assets/greynir-logo-large.png" alt="Greynir" width="200" height="200" align="right" style="margin-left:20px; margin-bottom: 20px;">

# GreynirSeq

GreynirSeq is a natural language parsing toolkit for Icelandic focused on sequence modeling with neural networks. It is under active development and is in its early stages.

The modeling part (nicenlp) of GreynirSeq is built on top of the excellent [fairseq](https://github.com/pytorch/fairseq) from Facebook (which is built on top of pytorch).

GreynirSeq is licensed under the GNU GPLv3 license unless otherwise stated at the top of a file.

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
* NER pre and post processing for NMT corpora

## Neural Icelandic Language Processing - NIceNLP

### IceBERT

IceBERT is an Icelandic language model.

The following fine tuning tasks are available

0. Fill Mask - IceBERT without fine tuning
1. [POS tagging](src/greynirseq/nicenlp/examples/pos/README.md)
2. [NER tagging](src/greynirseq/nicenlp/examples/ner/README.md)
3. Constituency tagging

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
