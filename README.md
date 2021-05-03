[![superlinter](https://github.com/mideind/greynirseq/actions/workflows/superlinter.yml/badge.svg)]() [![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

---

<img src="assets/greynir-logo-large.png" alt="Greynir" width="200" height="200" align="right" style="margin-left:20px; margin-bottom: 20px;">

# GreynirSeq

GreynirSeq is a natural language parsing toolkit for Icelandic focused on sequence modeling with neural networks. It is under active development and is in its early stages.

The modeling part (nicenlp) of GreynirSeq is built on top of the excellent [fairseq](https://github.com/pytorch/fairseq) from Facebook (which is built on top of pytorch).

GreynirSeq is licensed under the GNU AFFERO GPLv3 license unless otherwise stated at the top of a file.

**What's new?**
* This repository!
* An Icelandic RoBERTa model, **IceBERT** finetuned for NER and POS tagging.

**What's on the horizon?**
* More fine tuning tasks for Icelandic, constituency parsing and grammatical error detection
* Icelandic - English translation example

---

Be aware that usage of the CLI or otherwise downloading model files will result in downloading of **gigabytes** of data.

## Features

### TL;DR give me the CLI

The `greynirseq` CLI interface can be used to run state-of-the-art POS and NER tagging for Icelandic. Run `pip install greynirseq && greynirseq -h` to see what options are available. Input is accepted from file containing a single [tokenized](https://github.com/mideind/Tokenizer) sentence per line, or from stdin.
#### POS

``` bash
❯ pip install greynirseq
❯ echo "Systurnar Guðrún og Monique átu einar um jólin á McDonalds ." | greynirseq pos --input -

nvfng nven-s c n---s sfg3fþ lvfnsf af nhfog af n----s pl
```

#### NER

``` bash
❯ pip install greynirseq
❯ echo "Systurnar Guðrún og Monique átu einar um jólin á McDonalds ." | greynirseq ner --input -

O B-Person O B-Person O O O O O B-Organization O
```

### Neural Icelandic Language Processing - NIceNLP

IceBERT is an Icelandic BERT-based (RoBERTa) language model that is suitable for fine tuning on downstream tasks.

The following fine tuning tasks are available both through the `greynirseq` CLI and for loading programmatically.

1. [POS tagging](https://github.com/mideind/GreynirSeq/blob/main/src/greynirseq/nicenlp/examples/pos/README.md)
2. [NER tagging](https://github.com/mideind/GreynirSeq/blob/main/src/greynirseq/nicenlp/examples/ner/README.md)

## Installation

### From python packaging index

In a suitable virtual environment

```bash
pip install greynirseq
```

### Development

To install GreynirSeq in development mode we recommend using poetry as shown below

```bash
pip install poetry && poetry install
```

## Development

### Linting

All code is checked with [Super-Linter](https://github.com/github/super-linter) in a *GitHub Action*, we recommend running it locally before pushing

```bash
docker run -e RUN_LOCAL=true -v /path/to/local/GreynirSeq:/tmp/lint github/super-linter
```

### Type annotation

Type annotation will soon be checked with mypy and should be included.

