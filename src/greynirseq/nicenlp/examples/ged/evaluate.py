#!/usr/bin/env python

from fairseq.models.roberta import RobertaModel # type: ignore
import torch
from typing import List, Tuple, Dict
import itertools
import argparse
from pathlib import Path
from pprint import pprint


import greynirseq
from greynirseq.nicenlp.models.pos_model import IceBERTPOSModel

_model: RobertaModel


class TestData:
    def __init__(self, lines, labels):
        """ All the arguments MUST be the same length. """

#        if len(lines) != len(labels):
#            import pdb; pdb.set_trace()
#        assert len(lines) == len(labels), f"lines: {len(lines)} labels: {len(labels)}"
        self.lines = lines
        self.labels = labels


    def __getitem__(self, key):
        return TestData(
            self.lines[key],
            self.labels[key],
        )

    def __len__(self):
        return len(self.lines)


def load_model(model_dir: str, checkpoint_file: str, encoder_json: str, vocab_bpe: str):
    """ Load the pretrained classifier. """
    # NOTE: Some files are implicitly required in the binary-classifier folder.
    global _model
    _model = IceBERTPOSModel.from_pretrained(
        model_dir,
        checkpoint_file=checkpoint_file,
        bpe="gpt2",
        gpt2_encoder_json=encoder_json,
        gpt2_vocab_bpe=vocab_bpe,
        task="multi_label_token_classification_task",
    )
    """
    _model = RobertaModel.from_pretrained(
        "./binary-classifier/",
        checkpoint_file="model.pt",
        bpe="gpt2",
        gpt2_encoder_json="/data/models/icebert-base-36k/icebert-bpe-vocab.json",
        gpt2_vocab_bpe="/data/models/icebert-base-36k/icebert-bpe-merges.txt",
        task="sentence_prediction",
    )
    #"""
    _model.eval()
    _model.cuda()


def label(prediction_result) -> str:
    """ Get a string label for a prediction result. """
    return _model.task.label_dictionary.string(
        [prediction_result.argmax().item() + _model.task.label_dictionary.nspecial]
    )


def infer(sentence: str) -> str:
    """ Classify a single sentence. Return a simple string result. """
    return label(_model.predict("sentence_classification_head", _model.encode(sentence)))


def infer2(sentence: str) -> str:
    """ Classify a single sentence. Return a raw prediction tensor. """
    return _model.predict("sentence_classification_head", _model.encode(sentence))


def infer_batch(batch: torch.Tensor) -> torch.Tensor:
    """ Classify a batch. Return a raw prediction tensor. """
    #return _model.predict("sentence_classification_head", batch, "cpu")
    #import pdb; pdb.set_trace()
    ret = []
    for b in batch:
        ret.append(_model.predict(b, "cpu"))

    #return _model.predict(batch, "cpu")
    return ret


def infer_sentences(sentences: List[str]):
    """ 'cause icebertmodel has a different predict function """
    return _model.predict_labels(sentences)


def load_testset(testset_dir: str) -> TestData:
    """ Load the test set that we assume exists. """
    with open(Path(testset_dir) / "test.input0") as f:
        lines = f.read().split('\n')
    with open(Path(testset_dir) / "test.label") as f:
        labels = f.read().split('\n')
    return TestData(
        lines=lines,
        labels=labels,
    )


def create_batch(data: TestData) -> torch.Tensor:
    """ Encode a single batch of sentences. """
    encoded = []
    for d in data:
        encoded.append(_model.encode(d.lines))
    return torch.nn.utils.rnn.pad_sequence(encoded, padding_value=1).T


def sentences_to_batches(testdata: TestData, batch_size: int) -> List[Tuple[torch.Tensor, TestData]]:
    """ Turn a long list of sentences into batches. """
    current = 0
    batches = []
    while current < len(testdata):
        batch = testdata[current:current+batch_size]
        current += batch_size
        batches.append((create_batch(batch), batch))

    return batches


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an evaluation for a grammatical error detection model")

    parser.add_argument("test_data_dir")
    parser.add_argument("model_dir")
    parser.add_argument("checkpoint_file")
    parser.add_argument("encoder_json")
    parser.add_argument("vocab_bpe")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    load_model(args.model_dir, args.checkpoint_file, args.encoder_json, args.vocab_bpe)

    # Explanation of the data. Hopefully this helps with reading the code.
    # load_testset returns a dict with 5 lists of strings. The lists are the data and 4 different sets of labels:
    #       lines: The data itself
    #       labels: 0/1 for no error or error on the corresponding line
    #       multilabels: Text labels for errors in each sentence. Empty lines for no error.
    #       simcategories: Same as multilabels but different set of labels.
    #       supercategories: Same as multilabels but different set of labels.
    # sentences_to_batches encodes this list into something the model can evaluate.
    #   The return value is a list of (encoded batch, subset of the *truth*) items that correspond to each other
    # infer_batch takes in an encoded batch and returns a prediciton in tensor form (nb. this is iterable)

    # XXX: The batch size that works for this is smaller than expected (16 instead of 32).
    #      Are we maybe loading each value in two tensors?

    
    testset = load_testset(args.test_data_dir)
    sentences = testset.lines

    count = 0
    for s in sentences:
        res = infer_sentences(s)
        print((s, res))
        count += 1

    import sys
    sys.exit(0)

    batches = sentences_to_batches(load_testset(args.test_data_dir), 4)

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    count = 0
    for b in batches:
        batch_prediction = infer_batch(b[0])
        for i in range(len(batch_prediction)):
            prediction = label(batch_prediction[i])
            truth = b[1][i].labels # current batch -> list of true data -> the current item -> the item's label (true value)
            if prediction == truth:
                # True prediction
                if prediction == "1":
                    tp += 1
                else:
                    tn += 1
            else:
                # False prediction
                if prediction == "1":
                    fp += 1
                else:
                    fn += 1

            count += 1
            if count % 50 == 0:
                print(f"{count}: tp {tp} tn {tn} fp {fp} fn {fn}")

    print(f"Count {count}, tp {tp}, tn {tn}, fp {fp}, fn {fn}")
    precision = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1 = 2*precision*recall/(precision+recall)
    print(f"Precision {precision}, recall {recall}, F1 {f1}")

