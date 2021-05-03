# flake8: noqa

import torch
from fairseq.models.roberta import RobertaModel
from fairseq.models.transformer import TransformerModel
from flask import Flask, escape, request
from flask_cors import CORS, cross_origin

from greynirseq.nicenlp.criterions.multi_span_prediction_criterion import *
from greynirseq.nicenlp.data.datasets import *
from greynirseq.nicenlp.tasks.translation_with_backtranslation import *
from greynirseq.utils.tokenize_splitter import index_text


class IceBERTRunner:
    def __init__(self):
        self.model = RobertaModel.from_pretrained(
            "/data/models/icebert-base-36k",
            checkpoint_file="model.pt",
            bpe="gpt2",
            gpt2_encoder_json="/data/models/icebert-base-36k/icebert-bpe-vocab.json",
            gpt2_vocab_bpe="/data/models/icebert-base-36k/icebert-bpe-merges.txt",
        )
        self.model.to("cpu")
        self.model.eval()

    def infer(self, lines_of_text):
        return self.model.fill_mask(lines_of_text)


class RoBERTaRunner:
    def __init__(self):
        self.model = RobertaModel.from_pretrained(
            "/data/models/roberta.large",
            checkpoint_file="model.pt",
        )
        self.model.to("cpu")
        self.model.eval()

    def infer(self, lines_of_text):
        return self.model.fill_mask(lines_of_text)


class NERRunner:
    def __init__(self):
        self.model = IcebertConstModel.from_pretrained(  # pylint: disable=undefined-variable
            "/data/models/icebert_ner/ner_slset",
            checkpoint_file="checkpoint_last.pt",
            data_name_or_path="/data/models/MIM-GOLD-NER_split/8_entity_types/bin/bin",
            gpt2_encoder_json="/data/models/icebert-base-36k/icebert-bpe-vocab.json",
            gpt2_vocab_bpe="/data/models/icebert-base-36k/icebert-bpe-merges.txt",
            term_schema="/data/models/MIM-GOLD-NER_split/term.json",
        )
        self.model.to("cpu")
        self.model.eval()

    def infer(self, sentence):
        cat_idx, labels, sentence = self.model.predict_pos(sentence, device="cpu")
        return labels, sentence


class TranslationRunner:

    add_bos = False

    def __init__(self):

        self.model = TransformerModel.from_pretrained(
            "/data/models/eng-isl-base-v1",
            checkpoint_file="checkpoint.en-is.avg8.pt",
            data_name_or_path="/data/models/eng-isl-base-v1",
            gpt2_encoder_json="/data/models/fairseq-eng-isl-base-std-parice/eng-isl-bbpe/eng-isl-bbpe-32k/eng-isl-bbpe-32k-vocab.json",
            gpt2_vocab_bpe="/data/models/fairseq-eng-isl-base-std-parice/eng-isl-bbpe/eng-isl-bbpe-32k/eng-isl-bbpe-32k-merges.txt",
            source_lang="en",
            target_lang="is",
            bpe="gpt2",
            beam=5,
            len_penalty=0.6,
            task="translation_with_backtranslation",
        )

        self.model.to("cpu")
        self.model.eval()

    def infer(self, sentences, prefixes=None):

        if isinstance(sentences, str):
            sentences = " " + sentences
            return self.infer([sentences], [prefixes])

        bos_idx_tensor = torch.tensor([self.model.task.src_dict.bos()])

        prefix_args = {}
        if prefixes is not None:
            # prefix = self.model.encode(prefixes)[:-1].unsqueeze(0)

            pad_idx = self.model.task.src_dict.pad()
            if not self.add_bos:
                encoded = [torch.cat([bos_idx_tensor, self.model.encode(prefix)]) for prefix in prefixes]
            else:
                encoded = [self.model.encode(prefix) for prefix in prefixes]
            prefixes = torch.nn.utils.rnn.pad_sequence(encoded, padding_value=pad_idx).T
            # Remove EOS token
            prefixes[prefixes == self.model.task.src_dict.eos()] = pad_idx

            prefix_args = {"prefix_tokens": prefixes}
        if not self.add_bos:
            tokenized_sentences = [torch.cat([bos_idx_tensor, self.model.encode(sentence)]) for sentence in sentences]
        else:
            tokenized_sentences = [self.model.encode(" " + sentence) for sentence in sentences]
        batched_hypos = self.model.generate(tokenized_sentences, 10, inference_step_args=prefix_args)
        hypos = []
        for s_hypos in batched_hypos:
            hypos.append([self.model.decode(hypo["tokens"]) for hypo in s_hypos])
        return hypos


class TranslationRunnerIsEn(TranslationRunner):
    add_bos = True

    def __init__(self):

        self.model = TransformerModel.from_pretrained(
            "/data/models/isl-eng-base-v4",
            checkpoint_file="checkpoint_10_130000.is-en.base-v4.pt",
            data_name_or_path="/data/models/isl-eng-base-v4",
            gpt2_encoder_json="/data/models/fairseq-eng-isl-base-std-parice/eng-isl-bbpe/eng-isl-bbpe-32k/eng-isl-bbpe-32k-vocab.json",
            gpt2_vocab_bpe="/data/models/fairseq-eng-isl-base-std-parice/eng-isl-bbpe/eng-isl-bbpe-32k/eng-isl-bbpe-32k-merges.txt",
            source_lang="is",
            target_lang="en",
            bpe="gpt2",
            beam=5,
            len_penalty=0.6,
            task="translation_with_backtranslation",
        )

        self.model.to("cpu")
        self.model.eval()


application = Flask(__name__)
cors = CORS(application)
application.config["CORS_HEADERS"] = "Content-Type"

ner_runner = NERRunner()
translation_runner = TranslationRunner()
translation_runner_isen = TranslationRunnerIsEn()
icebert_runner = IceBERTRunner()
roberta_runner = RoBERTaRunner()
application.config["JSON_AS_ASCII"] = False


@application.route("/ner", methods=["POST"])
def ner():
    if request.json is None or "text" not in request.json:
        return {"error": '"text" field in JSON payload is required'}, 400
    text = request.json.get("text")

    labels, sentence = ner_runner.infer([text])
    return {"ok": True, "text": text, "labels": labels, "sentence": sentence}


@application.route("/translate", methods=["POST"])
def translate():
    if request.json is None or "contents" not in request.json:
        return {"error": '"contents" field in JSON payload is required'}, 400
    contents = request.json.get("contents")

    parsed_text = [index_text(t) for t in contents if t.strip()]
    sentences = [v for p in parsed_text for k, v in p[1].items()]
    prefixes = request.json.get("prefixes", ["" for i in range(len(sentences))])

    batched = [pair for pair in zip(sentences, prefixes)]

    src_lang = request.json.get("sourceLanguageCode", "en")
    if src_lang == "en":
        translation_hypos = [translation_runner.infer(pair[0], pair[1]) for pair in batched]
    else:
        translation_hypos = [translation_runner_isen.infer(pair[0], pair[1]) for pair in batched]

    translations = [hypos[0] for hypos in translation_hypos]
    translations = [t[0] for t in translations]

    parsed_response = []
    tot = 0
    for i in range(len(parsed_text)):
        idxs = parsed_text[i][0][0]  # list of ints if many pgs
        sentences = parsed_text[i][1]
        len_pg = len(sentences)
        source_sentences = [sentences[j] for j in idxs]
        target_pg = translations[tot : tot + len_pg]
        tot += len_pg

        parsed_response.append(
            {
                "translatedText": " ".join(target_pg),
                "model": "",
                "translatedTextStructured": [
                    (append_punctuation_previous(s), append_punctuation_previous(t))
                    for s, t in zip(source_sentences, target_pg)
                ],
                "source_sentences": source_sentences,
                "target_pg": target_pg,
                "idxs": idxs,
                "parse_text": parsed_text,
            }
        )

    return {"ok": True, "translations": parsed_response}


def append_punctuation_previous(sentence):
    tokens = sentence.split()
    clean_tokens = []
    for token in tokens:
        if token in [",", ".", ":", ";"]:
            if clean_tokens:
                clean_tokens[-1] += token
        else:
            clean_tokens.append(token)
    return " ".join(clean_tokens)


@application.route("/fill_mask", methods=["POST"])
def fill_mask():
    if request.json is None or "text" not in request.json:
        return {"error": '"text" field in JSON payload is required'}, 400
    text = request.json.get("text")
    language = request.json.get("language", "is")
    if language == "is":
        summary = [(s, tokens, hypo) for s, hypo, tokens in icebert_runner.infer(text)]
    else:
        summary = [(s, tokens, hypo) for s, hypo, tokens in roberta_runner.infer(text)]
    return {"ok": True, "text": text, "summary": summary}


if __name__ == "__main__":
    application.run("0.0.0.0", 3001)
