import argparse

import spacy
import torch
import tqdm
from spacy.gold import biluo_tags_from_offsets
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

from greynirseq.nicenlp.models.multiclass import MultiClassRobertaModel
from greynirseq.settings import IceBERT_NER_CONFIG, IceBERT_NER_PATH


def icelandic_ner(input_file, tagged_file):
    model = MultiClassRobertaModel.from_pretrained(IceBERT_NER_PATH, **IceBERT_NER_CONFIG)
    model.to("cpu")
    model.eval()

    infile = open(input_file)
    ofile = open(tagged_file, "w")

    for sentence in tqdm.tqdm(infile.readlines()):
        sentence = sentence.strip().split("\t")[1]
        # Todo, update for batching when predict_pos fixed
        labels, _ = model.predict_labels(sentence)  # type: ignore
        ofile.writelines(f"{sentence}\t{' '.join(labels)}\n")


def english_ner(input_file, output_file):

    nlp = spacy.load("en_core_web_lg")
    hnlp = pipeline("ner")  # noqa

    model = AutoModelForTokenClassification.from_pretrained(
        "dbmdz/bert-large-cased-finetuned-conll03-english"
    ).to(  # type: ignore
        "cuda"
    )
    tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
    label_list = [
        "O",  # Outside of a named entity
        "B-MISC",  # Beginning of a miscellaneous entity right after another miscellaneous entity
        "I-MISC",  # Miscellaneous entity
        "B-PER",  # Beginning of a person's name right after another person's name
        "I-PER",  # Person's name
        "B-ORG",  # Beginning of an organisation right after another organisation
        "I-ORG",  # Organisation
        "B-LOC",  # Beginning of a location right after another location
        "I-LOC",  # Location
    ]

    def spacy_tok_ner(sent):
        doc = nlp(sent)
        j = doc.to_json()

        ranges = [(a["start"], a["end"]) for a in j["tokens"]]
        ents = j["ents"]

        tokens = []
        for range in ranges:
            tokens.append(sent[range[0] : range[1]])  # noqa

        entlocs = [(a["start"], a["end"], a["label"]) for a in ents]
        labels = biluo_tags_from_offsets(doc, entlocs)

        return tokens, labels

    def hugface_tok_ner(sequence):
        # Bit of a hack to get the tokens with the special tokens
        tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))  # type: ignore
        inputs = tokenizer.encode(sequence, return_tensors="pt").to("cuda")  # type: ignore
        outputs = model(inputs)[0]
        predictions = torch.argmax(outputs, dim=2)
        bert_tokens = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())][
            1:-1
        ]
        tokens = " ".join([t[0] for t in bert_tokens]).replace(" ##", "").split(" ")
        labels = [t[1] for t in bert_tokens if len(t[0]) < 2 or t[0][:2] != "##"]
        return tokens, labels

    ofile = open(output_file, "w")
    with open(input_file) as open_file:
        for line in tqdm.tqdm(open_file.readlines()):
            source = line.strip().split("\t")[0]
            using = "hf"
            if len(source) < 512:
                tokens, ents = hugface_tok_ner(source)
            else:
                using = "sp"
                tokens, ents = spacy_tok_ner(source)
            ofile.writelines("{}\t{}\t{}\n".format(" ".join(tokens), " ".join(ents), using))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["is", "en"])
    parser.add_argument("--input")
    parser.add_argument("--output")

    args = parser.parse_args()

    if args.language == "is":
        icelandic_ner(args.input, args.output)
    elif args.language == "en":
        english_ner(args.input, args.output)


if __name__ == "__main__":
    main()
