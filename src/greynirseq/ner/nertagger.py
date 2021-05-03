import argparse
import logging
import sys
from typing import Generator, Iterable, List, Tuple

import spacy
import torch
import tqdm
from spacy.gold import biluo_tags_from_offsets
from tokenizer import split_into_sentences
from transformers import AutoModelForTokenClassification, AutoTokenizer

from greynirseq.nicenlp.models.multiclass import MultiClassRobertaModel
from greynirseq.settings import IceBERT_NER_CONFIG, IceBERT_NER_PATH

log = logging.getLogger(__name__)
NER_RESULTS = Generator[Tuple[List[str], List[str], str], None, None]


def icelandic_ner(lines_in: Iterable[str], device: str, batch_size=1) -> NER_RESULTS:
    """NER tags a given collection sentences.

    Args:
        lines_in: The sentences should be given as a string, tokenized and joined by ' ' as the model expects.

    Returns:
        An iterable of a list of tokens, labels and a string representing the model used to NER tagging.
    """
    model = MultiClassRobertaModel.from_pretrained(IceBERT_NER_PATH, **IceBERT_NER_CONFIG)
    model.to(device)
    model.eval()

    tokenized_sents = list(lines_in)
    for ndx in range(0, len(tokenized_sents), batch_size):
        batch = tokenized_sents[ndx : min(ndx + batch_size, len(tokenized_sents))]
        # Todo, update for batching when predict_pos fixed
        batch_labels = model.predict_labels(batch)  # type: ignore
        for to_model, labels in zip(batch, batch_labels):
            toks = to_model.split(" ")
            assert len(labels) == len(
                toks
            ), f"We expect the tokens to be of equal length to the labels: {len(toks)}, {len(labels)}, {toks}, {labels}"
            yield toks, labels, "is"


def icelandic_tok(lines_in: Iterable[str]) -> Iterable[str]:
    """Tokenizes Icelandic sentences. Can be split on spaces; " " to retrieve tokens."""
    for idx, line in enumerate(lines_in):
        line = line.strip()
        if not line:
            log.warning(f"Found empty line at index={idx}")
            continue
        yield " ".join(list(split_into_sentences(line)))


def english_ner(lines_in: Iterable[str], device: str) -> NER_RESULTS:
    """NER tags a given collection sentences.

    Args:
        lines_in: The sentences should be given as a string, tokenized and joined by ' ' as the model expects.

    Returns:
        An iterable of a list of tokens, labels and a string representing the model used to NER tagging.
    """

    nlp = spacy.load("en_core_web_lg")

    model = AutoModelForTokenClassification.from_pretrained("dbmdz/bert-large-cased-finetuned-conll03-english").to(
        device
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

    def spacy_tok_ner(sent: str):
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

    def hugface_tok_ner(sequence: str):
        # Bit of a hack to get the tokens with the special tokens
        tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sequence)))  # type: ignore
        inputs = tokenizer.encode(sequence, return_tensors="pt").to(device)
        outputs = model(inputs)[0]
        predictions = torch.argmax(outputs, dim=2)
        bert_tokens = [(token, label_list[prediction]) for token, prediction in zip(tokens, predictions[0].tolist())][
            1:-1
        ]
        tokens = " ".join([t[0] for t in bert_tokens]).replace(" ##", "").split(" ")
        labels = [t[1] for t in bert_tokens if len(t[0]) < 2 or t[0][:2] != "##"]
        return tokens, labels

    for line in lines_in:
        using = "hf"
        if len(line.split()) < 400:
            tokens, ents = hugface_tok_ner(line)
        else:
            using = "sp"
            tokens, ents = spacy_tok_ner(line)
        assert len(ents) == len(tokens), "We expect the tokens to be of equal length to the labels"
        yield tokens, ents, using


def english_tok(lines_in: Iterable[str]) -> Iterable[str]:
    """Tokenizes English sentences. Can be split on spaces; " " to retrieve tokens."""
    from spacy.lang.en import English

    nlp = English()
    # Create a Tokenizer with the default settings for English
    # including punctuation rules and exceptions
    tokenizer = nlp.tokenizer
    for idx, line in enumerate(lines_in):
        line = line.strip()
        if not line:
            log.warning(f"Found empty line at index={idx}")
            continue
        yield " ".join(tok.text for tok in tokenizer(line))


def ner(lang: str, lines_iter: Iterable[str], device: str) -> NER_RESULTS:
    """Apply NER tagging on a collection of lines. Assumes input is tokenized."""
    if lang == "is":
        return icelandic_ner(lines_iter, device=device)
    elif lang == "en":
        return english_ner(lines_iter, device=device)
    else:
        raise ValueError(f"Unsupported language={lang}")


def tok(lang: str, lines_iter: Iterable[str]) -> Iterable[str]:
    """Tokenize a collection of lines."""
    if lang == "is":
        return icelandic_tok(lines_iter)
    elif lang == "en":
        return english_tok(lines_iter)
    else:
        raise ValueError(f"Unsupported language={lang}")


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", choices=["is", "en"])
    parser.add_argument("--input", nargs="?", type=argparse.FileType("r"), default=sys.stdin)
    parser.add_argument("--output", nargs="?", type=argparse.FileType("w"), default=sys.stdout)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    f_in = args.input
    f_out = args.output

    log.info(f"NER tagging {args.input}->{args.output}")
    toks = tok(lang=args.language, lines_iter=f_in)
    tagged_iter = ner(lang=args.language, lines_iter=tqdm.tqdm(toks), device=args.device)
    for tokens, labels, using in tagged_iter:
        f_out.write(f"{' '.join(tokens)}\t{' '.join(labels)}\t{using}\n")


if __name__ == "__main__":
    main()
