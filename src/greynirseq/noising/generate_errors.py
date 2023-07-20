import argparse
import json
import sys

import torch
from ieg.dataset import ErrorDataset, worker_init_fn
from ieg.errorrules import (
    DativitisErrorRule,
    DeleteSpaceErrorRule,
    DuplicateWordsRule,
    MoodErrorRule,
    NoiseErrorRule,
    NounCaseErrorRule,
    SplitWordsRule,
    SwapErrorRule,
)
from tokenizer import correct_spaces


def insert_errors(text, posfile, arguments, error_generators):
    """Insert errors into input text, returning the errored sentences"""

    error_dataset = ErrorDataset(text, posfile, arguments, error_generators=error_generators)

    error_loader = torch.utils.data.DataLoader(
        error_dataset,
        num_workers=arguments.nproc,
        worker_init_fn=worker_init_fn,
        batch_size=arguments.batch_size,
    )

    return error_loader


def write_jsonl(doc, full_text, outfile) -> None:
    """Write an output jsonl file with errored sentences"""

    # Check whether text consists of more than one paragraph
    if any(isinstance(paragraph, list) for paragraph in full_text):
        # Paragraph division is preserved in output file
        document = []
        for element in full_text:
            document.append([" ".join(sent) for sent in element])
    else:
        document = [full_text]

    # All information from original .jsonl file is preserved and written to output file
    uuid = json.loads(doc)["uuid"]
    lang = json.loads(doc)["lang"]
    domains = json.loads(doc)["domains"]  # TODO: add 'noised' domain
    title = json.loads(doc)["title"]
    ordering = json.loads(doc)["ordering"]

    json_output = [
        {
            "uuid": uuid,
            "lang": lang,
            "document": document,
            "domains": domains,
            "title": title,
            "ordering": ordering,
        }
    ]

    for item in json_output:
        outfile.write(json.dumps(item, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", nargs="?", type=argparse.FileType("r"), default=sys.stdin)
    parser.add_argument("--posfile", help="File with POS tags", required=False)
    parser.add_argument(
        "-w",
        "--word-spelling-error-rate",
        type=float,
        default=0.3,
        help="Error rate used for spelling of words.",
        required=False,
    )
    parser.add_argument(
        "-r", "--rule-chance-error-rate", help="Chance for each rule to be applied", default=0.9, type=float
    )
    parser.add_argument(
        "-p", "--parse-online", help="Parse sentence with Greynir if pos not provided", type=bool, default=True
    )
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("-t", "--dont-detokenize", action="store_true")
    parser.add_argument("-n", "--nproc", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    args = parser.parse_args()

    error_generators = [
        DativitisErrorRule,
        MoodErrorRule,
        NounCaseErrorRule,
        SwapErrorRule,
        DuplicateWordsRule,
        SplitWordsRule,
        NoiseErrorRule,
        DeleteSpaceErrorRule,
    ]

    # Handling for .jsonl files, which returns a noised .jsonl file
    if args.infile.name.endswith(".jsonl"):
        with open(args.infile.name, "r") as json_file:
            json_list = list(json_file)

        output_file = args.infile.name.split(".")[0]
        with open(f"{output_file}_noised.jsonl", "w", encoding="utf-8") as outfile:
            for doc in json_list:
                if len(json.loads(doc)["document"]) > 1:
                    # More than one paragraph in file
                    full_text = []
                    for paragraph in json.loads(doc)["document"]:
                        # Each paragraph is a list within the list of full text
                        paragraph_text = []
                        full_text.append(paragraph_text)

                        error_loader = insert_errors(paragraph, args.posfile, args, error_generators=error_generators)
                        for error_batch in error_loader:
                            paragraph_text.append([correct_spaces(error_sentence) for error_sentence in error_batch])

                            for error_sentence in error_batch:
                                if args.dont_detokenize:
                                    print(error_sentence)
                                else:
                                    print(correct_spaces(error_sentence))

                    write_jsonl(doc, full_text, outfile)

                else:
                    [doc_text] = json.loads(doc)["document"]

                    full_text = []

                    error_loader = insert_errors(doc_text, args.posfile, args, error_generators=error_generators)
                    for error_batch in error_loader:
                        for error_sentence in error_batch:
                            full_text.append(correct_spaces(error_sentence))

                        for error_sentence in error_batch:
                            if args.dont_detokenize:
                                print(error_sentence)
                            else:
                                print(correct_spaces(error_sentence))

                    write_jsonl(doc, full_text, outfile)

    else:
        sentences = args.infile.read().split("\n")
        error_loader = insert_errors(sentences, args.posfile, args, error_generators=error_generators)

        for error_batch in error_loader:
            for error_sentence in error_batch:
                if args.dont_detokenize:
                    print(error_sentence)
                else:
                    print(correct_spaces(error_sentence))


if __name__ == "__main__":
    main()
