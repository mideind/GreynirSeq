"""Development terminal client for greynirseq, work in progress."""
import json
import logging
from typing import List, Optional, TextIO, Tuple

import click
from tqdm import tqdm

from greynirseq.utils.jsonl_utils import LineSemantics, MonolingualJSONDocument, read_documents


@click.group()
def cli():
    """
    CLI for development.
    """
    logging.basicConfig(level=logging.INFO)


@cli.command()
@click.argument("input_file", type=click.File("r"))
@click.argument("output_file", type=click.File("w"))
@click.option(
    "--line_semantics",
    type=click.Choice(LineSemantics),
    default=LineSemantics.sent_ignore,
    help=LineSemantics.__doc__,
)
@click.option(
    "--domains",
    default=None,
    multiple=True,
    help="A list of domains for the document. If not given, the document will be assigned no domain. "
    "Usage: --domains domain1 --domains domain2 ...",
)
@click.option(
    "--lang",
    default=None,
    required=True,
    help="The language of the document.",
)
@click.option(
    "--title",
    default=None,
    help="The title of the document. If not given, the document will be assigned no title.",
)
def to_jsonl(
    input_file: TextIO,
    output_file: TextIO,
    line_semantics: LineSemantics,
    domains: Optional[List[str]],
    lang: str,
    title: Optional[str],
):
    """Convert a text file to monolingual JSON documents (JSONL) file.
    The text file can have structure which is preserved by specifying --line_semantics."""
    for document in read_documents(
        tqdm(input_file, desc="Reading lines"),
        line_sematics=line_semantics,
        domains=domains,
        lang=lang,
        title=title,
    ):
        output_file.write(document.to_json_str())  # json_str ends with newline.


@cli.command()
@click.argument("input_file", type=click.File("r"))
@click.argument("output_file", type=click.File("w"))
@click.option(
    "--sentence_postfix",
    default="\n",
    required=False,
    help="A string to add after each sentence.",
)
@click.option(
    "--paragraph_postfix",
    default="\n",
    required=False,
    help="A string to add after each paragraph.",
)
@click.option(
    "--document_postfix",
    default="\n",
    required=False,
    help="A string to add after each document.",
)
def to_text(
    input_file: TextIO,
    output_file: TextIO,
    sentence_postfix: str,
    paragraph_postfix: str,
    document_postfix: str,
):
    """Convert monolingual JSON documents (JSONL) file to a text file.

    Postfixes at the sentence, paragraph and document level can be specified.
    The default behaviour is to add a newline after each sentence, paragraph and the document.
    document = [[sent1], [sent2, sent3]] becomes:
        sent1 # newline after sentences
            # newline after each paragraph
        sent2
        sent3
            # newline after each paragraph
            # newline after document
    This text file can be read by to_json using LineSemantics.sent_para_doc,
    to return the same document (excluding metadata)."""
    for line in tqdm(input_file, desc="Reading lines"):
        document = MonolingualJSONDocument.from_json_str(line)
        output_file.write(
            document.to_text(
                sentence_postfix=sentence_postfix,
                paragraph_postfix=paragraph_postfix,
                document_postfix=document_postfix,
            )
        )


@cli.command()
@click.argument("input_files", type=str, nargs=-1)
def jsonl_stats(input_files: Tuple[str]):
    """Return statistics about Monolingual JSONL files, supports multiple files.
    Statistics include:
    - Number of documents.
    - Number of paragrahs.
    - Number of sentences.
    - Number of words (whitespace separated tokens).
    - Number of unicode characters.
    """
    for input_file in input_files:
        print(input_file)
        with open(input_file, "r") as f:
            stats = {
                "total_documents": 0,
                "total_paragraphs": 0,
                "total_sentences": 0,
                "total_words": 0,
                "total_chars": 0,
            }
            for line in tqdm(f, desc="Reading JSONL file"):
                mono_doc = MonolingualJSONDocument(**json.loads(line))
                stats["total_documents"] += 1
                stats["total_paragraphs"] += len(mono_doc.document)
                for paragraph in mono_doc.document:
                    stats["total_sentences"] += len(paragraph)
                    for sentence in paragraph:
                        stats["total_words"] += len(sentence.split())
                        stats["total_chars"] += len(sentence)
        print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    cli()
