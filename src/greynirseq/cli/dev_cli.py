"""Development terminal client for greynirseq, work in progress."""
import json
import logging
from dataclasses import asdict
from typing import List, Optional, TextIO

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
    """Convert a text file to monolingual JSONL document.
    The text file can have structure which is preserved by specifying --line_semantics."""
    for document in read_documents(
        tqdm(input_file, desc="Reading lines"),
        line_sematics=line_semantics,
        domains=domains,
        lang=lang,
        title=title,
    ):
        output_file.write(json.dumps(asdict(document), ensure_ascii=False) + "\n")


@cli.command()
@click.argument("input_file", type=str)
def jsonl_stats(input_file):
    """Return statistics about a Monolingual JSONL file.
    Statistics include:
    - Number of documents.
    - Number of paragrahs.
    - Number of sentences.
    - Number of words (whitespace separated tokens).
    - Number of unicode characters.
    """
    stats = {"total_documents": 0, "total_sentences": 0, "total_words": 0, "total_paragraphs": 0, "total_chars": 0}
    with open(input_file, "r") as f:
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
