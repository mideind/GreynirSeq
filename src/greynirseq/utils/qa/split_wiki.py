#!/usr/bin/env python
import argparse
from collections import namedtuple

# TOOD: use actual mp
from multiprocessing.dummy import Pool
from typing import Generator, List

from lemmatizer import Lemmatizer

WikiArticle = namedtuple("WikiArticle", ["title_id", "title", "linking_title", "text"])


class LemmatizeWiki:
    def __init__(self, wiki_file: str) -> None:
        self.wiki_file = wiki_file
        self.l = Lemmatizer()  # noqa

    def _read_wiki_dump(self) -> Generator[WikiArticle, None, None]:
        with open(self.wiki_file) as inputfile:
            for line in inputfile.readlines():
                title_id, title, linking_title, text = line.strip().split("\t")
                yield WikiArticle(title_id, title, linking_title, text)

    def write_sentences(self, outfile_name: str, thread_count: int = 16) -> None:
        def parse_batch(article: WikiArticle, lemmatizer: Lemmatizer = self.l) -> List:  # noqa
            data = []
            for idx, (lemmas, tokens, sentence) in enumerate(lemmatizer.lemmatize(article.text)):
                data.append("{}\t{}\t{}\t{}\n".format(article.title_id, idx, sentence, " ".join(lemmas)))
            return data

        pool = Pool(thread_count)
        with open(outfile_name, "w") as outfile:
            i = 0
            article_pool = []
            for article in self._read_wiki_dump():
                i += 1
                article_pool.append(article)
                if len(article_pool) == thread_count:
                    results = pool.map(parse_batch, article_pool)
                    for result in results:
                        outfile.writelines(result)
                    article_pool = []
            for article in article_pool:
                # Empty the pool
                outfile.writelines(parse_batch(article))


def main() -> None:
    # TODO Add suport for multiprocessing, selection of processes
    #      and selection of file format.
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to Wikipedia dump")
    parser.add_argument("--output", type=str, help="Path where output should be written")
    parser.add_argument("--threads", type=int, help="Number of threads to use")
    args = parser.parse_args()

    lw = LemmatizeWiki(args.input)
    lw.write_sentences(args.output, args.threads)


if __name__ == "__main__":
    main()
