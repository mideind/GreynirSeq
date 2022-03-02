import argparse
from pathlib import Path
from typing import Container, Dict, Tuple

from greynirseq.utils.lemmatizer import Lemmatizer, get_lemmatizer_for_lang


def read_glossary(path: Path) -> Dict[str, str]:
    """Read the glossary from the given file. Return an empty dict if it does not exist."""
    if not path.exists():
        return {}
    with path.open("r") as f:
        return {line.split("\t")[0]: line.split("\t")[1].strip() for line in f}

class GlossaryEvaluator:
    """A glossary evaluator. Processes source and target w.r.t. a glossary.
    Counts occurances of terms in source and target and reports accuracy.
    Evaluates terms based on lower-case."""

    def __init__(self, glossary: Dict[str, str], src_lemmatizer: Lemmatizer, tgt_lemmatizer: Lemmatizer):
        self.src_lemmatizer = src_lemmatizer
        self.tgt_lemmatizer = tgt_lemmatizer
        # The lower-case version of the glossary is only used for evaluation.
        self._glossary_lower = {k.lower(): v.lower() for k, v in glossary.items()}
        self.src_counts, self.tgt_counts = self._get_initial_glossary_counts()

    def _get_initial_glossary_counts(self) -> Tuple[Dict[str, int], Dict[str, int]]:
        """Return the initial glossary counts. Lower-cases the terms. Used for validation."""
        return (
            {term: 0 for term in self._glossary_lower.keys()},
            {term: 0 for term in self._glossary_lower.values()},
        )

    def reset(self):
        """Reset the counting."""
        self.src_counts, self.tgt_counts = self._get_initial_glossary_counts()

    @staticmethod
    def _get_term_count_in_sents(sent: str, glossary: Container[str], lemmatizer: Lemmatizer) -> Dict[str, int]:
        # We also lower-case the lemmas, since we count against lower-cased terms.
        lemmas = [lemma.lower() for lemma in lemmatizer.lemmatize(sent)]
        counts: Dict[str, int] = dict()
        for lemma in lemmas:
            if lemma in glossary:
                counts[lemma] = counts.get(lemma, 0) + 1
        return counts

    def process_line(self, source: str, target: str):
        src_term_counts = self._get_term_count_in_sents(source, self._glossary_lower.keys(), self.src_lemmatizer)
        tgt_term_counts = self._get_term_count_in_sents(target, self._glossary_lower.values(), self.tgt_lemmatizer)
        for term, count in src_term_counts.items():
            self.src_counts[term] += count
            tgt_term_count_true_positive = tgt_term_counts.get(self._glossary_lower[term], 0)
            # We do not overcount the number of times a term is used in the target.
            self.tgt_counts[self._glossary_lower[term]] += min(tgt_term_count_true_positive, count)

    def compute_glossary_accuracy(self) -> float:
        total_count = sum(self.src_counts.values())
        total_correct = sum(self.tgt_counts.values())
        try:
            return round(total_correct / total_count, 4)
        except ZeroDivisionError:
            return 0.0


if __name__ == "__main__":
    # We load the glossary from a file.
    parser = argparse.ArgumentParser()
    parser.add_argument("--glossary-file", type=str, required=True, help="Glossary file")
    parser.add_argument("--src-lang", type=str, required=True, help="Source language")
    parser.add_argument("--tgt-lang", type=str, required=True, help="Target language")
    parser.add_argument(
        "--tsv-file-to-evaluate",
        type=str,
        required=True,
        help="The file to evaluate. Should be tsv with 'source'*tab*'target'.",
    )
    args = parser.parse_args()
    glossary = read_glossary(Path(args.glossary_file))
    src_lemmatizer = get_lemmatizer_for_lang(args.src_lang)
    tgt_lemmatizer = get_lemmatizer_for_lang(args.tgt_lang)
    evaluator = GlossaryEvaluator(glossary, src_lemmatizer, tgt_lemmatizer)
    with open(args.tsv_file_to_evaluate, "r") as f:
        for line in f:
            src, best_translation = line.strip().split("\t")
            evaluator.process_line(src, best_translation)
    print(f"Glossary accuracy: {evaluator.compute_glossary_accuracy()}")
