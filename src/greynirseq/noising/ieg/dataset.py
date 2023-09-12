import math
from typing import List, Optional, TextIO, Tuple, Type, Union

from reynir.simpletree import SimpleTree
from torch.utils.data import Dataset, get_worker_info

from . import greynir
from .errorrules.errors import ErrorRule


def worker_init_fn(worker_id):
    worker_info = get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    start = overall_start + worker_id * per_worker
    end = min(start + per_worker, overall_end)
    dataset.set_start_end(start, end)


class ErrorDataset(Dataset):
    def __init__(self, infile: TextIO, posfile: str, args, error_generators: List[Type[ErrorRule]]) -> None:
        self.has_pos = posfile is not None
        self.args = args
        self.sentences = infile.read().split("\n")
        self.start = 0
        self.end = len(self.sentences)  # to remove empty lines
        if self.has_pos:
            with open(posfile) as posfilehandler:
                self.postags = posfilehandler.readlines()

        self.error_generators = error_generators

    def __getitem__(self, index) -> Union[Tuple[str, str], None]:
        sentence_to_noise: str = self.sentences[index].rstrip()
        errored_sentence: str = self.sentences[index].rstrip()
        if not errored_sentence.strip():
            # Empty or None, do nothing
            pass

        parsed_sentence: str = ""
        sentence_tree = None
        if self.args.parse_online:
            parse_result = self.parse_sentence(errored_sentence)
            if parse_result:
                parsed_sentence = parse_result["pos"]
                sentence_tree = parse_result["tree"]
        for error_generator in self.error_generators:
            if error_generator.needs_pos and not (self.has_pos or self.args.parse_online):  # type: ignore
                continue
            elif error_generator.needs_pos and self.args.parse_online:  # type: ignore
                if parsed_sentence is None:
                    continue

            if self.has_pos:
                pos = self.postags[index]
            elif self.args.parse_online:
                pos = parsed_sentence
            else:
                pos = None

            errored_sentence = error_generator.apply(
                {"text": errored_sentence, "pos": pos, "tree": sentence_tree, "args": self.args}
            )
            if not errored_sentence:
                # Rule broke sentence
                errored_sentence = self.sentences[index].rstrip()
        # retuning tuple with errored sentence and original sentence
        return (errored_sentence, sentence_to_noise)

    def set_start_end(self, start, end) -> None:
        self.start = start
        self.end = end

    def __len__(self) -> int:
        return self.end - self.start

    def parse_sentence(self, text) -> Optional[dict]:
        """Parse text with greynir. Supports multiple sentences in
        input string, joins pos for each sentences before returning.
        """
        parsed = greynir.parse(text)
        pos_data = []
        parse_tree: Optional[SimpleTree] = None
        for sentence in parsed["sentences"]:
            if sentence.terminals is None:
                return None
            pos_data += sentence.terminals
            parse_tree = sentence.tree
        return {"pos": pos_data, "tree": parse_tree}
