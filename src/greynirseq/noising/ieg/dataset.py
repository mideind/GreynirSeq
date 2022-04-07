import math

import torch
from ieg import g
from torch.utils.data import Dataset


def worker_init_fn(worker_id):
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset  # the dataset copy in this worker process
    overall_start = dataset.start
    overall_end = dataset.end
    # configure the dataset to only process the split workload
    per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
    worker_id = worker_info.id
    start = overall_start + worker_id * per_worker
    end = min(dataset.start + per_worker, overall_end)
    dataset.set_start_end(start, end)


class ErrorDataset(Dataset):

    has_pos: bool = False
    start = 0
    end = None

    def __init__(self, infile, posfile, args, error_handlers=[]) -> None:
        self.has_pos = posfile is not None
        self.args = args

        self.sentences = infile.read().split("\n")
        self.end = len(self.sentences)

        if self.has_pos:
            with open(posfile) as posfilehandler:
                self.postags = posfilehandler.readlines()

        self.error_handlers = error_handlers

    def __getitem__(self, index) -> str:
        errored_sentence: str = self.sentences[index].rstrip()
        if not errored_sentence.strip():
            # Empty or None, do nothing
            return errored_sentence

        pos_sentence: str = None
        sentence_tree = None
        if self.args.parse_online:
            if self.pos_sentence(errored_sentence):
                pos_sentence = self.pos_sentence(errored_sentence)["pos"]
                sentence_tree = self.pos_sentence(errored_sentence)["tree"]

        for error_handler in self.error_handlers:

            if error_handler.needs_pos and not (self.has_pos or self.args.parse_online):
                continue
            elif error_handler.needs_pos and self.args.parse_online:
                if pos_sentence is None:
                    continue

            if self.has_pos:
                pos = self.postags[index]
            elif self.args.parse_online:
                pos = pos_sentence
            else:
                pos = None

            errored_sentence = error_handler.apply(
                {"text": errored_sentence, "pos": pos, "tree": sentence_tree, "args": self.args}
            )
            if not errored_sentence:
                # Rule broke sentence
                return self.sentences[index].rstrip()

        return errored_sentence

    def set_start_end(self, start, end) -> None:
        self.start = start
        self.end = end

    def __len__(self) -> int:
        return self.end - self.start

    def pos_sentence(self, text) -> dict:
        """Parse text with greynir. Supports multiple sentences in
        input string, joins pos for each sentences before returning.
        """
        parsed = g.parse(text)
        pos_data = []
        parse_tree = []
        for sentence in parsed["sentences"]:
            if sentence.terminals is None:
                return None
            pos_data += sentence.terminals
            parse_tree = sentence.tree
        return {"pos": pos_data, "tree": parse_tree}
