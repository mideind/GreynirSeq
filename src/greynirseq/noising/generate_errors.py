import argparse
import sys

from ieg.dataset import ErrorDataset, worker_init_fn
from ieg.errorrules import (
    DativitisErrorRule,
    DeleteSpaceErrorRule,
    DropCommaRule,
    DuplicateWordsRule,
    MoodErrorRule,
    NoiseErrorRule,
    NounCaseErrorRule,
    SplitWordsRule,
    SwapErrorRule,
)
from tokenizer import correct_spaces
from torch.utils.data import DataLoader


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("infile", nargs="?", type=argparse.FileType("r"), default=sys.stdin)
    parser.add_argument("--posfile", help="File with POS tags", required=False)
    parser.add_argument(
        "-w",
        "--word-spelling-error-rate",
        type=float,
        default=0.6,
        help="Error rate used for spelling of words.",
        required=False,
    )
    parser.add_argument(
        "-r", "--rule-chance-error-rate", help="Chance for each rule to be applied", default=0.3, type=float
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
        DropCommaRule,
        SwapErrorRule,
        DuplicateWordsRule,
        SplitWordsRule,
        NoiseErrorRule,
        DeleteSpaceErrorRule,
    ]
    error_dataset = ErrorDataset(args.infile, args.posfile, args, error_generators=error_generators)

    error_loader = DataLoader(
        error_dataset,
        num_workers=args.nproc,
        worker_init_fn=worker_init_fn,
        batch_size=args.batch_size,
    )

    for error_batch in error_loader:
        if len(error_batch) == 2:
            sent_pair = []
            for error_sentence in error_batch:
                sent_pair.append(error_sentence[0])
            print(f"{correct_spaces(sent_pair[0])}\t{correct_spaces(sent_pair[1])}")


if __name__ == "__main__":
    main()
