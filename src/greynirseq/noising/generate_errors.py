import argparse
import sys

import torch
from ieg.dataset import ErrorDataset, worker_init_fn
from ieg.errorrules import NounCaseErrorRule
from ieg.errorrules.errors import DativitisErrorRule, MoodErrorRule, NoiseErrorRule, SwapErrorRule
from tokenizer import correct_spaces


def main() -> None:

    parser = argparse.ArgumentParser()
    parser.add_argument("infile", nargs="?", type=argparse.FileType("r"), default=sys.stdin)
    parser.add_argument("-pos", "--posfile", help="File with POS tags", required=False)
    parser.add_argument(
        "-wer",
        "--word-spelling-error-rate",
        type=float,
        default=0.3,
        help="Error rate used for spelling of words.",
        required=False,
    )
    parser.add_argument(
        "-rer", "--rule-chance-error-rate", help="Chance for each rule to be applied", default=0.9, type=float
    )
    parser.add_argument(
        "-parse", "--parse-online", help="Parse sentence with Greynir if pos not provided", type=bool, default=True
    )
    parser.add_argument("-seed", "--seed", default=1, type=int)
    parser.add_argument("-no-detok", "--dont-detokenize", default=False, type=bool)
    parser.add_argument("-n", "--nproc", default=1, type=int)
    parser.add_argument("-b", "--batch-size", default=1, type=int)
    args = parser.parse_args()

    error_handlers = [DativitisErrorRule, NounCaseErrorRule, SwapErrorRule, MoodErrorRule, NoiseErrorRule]

    error_dataset = ErrorDataset(args.infile, args.posfile, args, error_handlers=error_handlers)

    error_loader = torch.utils.data.DataLoader(
        error_dataset,
        num_workers=args.nproc,
        worker_init_fn=worker_init_fn,
        batch_size=args.batch_size,
    )

    for error_batch in error_loader:
        for error_sentence in error_batch:
            if args.dont_detokenize:
                print(error_sentence)
            else:
                print(correct_spaces(error_sentence))


if __name__ == "__main__":
    main()
