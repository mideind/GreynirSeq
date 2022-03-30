import argparse
import sys

from ieg.errorrules.errors import DativeSicknessErrorRule, NoiseErrorRule, SwapErrorRule, MoodErrorRule
from ieg.errorrules import NounCaseErrorRule 
from ieg.dataset import ErrorDataset


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('infile', nargs='?', type=argparse.FileType('r'), default=sys.stdin)
    parser.add_argument("-pos", "--posfile", help="File with POS tags", required=False)
    parser.add_argument("-wer", "--word-spelling-error-rate", type=float, default=0.3, help="Error rate used for spelling of words.", required=False)
    parser.add_argument("-rer", "--rule-chance-error-rate", help="Chance for each rule to be applied", default=0.9, type=float)
    parser.add_argument("-parse", "--parse-online", help="Parse sentence with Greynir if pos not provided", type=bool, default=True)
    parser.add_argument("-seed", "--seed", default=1, type=int)
    args = parser.parse_args()

    error_handlers = [
        DativeSicknessErrorRule,
        NounCaseErrorRule,
        #NoiseErrorRule,
        #SwapErrorRule,
        #MoodErrorRule
    ]

    error_data = ErrorDataset(
        args.infile,
        args.posfile,
        args,
        error_handlers=error_handlers
    )

    for error_sentence in error_data:
        print(error_sentence)


if __name__ == '__main__':
    main()
