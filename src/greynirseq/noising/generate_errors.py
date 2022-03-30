import argparse
from ieg.errorrules.errors import DativeSicknessErrorRule, NoiseErrorRule, SwapErrorRule, NounCaseErrorRule, MoodErrorRule
from ieg.dataset import ErrorDataset


ERROR_RULES = [DativeSicknessErrorRule, NounCaseErrorRule, NoiseErrorRule,  SwapErrorRule, MoodErrorRule]


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", help="File to errorify", required=True)
    parser.add_argument("-pos", "--posfile", help="File with POS tags", required=False)
    parser.add_argument("-seed", "--seed", default=1, type=int)
    args = parser.parse_args()

    error_data = ErrorDataset(
        args.infile,
        args.posfile,
        error_handlers=ERROR_RULES
    )

    for error_sentence in error_data:
        print(error_sentence)


if __name__ == '__main__':
    main()
