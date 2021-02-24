from pathlib import Path

try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


def main(basedir, destdir, train_size=None, valid_size=None, test_size=None):
    train_size = train_size or -1
    if train_size < 0:
        train_size = sys.maxsize

    assert basedir.is_dir(), "Expected '{}' to be a directory".format(basedir)
    print("Reading presplit constituency data from: {}".format(basedir))

    assert destdir.is_dir(), "Expected '{}' to be a directory".format(destdir)
    print("Writing data splits to: {}".format(destdir))

    CONTENT_TYPE = ["nonterm", "term", "text"]

    splits = {
        "valid": valid_size,
        "test": test_size,
        "train": train_size,
    }
    ic(splits)

    for content_type in CONTENT_TYPE:
        path = basedir / ("{0}.txt".format(content_type))
        with open(path, "r") as fp_in:
            lines = fp_in.readlines()
        for split_name, split_size in splits.items():
            with open(destdir / ("{0}.{1}.txt".format(split_name, content_type)), "w") as fp_out:
                out_lines = lines[:split_size]
                lines = lines[split_size:]
                fp_out.write("".join(out_lines))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("Description")

    parser.add_argument(
        "BASE", type=str, metavar="FILE",
    )
    parser.add_argument(
        "DEST", type=str, metavar="FILE",
    )

    parser.add_argument(
        "--train-size", dest="train_size", type=int, required=False, default=sys.maxsize, metavar="NUM",
    )
    parser.add_argument(
        "--valid-size", dest="valid_size", type=int, required=False, default=10000, metavar="NUM",
    )
    parser.add_argument(
        "--test-size", dest="test_size", type=int, required=False, default=10000, metavar="NUM",
    )

    args = parser.parse_args()
    main(
        Path(args.BASE), Path(args.DEST), args.train_size, args.valid_size, args.test_size,
    )
