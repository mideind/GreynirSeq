#!/usr/bin/env python3

import argparse

from greynirseq.utils.ifd_utils import ifd2labels


def load_file(input, output_folder, prefix, sep="<SEP>"):
    o_d = open("{}/{}.input0".format(output_folder, prefix), "w")
    o_l = open("{}/{}.label0".format(output_folder, prefix), "w")

    sep = f" {sep} "

    with open(input) as fp:
        line = fp.readline()
        inp = None
        lab = None
        while line:
            if inp is not None:
                o_d.writelines("{}\n".format(" ".join(inp)))
            if lab is not None:
                o_l.writelines(
                    "{}\n".format(" <SEP> ".join([" ".join(slab) for slab in lab]))  # pylint: disable=not-an-iterable
                )
            inp = []
            lab = []
            while line and line != "\n":
                line = line.strip().split()
                if len(line):
                    inp.append(line[0])
                    lab.append(ifd2labels(line[1]))
                line = fp.readline()
            line = fp.readline()

    o_d.close()
    o_l.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output_folder")
    parser.add_argument("--prefix")
    parser.add_argument("--sep")
    args = parser.parse_args()

    load_file(args.input, args.output_folder, args.prefix, args.sep)


if __name__ == "__main__":
    main()
