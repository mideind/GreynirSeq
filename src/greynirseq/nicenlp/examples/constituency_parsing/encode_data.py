# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.


import sys

import tokenizers

tokenizer = tokenizers.ByteLevelBPETokenizer(
    "/data/models/icebert/icebert-first-rmh_different_vocab/icebert-base-36k/icebert-bpe-vocab.json",
    "/data/models/icebert/icebert-first-rmh_different_vocab/icebert-base-36k/icebert-bpe-merges.txt",
    add_prefix_space=True,
)

in_path = sys.argv[1]
out_path = sys.argv[2]

with open(in_path, "r", encoding="utf8") as text_in:
    with open(out_path, "w", encoding="utf8") as text_out:
        for line in text_in:
            fields = [" ".join(str(id) for id in tokenizer.encode(field).ids) for field in line.strip("\n").split("\t")]
            text_out.write("\t".join(fields))
            text_out.write("\n")
