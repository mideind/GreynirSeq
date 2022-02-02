# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.


import tokenizer


def tokenize(text, allow_multiword=False):
    mw_tokens = [tok.txt for tok in tokenizer.tokenize(text) if tok.txt is not None and tok.txt]
    if allow_multiword:
        return mw_tokens
    tokens = []
    for mw_token in mw_tokens:
        tokens.extend(mw_token.split(" "))
    return tokens


def tokenize_to_string(text, add_prefix_space=False):
    ret = " ".join(tokenize(text))
    if not ret:
        return ret
    if not add_prefix_space or ret[0] == " ":
        return ret
    return " " + ret
