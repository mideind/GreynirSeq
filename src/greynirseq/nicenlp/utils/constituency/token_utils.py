import tokenizer


def tokenize(text, allow_multiword=False):
    mw_tokens = [tok.txt for tok in tokenizer.tokenize(text) if tok.txt is not None]
    if allow_multiword:
        return mw_tokens
    tokens = []
    for mw_token in mw_tokens:
        tokens.extend(mw_token.split(" "))
    return tokens


def tokenize_to_string(text):
    return " ".join(tokenize(text))


