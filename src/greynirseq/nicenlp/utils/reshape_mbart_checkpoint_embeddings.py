from pathlib import Path

import torch
from fairseq import utils


def main(in_path, out_path, seed, num_extra_vecs):
    checkpoint = torch.load(in_path, map_location=torch.device("cpu"))
    model = checkpoint["model"]

    enc_w = model["encoder.embed_tokens.weight"]
    dec_w = model["decoder.embed_tokens.weight"]

    old_vocab_size, nfeatures = enc_w.shape
    print("Old embedding shape:", [old_vocab_size, nfeatures])
    new_vocab_size = old_vocab_size + num_extra_vecs
    print("New embedding shape:", [new_vocab_size, nfeatures])

    assert torch.all(enc_w.eq(dec_w)), "Expecting input and output embeddings should be shared"

    utils.set_torch_seed(seed)
    new_vecs = enc_w.new_zeros(num_extra_vecs, nfeatures).uniform_()
    torch.nn.init.xavier_uniform(new_vecs)

    new_emb_mat = enc_w.new_zeros(new_vocab_size, nfeatures)
    new_emb_mat[:old_vocab_size, :] = enc_w[:, :]
    new_emb_mat[old_vocab_size:, :] = new_vecs

    model["encoder.embed_tokens.weight"] = new_emb_mat
    model["decoder.embed_tokens.weight"] = new_emb_mat.clone()
    model["decoder.output_projection.weight"] = new_emb_mat.clone()
    torch.save(checkpoint, out_path)


if __name__ == "__main__":
    import argparse

    # fmt: off
    parser = argparse.ArgumentParser(
        "Embedding extension",
        description="Extend the embedding matrix of a Pytorch model checkpoint "
                    "by a set of vectors. The k new embeddings are created "
                    "with Xavier intialization",
    )

    parser.add_argument("--input", type=str, help="Input checkpoint file", required=True)
    parser.add_argument("--output", type=str, help="Output checkpoint file", required=True)
    parser.add_argument("--nvecs", type=int, help="Number of additional vectors", required=True)
    parser.add_argument("--seed", type=int, default=1)
    # fmt: on

    args = parser.parse_args()

    if not Path(args.input).exists():
        raise ValueError(f"Could not find file {args.input}")

    main(args.input, args.output, args.seed, args.nvecs)
