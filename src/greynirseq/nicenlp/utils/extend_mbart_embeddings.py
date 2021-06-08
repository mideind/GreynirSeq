from pathlib import Path
import torch


def main(in_path, out_path, args):
    checkpoint = torch.load(in_path, map_location=torch.device("cpu"))
    model = checkpoint["model"]
    enc_w = model["encoder.embed_tokens.weight"]
    dec_w = model["decoder.embed_tokens.weight"]

    assert torch.all(enc_w.eq(dec_w)), "Input and output embeddings should be shared"

    vocab_size, nfeatures = enc_w.shape
    new_vec = enc_w[args.slice_start : args.slice_end].mean(dim=0)

    if args.noise_scale > 0:
        noise_vec = args.noise_scale * new_vec.new_zeros(nfeatures).uniform_()
        new_vec += noise_vec

    insertion_index = args.insertion_index
    if args.insertion_index == -1:
        insertion_index = vocab_size - 2  # vocab_size - 1 is <mask>
    new_emb_mat = enc_w.new_zeros(vocab_size + 1, nfeatures)
    new_emb_mat[:insertion_index] = enc_w[:insertion_index]
    new_emb_mat[insertion_index] = new_vec
    new_emb_mat[insertion_index + 1:] = enc_w[insertion_index:]

    model["encoder.embed_tokens.weight"] = new_emb_mat
    model["decoder.embed_tokens.weight"] = new_emb_mat.clone()
    torch.save(checkpoint, out_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Vocab extension - add new language id to mBART, ")

    parser.add_argument(
        "input",
        type=str,
        help="Input model file",
    )
    parser.add_argument(
        "output",
        type=str,
        help="Output file",
    )
    parser.add_argument(
        "--slice-start",
        type=int,
        required=True,
        help="First embedding index of slice that will be averaged (mBART has 250.027 embeddings)",
    )
    parser.add_argument(
        "--slice-end",
        type=int,
        required=True,
        help="Last embedding index of slice that will be averaged",
    )
    parser.add_argument(
        "--insertion-index",
        type=int,
        default=-1,
        required=False,
        help="Location of new language id (if set to -1, will insert as second to last)",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.01,
        required=False,
        help="Add uniform noise to averaged embedding (default scale is 0.01). Two language id vectors x1 and x2 have mean(abs(x2-x1)) in the range 0.03 to 0.05 or so.",
    )

    args = parser.parse_args()

    if not Path(args.input).exists():
        raise ValueError(f"Could not find file {args.input}")

    main(args.input, args.output, args)
