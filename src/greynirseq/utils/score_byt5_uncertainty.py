"""
Script that performs dropout Monte-Carlo uncertainty estimation with a ByT5 model from checkpoint file.

The output is a JSONL file with the following (line-wise) format:
{
    "mean": [float, float, ...],
    "std": [float, float, ...],
    "source": "string",
    "output": "string",
}
"""
import itertools
import json
from pathlib import Path

import click
import torch
import tqdm
from transformers import AutoTokenizer, T5ForConditionalGeneration

MAX_LENGTH = 512


def batch_by_n(iterable, batch_size):
    # collect into batches of n items, yielding after each batch
    _iter = iter(iterable)
    while True:
        # the list call is necessary since islice is a lazily evaluated generator
        batch = list(itertools.islice(_iter, batch_size))
        if not batch:
            break
        yield batch


def set_t5_dropout(model: torch.nn.Module, dropout: float):
    """
    Adjust the internal state of dropout modules in a T5 huggingface model.

    Storing the computation tree for backpropagation (called "training mode") more
      than quadruples the CUDA memory needed for a given batch!
    But the torch dropout modules have no "do-inference-at-test-time" hyperparameter,
      but we can produce that behavior by changing the internal state ourselves, i.e.
      only partially activate the "training mode" of the dropout modules.
    This does not mean that the memory usage is the same, since generation: is
    - is guaranteed to discard all of the computation tree
    - discards the decoder hidden states (only caches k and v values for the attention instead)
    - if the batch is unbalanced in terms of padding, the decoding prunes the short sequences
      and they don't cause the same memory overhead as the longest sequences like during training.
    """
    assert 0 <= dropout <= 1
    # flattened recursive traversal
    for mod in model.modules():
        if not isinstance(mod, torch.nn.Dropout):
            continue
        mod.p = dropout
        if dropout > 0:
            # Save some memory
            mod.inplace = True
            # Calling mod.train() causes the computation tree to be stored, which we don't want.
            # This may not be guaranteed by the module "Interface" (in java terminology),
            #   but at least in current version of torch (tested on CUDA)
            #   this causes dropout to be performed without storing most or none of the
            #   computation tree needed for backpropagation.
            mod.training = True
            assert mod.training
        else:
            # mod.inplace = False
            mod.eval()
            assert not mod.training


# fmt: off
@click.command()
@click.option( "--checkpoint-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option( "--input-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option( "--output-path", type=click.Path(exists=False, path_type=Path), required=True)
@click.option("--use-cpu", is_flag=True, default=False)
@click.option("--dropout", type=float, default=0.1)
@click.option("--seed", type=int, default=1)
@click.option("--num-iter", type=int, default=5)
@click.option("--batch-size", type=int, default=64)
def main(
    checkpoint_path,
    input_path,
    output_path,
    use_cpu,
    dropout,
    seed,
    num_iter,
    batch_size,
):
    # fmt: on
    # (assuming max_seqlen=512 tokens on 40GB A100)
    # batch size of 768 works for generation, but 14-16 is max for scoring
    #   however, using torch.no_grad allows scoring with batch size of
    #   at least 500 (but less than 768).
    # Using such high batch size is extremely suboptimal unless
    #   the sequence lengths in a batch are homogeneous. We can make
    #   the batches more uniform by sorting the input by length prior to batching.
    # [1000 samples total] @ 64 bsz got 69.3 sec with sorting
    #                        32 bsz got 83.6 sec with sorting
    #                       128 bsz got 67.4 sec with sorting
    #                       256 bsz got 79.4 sec with sorting

    print(f"Writing to {output_path} with dropout={dropout} and num_iter={num_iter}")
    use_gpu = not use_cpu
    if use_gpu:
        assert torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")

    print("Loading model...")
    # We are using the byte-level version, ByT5 (which is implemented using the same class)
    model = T5ForConditionalGeneration.from_pretrained(str(checkpoint_path))
    model = model.to("cuda").half() if use_gpu else model

    def prepare_model_inputs(data_lines):
        model_inputs = tokenizer(
            data_lines,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        if use_gpu:
            model_inputs = model_inputs.to("cuda")
        return model_inputs

    with open(input_path) as f:
        # strip is important
        src_lines = [line.strip() for line in f]

    # NOTE: this sorts the whole file, it would be premature optimization to do this with
    #  in a cleverer manner such as incremental partial sorting where we separately sort the
    #  first 10k, then sort 10k-20k, etc.
    seqlens = torch.tensor([len(line.encode("utf-8")) for line in src_lines]).long()
    sorted_indices = seqlens.argsort(descending=True)
    src_lines = [src_lines[i] for i in sorted_indices.tolist()]

    # parameters for model.generate
    default_generate_kwargs = {
        "max_length": MAX_LENGTH,
        "num_beams": 1,
        "output_scores": True,
        "return_dict_in_generate": False,
    }

    print("Generating...")
    model = model.eval()
    permuted_means = []
    permuted_stds = []
    decoded_outputs = []
    with torch.no_grad():
        for batch_idx, batch_lines in enumerate(
            batch_by_n(tqdm.tqdm(src_lines), batch_size=batch_size)
        ):
            # this call is very cheap
            set_t5_dropout(model, dropout=0.0)
            model_inputs = prepare_model_inputs(batch_lines)

            # generate hypothesis in eval mode
            model_outputs = model.generate(**model_inputs, **default_generate_kwargs)
            decoded_output = tokenizer.batch_decode(
                model_outputs, skip_special_tokens=True
            )
            decoded_outputs.extend(decoded_output)

            set_t5_dropout(model, dropout=dropout)

            # score the hypothesis with inference-time dropout
            iter_scores = []
            src_ids = model_inputs["input_ids"]
            # we need to pass the attention mask to the model (encoder attention mask)
            encoder_attention_mask = model_inputs["attention_mask"]
            tgt_ids = prepare_model_inputs(decoded_output)["input_ids"]
            cpu = torch.device("cpu")
            tgt_ids_cpu = tgt_ids.to(cpu)
            # after we get scores we move the rest of the work to the cpu
            decoder_target_mask = tgt_ids_cpu.eq(tokenizer.pad_token_id).logical_not()
            # we need to know how many tokens are in each sequence (to filter out padding tokens)
            lens = decoder_target_mask.sum(dim=1)
            # since the trailing batch may be smaller than batch_size
            for iteration_idx in range(num_iter):
                # for reproducibility
                batch_iter_seed = hash((seed, batch_idx, iteration_idx))
                _rng_gen = torch.manual_seed(batch_iter_seed)

                # get one monte-carlo iteration of scores with dropout
                output = model(
                    input_ids=src_ids,
                    labels=tgt_ids,
                    attention_mask=encoder_attention_mask,
                    use_cache=True,
                )
                unnormalized_score = output.logits.detach()
                del output
                # normalize the scores to get a (log) probability distribution
                # and move to cpu (so the reference counting can free it sooner)
                score = (
                    unnormalized_score.log_softmax(-1).detach().clone().float().to(cpu)
                )

                # Select out the scores of the tokens in the sequence (we get scores for the whole vocab inventory)
                # We have:
                # - array of indices I_ij
                # - array of floats  S_ijk
                # We want output matrix O_ij after using I as an index into S as follows:
                # O_ij = S_{i}{j}{I_ij}
                scores_without_padding = score.gather(
                    dim=2, index=tgt_ids_cpu.unsqueeze(-1)
                ).squeeze(-1)
                # filter out padding tokens
                scores_without_padding_tuples = scores_without_padding[
                    decoder_target_mask
                ].split(lens.tolist())
                iter_scores.append(scores_without_padding_tuples)

            # collect, reduce and store the scores
            actual_batch_size, _ = tgt_ids_cpu.shape
            for seq_index in range(actual_batch_size):
                seq_scores = [
                    iter_scores[iter_index][seq_index] for iter_index in range(num_iter)
                ]
                seq_scores = torch.stack(seq_scores, dim=0)
                seq_means = seq_scores.mean(dim=0)
                seq_stds = seq_scores.std(dim=0)
                permuted_means.append(seq_means)
                permuted_stds.append(seq_stds)

    # # recover the original order by inverting the length-sorted indices
    inverse_indices = sorted_indices.argsort().tolist()
    permuted_means = [permuted_means[i] for i in inverse_indices]
    permuted_stds = [permuted_stds[i] for i in inverse_indices]
    decoded_outputs = [decoded_outputs[i] for i in inverse_indices]

    # write results to disk
    with open(output_path, "w") as f:
        for mean, std, src_line, decoded_output in zip(
            permuted_means, permuted_stds, src_lines, decoded_outputs
        ):
            obj = {
                "mean": mean.tolist(),
                "std": std.tolist(),
                "source": src_line,
                "output": decoded_output,
            }
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")


if __name__ == "__main__":
    main()
