"""
"""
import copy
import itertools
import time
from pathlib import Path

import click
import torch
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


# fmt: off
@click.command()
@click.option("--checkpoint-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--input-path", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--output-path", type=click.Path(exists=False, path_type=Path), required=True)
@click.option("--use-cpu", is_flag=True, default=False)
@click.option("--num-beams", type=int, default=5)
@click.option("--dropout", type=float, default=0.0)
@click.option("--batch-size", type=int, default=64)
def main(checkpoint_path, input_path, output_path, use_cpu, num_beams, dropout, batch_size):
    # fmt: on
    use_gpu = not use_cpu
    if use_gpu:
        assert torch.cuda.is_available()

    tokenizer = AutoTokenizer.from_pretrained("google/byt5-base")
    #   this may require not being in eval mode (i.e. computation graph is cached for gradient)

    model = T5ForConditionalGeneration.from_pretrained(str(checkpoint_path))
    if dropout > 0.0:
        # Inference-time dropout.
        # To make sure the correct dropout value is propagated properly,
        #   we do a "round trip" by reconstructing it with a modified config.
        config = copy.deepcopy(model.config)
        config.dropout_rate = dropout
        model = T5ForConditionalGeneration.from_pretrained(str(checkpoint_path), config=config)
        model = model.train()
    else:
        model = model.eval()
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
            model_inputs.to("cuda")
        return model_inputs

    with open(input_path) as f:
        # strip is important
        test_lines = [line.strip() for line in f]
    
    # parameters for model.generate
    default_generate_kwargs = {
        "max_length": MAX_LENGTH,
        "num_beams": num_beams,
        "output_scores": True,
        "return_dict_in_generate": False,
    }

    start_time = time.time()
    # make sure output dir exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as out_f:
        for batch_lines in batch_by_n(test_lines, batch_size=batch_size):
            model_inputs = prepare_model_inputs(batch_lines)

            model_output = model.generate(**model_inputs, **default_generate_kwargs)

            decoded_output = tokenizer.batch_decode(
                model_output, skip_special_tokens=True
            )
            for line in decoded_output:
                out_f.write(line + "\n")

    end_time = time.time()
    print(f"Finished {output_path} in {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
