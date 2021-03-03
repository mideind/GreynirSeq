from pathlib import Path
from itertools import zip_longest

import numpy as np
import torch
from fairseq import hub_utils
from fairseq.data import data_utils

import tokenizer

from greynirseq.nicenlp.models.bytebert import ByteBertModel
from greynirseq.nicenlp.tasks.byte_masked_lm import ByteMaskedLMTask
from greynirseq.nicenlp.tasks.byte_masked_lm import calculate_offsets, calculate_byte_offsets

try:
    from icecream import ic
    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

SCRATCH = Path("/data/scratch/haukur/bytebert")
# CKPT_DIR = SCRATCH / "checkpoints_w_token_blocks"
CKPT_DIR = SCRATCH / "checkpoints_no_noise"
CKPT_NAME = "checkpoint_best.pt"
DATA_DIR = SCRATCH / "data"

ENCODER_JSON = SCRATCH / "vocab.json"
VOCAB_BPE = SCRATCH / "merges.txt"

assert (CKPT_DIR / CKPT_NAME).exists()

ckpt = hub_utils.from_pretrained(
    str(CKPT_DIR),
    "checkpoint_best.pt",
    str(DATA_DIR),
    bpe="gpt2",
    gpt2_encoder_json=str(ENCODER_JSON),
    gpt2_vocab_bpe=str(VOCAB_BPE),
)
task = ckpt["task"]
model = ckpt["models"][0]
model.eval()
assert not model.training
args_ = ckpt["args"]

ic(type(ckpt), ckpt.keys(), task, type(model), args_)

text_idx = 11
# all_text = [line.strip("\n") for line in (DATA_DIR / "train.txt").open().readlines()]
all_text = [line.strip("\n") for line in (DATA_DIR / "train.txt").open().readlines()]

# text = all_text[text_idx:text_idx + 1]
# offsets = [[int(word) for word in line.split(" ")] for line in (DATA_DIR / "train.offsets.byte").open().readlines()][text_idx:text_idx + 1]
# ic(text[text_idx:text_idx + 1], offsets[text_idx:text_idx + 1])

seed = 1
for i in range(10):
    with data_utils.numpy_seed(seed, 0, i):
        line = all_text[i]

        # _, _char_offsets, byte_offsets = list(calculate_offsets(line + "\n"))
        byte_offsets = calculate_byte_offsets(line)
        num_ptoks = len(byte_offsets)
        byte_offsets.append(len(line.encode()))
        idx_masked_word = np.random.randint(num_ptoks) # randomly pick masked word location
        # line[byte_offsets[idx_masked_word] : byte_offsets[idx_masked_word + 1]] = "<mask>"
        line = line.encode()
        # masked_line = line[:byte_offsets[idx_masked_word]].decode() + "<mask><mask><mask>" + line[byte_offsets[idx_masked_word + 1]:].decode()
        masked_line = line[:byte_offsets[idx_masked_word]].decode() + "<mask>" + line[byte_offsets[idx_masked_word + 1]:].decode()
        ic(line.decode("utf8"), masked_line, byte_offsets)
        # ic([(start, line[start:].decode()) for start in byte_offsets])
        mask_fill_dataset = task.prepare_sentence_for_fill_masks(masked_line)
        ic(mask_fill_dataset[0])

        # mask_fill_dataset = task.prepare_sentence_for_fill_masks("Ég er <mask>ur, en ekki þú!")

        # dataset = task.prepare_sentences(text, offsets, dropout=0.1)
        dataset = mask_fill_dataset

        # ic(dataset, dataset[0])
        sample = dataset.collater([dataset[0]])
        # ic(sample)

        features = model(**sample["net_input"], features_only=True)
        # ic(features)

        lm_head_pred, _extra = model(
            **sample["net_input"],
            features_only=False,
            masked_tokens=torch.ones_like(sample["net_input"]["pool_lengths"]).bool() * sample["net_input"]["pool_lengths"].gt(0),
        )
        pred_toks = []
        mask_locs = sample["target"].squeeze().eq(task.bpe_dictionary.index("<mask>")).nonzero(as_tuple=True)[0]
        for mask_loc in mask_locs.tolist():
            output_ids = lm_head_pred.max(dim=-1).indices
            pred_logits, pred_idxs = (-lm_head_pred[mask_loc]).squeeze().sort()
            preds_bpe = [task.decode([idx]) for idx in pred_idxs[:10]]
            pred_toks.append(preds_bpe[0])
            pred_logits = [round(-it, 5) for it in pred_logits.tolist()]
            ic(list(zip(pred_logits, preds_bpe, pred_idxs.tolist())))
        # interleave_longest (cf. zip_longest)
        line_parts = masked_line.split("<mask>")
        interleaved_parts = [item for pair in zip_longest(line_parts, pred_toks) for item in pair if item]
        output_pred_line = "".join(interleaved_parts)
        ic(output_pred_line)

        # output_toks = task.decode(output_ids)
        # reg_out_toks = [task.decode([idx]) for idx in output_ids]
        if mask_locs.numel() == 1:
            targets = [task.decode([idx]) for idx in sample["target"].squeeze()]
            ic(targets)
        # ic(list(zip(targets, reg_out_toks, output_ids)))

        # runner_up = lm_head_pred.clone()
        # mask = torch.zeros_like(lm_head_pred).bool()
        # max_idxs = torch.stack([torch.arange(len(output_ids)), output_ids]).tolist()
        # mask[max_idxs] = 1
        # runner_up[mask] = runner_up.min()
        # runner_up_ids = runner_up.max(dim=-1).indices
        # ic(task.decode(runner_up_ids))
        # runner_up_toks = [task.decode([idx]) for idx in runner_up_ids]
        # ic(list(zip(reg_out_toks, runner_up_toks)))

        import pdb; pdb.set_trace()

        print()
