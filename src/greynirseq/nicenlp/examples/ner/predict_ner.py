import torch
import numpy as np
import time
import tqdm

from greynirseq.nicenlp.data.datasets import *
from greynirseq.nicenlp.models.multi_span_model import *
from greynirseq.nicenlp.tasks.multi_span_prediction_task import *
from greynirseq.nicenlp.criterions.multi_span_prediction_criterion import *
from greynirseq.utils.ner import EvalNER
from greynirseq.nicenlp.utils.greynir.greynir_utils import Node
import greynirseq.nicenlp.utils.greynir.tree_dist as tree_dist

model = IcebertConstModel.from_pretrained(
    '/media/hd/MIDEIND/data/models/icebert_ner/ner_slset',
    checkpoint_file='checkpoint_last.pt',
    data_name_or_path='/media/hd/MIDEIND/data/models/MIM-GOLD-NER/8_entity_types/bin/bin',
    gpt2_encoder_json="/media/hd/MIDEIND/data/models/icebert-base-36k/icebert-bpe-vocab.json",  
    gpt2_vocab_bpe="/media/hd/MIDEIND/data/models/icebert-base-36k/icebert-bpe-merges.txt",
    term_schema="/media/hd/MIDEIND/data/models/MIM-GOLD-NER_split/term.json"
)
model.to("cpu")
model.eval()

dataset_name = "test"
dataset = model.task.load_dataset(dataset_name)
dataset_size = dataset.sizes[0].shape[0]
ldict = model.task.label_dictionary
lbl_shift = ldict.nspecial
batch_size = 1

eval_ner = EvalNER(model)

for dataset_offset in range(dataset_size): #np.random.randint(0, dataset_size, 100):
    start = time.time()
    sample = dataset.collater([dataset[idx_] for idx_ in range(dataset_offset, dataset_offset + batch_size)])
    ntokens = sample["net_input"]["nsrc_tokens"]
    tokens = [tokens for tokens in sample["net_input"]["src_tokens"]]
    sentences = [model.decode(seq[:ntokens[seq_idx]]) for seq_idx, seq in enumerate(tokens)]
    seq_idx = 0
    target_cats = sample["target_cats"][seq_idx]
    pred_cats, labels, tokenized = model.predict_sample_pos(
        sample,
        sentences,
        device="cpu"
    )
    eval_ner.compare(res[0], pred_cats)
    eval_ner.print_all_stats()

 