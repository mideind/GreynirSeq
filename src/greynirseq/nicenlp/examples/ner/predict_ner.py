# flake8: noqa

import time

from greynirseq.ner.ner_f1_stats import EvalNER
from greynirseq.nicenlp.models.multiclass import MultiClassRobertaModel
from greynirseq.settings import IceBERT_NER_CONFIG, IceBERT_NER_PATH

model = MultiClassRobertaModel.from_pretrained(IceBERT_NER_PATH, **IceBERT_NER_CONFIG)
model.to("cpu")
model.eval()

dataset_name = "valid"
dataset = model.task.load_dataset(dataset_name)
dataset_size = dataset.sizes[0].shape[0]
ldict = model.task.label_dictionary
lbl_shift = ldict.nspecial
batch_size = 1
symbols = model.task.label_dictionary.symbols[model.task.label_dictionary.nspecial :]
eval_ner = EvalNER(symbols)

for dataset_offset in range(dataset_size):
    start = time.time()
    sample = dataset.collater([dataset[idx_] for idx_ in range(dataset_offset, dataset_offset + batch_size)])
    ntokens = sample["net_input"]["nsrc_tokens"]
    tokens = [tokens for tokens in sample["net_input"]["src_tokens"]]
    sentences = [model.decode(seq[: ntokens[seq_idx]]) for seq_idx, seq in enumerate(tokens)]
    seq_idx = 0
    target_idxs = sample["target_attrs"][seq_idx]
    pred_labels, pred_idxs = model.predict_labels(sentences[0])
    eval_ner.compare(pred_idxs, target_idxs)
    eval_ner.print_all_stats()
