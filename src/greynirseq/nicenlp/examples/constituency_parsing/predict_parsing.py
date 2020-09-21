import torch
import numpy as np
import time

try:
    from icecream import ic
    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from greynirseq.nicenlp.data.datasets import *
from greynirseq.nicenlp.models.multi_span_model import *
from greynirseq.nicenlp.tasks.multi_span_prediction_task import *
from greynirseq.nicenlp.criterions.multi_span_prediction_criterion import *
from greynirseq.nicenlp.utils.greynir.greynir_utils import Node
import greynirseq.nicenlp.utils.greynir.tree_dist as tree_dist

model = IcebertConstModel.from_pretrained(
    '/data/models/constituency_parser/icebert_const_debug_07',
    checkpoint_file='checkpoint_last.pt',
    data_name_or_path='/home/haukur/github/nicenlp-icebert/data-bin/sym',
)
model.to("cpu")
model.eval()

np.random.seed(12345)

torch.set_printoptions(precision=4, linewidth=160)

# dataset_name = "train"
# dataset_name = "valid.cfg"
dataset_name = "valid.gold"
dataset = model.task.load_dataset(dataset_name)
dataset_size = dataset.sizes[0].shape[0]
ldict = model.task.label_dictionary
lbl_shift = ldict.nspecial
batch_size = 1

for dataset_offset in np.random.randint(0, dataset_size, 100):
    start = time.time()
    sample = dataset.collater([dataset[idx_] for idx_ in range(dataset_offset, dataset_offset + batch_size)])
    ntokens = sample["net_input"]["nsrc_tokens"]
    tokens = [tokens for tokens in sample["net_input"]["src_tokens"]]
    sentences = [model.decode(seq[:ntokens[seq_idx]]) for seq_idx, seq in enumerate(tokens)]

    seq_idx = 0
    ntargets = sample["ntargets"][seq_idx]
    targets = sample["targets"][seq_idx]
    target_spans = sample["target_spans"][seq_idx, : 2 * ntargets].reshape(-1, 2)
    labelled_constituents = [
        (ii, jj, target)
        for ((ii, jj), target) in zip(target_spans.tolist(), targets.tolist())
    ]
    # should already be pre-sorted, but just in case
    labelled_constituents = sorted(labelled_constituents, key=lambda x: (x[0], -x[0]))
    seq_spans = [(it[0], it[1]) for it in labelled_constituents]
    seq_label_idxs = [it[2] for it in labelled_constituents]
    seq_labels = [model.task.label_dictionary.symbols[idx] for idx in seq_label_idxs]

    # pred_tree, pred_spans = model.predict_sample(sample, sentences)
    pred_tree, presult = model.predict_sample(sample, sentences)
    pred_tree = pred_tree.debinarize()
    ic("predicted tree")
    pred_tree.pretty_print()
    pred_roof = pred_tree.roof()

    gold_tree = Node.from_labelled_spans(seq_spans, seq_labels, tokenize(sentences[seq_idx]))
    gold_tree = gold_tree.debinarize()
    ic(dataset_name)
    gold_tree.pretty_print()
    gold_roof = gold_tree.roof()

    seq_ii, seq_jj = zip(*seq_spans)
    ncorr = (
        (presult.masked_lchart[seq_ii, seq_jj] == (torch.tensor(seq_label_idxs) - lbl_shift))
        * presult.masked_lchart[seq_ii, seq_jj].gt(1)
    ).sum().item()
    npred = presult.labels.gt(1).sum().item()
    ngold = targets.gt(model.task.label_dictionary.nspecial + 1).sum().item()
    prec = round(ncorr / (npred or 1), 3)
    recall = round(ncorr / ngold, 3)
    f1 = round(2 * (prec * recall)/((prec + recall) or 1), 3)

    ic((ncorr, npred, ngold), (prec, recall, f1))
    gold_pred_tree_dist = int(tree_dist.tree_dist(gold_tree, pred_tree, None))
    gold_pred_roof_dist = int(tree_dist.tree_dist(gold_roof, pred_roof, None))
    unif_gold_pred_dist = int(tree_dist.tree_dist(gold_roof.uniform(), pred_roof.uniform(), None))
    ic(gold_pred_roof_dist)
    ic(unif_gold_pred_dist)

    lbls = presult.labels

    ic((time.time() - start)/batch_size)
    input("Press enter to continue...")


# sentences = [
#     "Hundurinn ákvað að fara í sund.",
#     "Leitarvél Greynis byggir á umfjöllunarefnum (þemum) frekar en bókstaflegum leitarorðum.",
#     "Bestu niðurstöður fást með því að skrifa hnitmiðaðar setningar sem líkjast þeim niðurstöðum sem óskað er eftir.",
#     "Mannanöfn og sérnöfn hafa þó sérstaka vigt.",
#     "Portúgal sigraði Eurovision með hugljúfu lagi.",
#     "Mig langar að elda fisk í kvöldmatinn.",
#     "Verið er að byggja fjölda hótela á landinu.",
#     "Einkaneysla fer hraðvaxandi.",
#     "Sláðu inn leitarstreng eða spurningu um mannanöfn, titla, starfsheiti eða sérnöfn hér að ofan.",
# ]
# # for sentence in sentences:
# while True:
#     # model.predict([sentence])
#     inp = input()
#     print(inp)
#     pred_tree, presult = model.predict([inp])
#     pred_tree = pred_tree.debinarize()
#     pred_tree.pretty_print()
#     # input()

# import code; code.interact(local=locals())

