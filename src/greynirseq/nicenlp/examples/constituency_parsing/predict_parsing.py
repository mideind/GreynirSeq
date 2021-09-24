# flake8: noqa

import time
from pathlib import Path

import numpy as np
import torch

import greynirseq.nicenlp.utils.constituency.tree_dist as tree_dist
from greynirseq.nicenlp.utils.constituency.token_utils import tokenize
from greynirseq.nicenlp.criterions.parser_criterion import compute_parse_stats, safe_div, f1_score
from greynirseq.nicenlp.models.simple_parser import SimpleParserModel
from greynirseq.nicenlp.utils.constituency.greynir_utils import Node
from greynirseq.nicenlp.utils.label_schema.label_schema import make_dict_idx_to_vec_idx

try:
    from icecream import ic

    ic.configureOutput(includeContext=True)
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

model = SimpleParserModel.from_pretrained(  # pylint: disable=undefined-variable
    "/data/scratch/haukur/parser/checkpoints/pretrain-02",
    checkpoint_file="checkpoint30.pt",
    data_name_or_path="/data/scratch/haukur/parser/data/bin-finetune",
)
model.to("cpu")
model.eval()

np.random.seed(12345)

torch.set_printoptions(precision=4, linewidth=160)

# dataset_name = "train"
# dataset_name = "valid.cfg"
dataset_name = "valid"
dataset = model.task.load_dataset(dataset_name)
dataset_size = dataset.sizes[0].shape[0]
ldict = model.task.nterm_dictionary
lbl_shift = ldict.nspecial
batch_size = 2

for dataset_offset in np.random.randint(0, dataset_size, 100):
    start = time.time()
    sample = dataset.collater([dataset[idx_] for idx_ in range(dataset_offset, dataset_offset + batch_size)])
    ntokens = sample["net_input"]["nsrc_tokens"]
    tokens = [tokens for tokens in sample["net_input"]["src_tokens"]]
    sentences = [model.decode(seq[: ntokens[seq_idx]]) for seq_idx, seq in enumerate(tokens)]

    seq_idx = 0
    ntargets = sample["ntarget_span_labels"][seq_idx]
    targets = sample["target_span_labels"][seq_idx]
    target_spans = sample["target_spans"][seq_idx, : 2 * ntargets].reshape(-1, 2)
    labelled_constituents = [(ii, jj, target) for ((ii, jj), target) in zip(target_spans.tolist(), targets.tolist())]
    # should already be pre-sorted, but just in case
    labelled_constituents = sorted(labelled_constituents, key=lambda x: (x[0], -x[0]))
    seq_spans = [(it[0], it[1]) for it in labelled_constituents]
    seq_label_idxs = [it[2] for it in labelled_constituents]
    seq_labels = [model.task.nterm_dictionary.symbols[idx] for idx in seq_label_idxs]

    pred_trees, (lspans, span_labels, _lmask) = model.predict_sample(sample)
    ic("predicted tree")
    pred_tree = pred_trees[0]
    pred_tree.pretty_print()
    pred_roof = pred_tree.roof()

    gold_tree = Node.from_labelled_spans(
        seq_spans, seq_labels, tokenize(sentences[seq_idx].lstrip())  # pylint: disable=undefined-variable
    )
    gold_tree = gold_tree.debinarize()
    ic(dataset_name)
    gold_tree.pretty_print()

    nterm_dict_idx_to_vec_idx = make_dict_idx_to_vec_idx(
        model.task.nterm_dictionary,
        model.task.nterm_schema.label_categories,
        device=ntargets.device,
    )

    parse_stats = compute_parse_stats(
        _lmask,
        nterm_dict_idx_to_vec_idx[sample["target_span_labels"]].cpu(),
        sample["target_spans"].reshape(batch_size, -1, 2).transpose(2, 1).cpu(),
        sample["ntarget_span_labels"].cpu(),
    )
    gold_roof = gold_tree.roof()

    seq_ii, seq_jj = zip(*seq_spans)
    prec = round(safe_div(parse_stats.ncorrect, parse_stats.npred), 3)
    recall = round(safe_div(parse_stats.ncorrect, parse_stats.ngold), 3)
    f1 = round(f1_score(prec, recall), 3)

    ic(parse_stats, (prec, recall, f1))
    gold_pred_tree_dist = int(tree_dist.tree_dist(gold_tree, pred_tree, None))
    gold_pred_roof_dist = int(tree_dist.tree_dist(gold_roof, pred_roof, None))
    unif_gold_pred_dist = int(tree_dist.tree_dist(gold_roof.uniform(), pred_roof.uniform(), None))
    ic(gold_pred_roof_dist, unif_gold_pred_dist)

    time_per_seq_in_ms = round(1000*(time.time() - start) / batch_size, 3)
    ic(time_per_seq_in_ms)
    input("Press enter to continue...")


sentences = [
    "Hundurinn ákvað að fara í sund.",
    "Leitarvél Greynis byggir á umfjöllunarefnum (þemum) frekar en bókstaflegum leitarorðum.",
    "Bestu niðurstöður fást með því að skrifa hnitmiðaðar setningar sem líkjast þeim niðurstöðum sem óskað er eftir.",
    "Mannanöfn og sérnöfn hafa þó sérstaka vigt.",
    "Portúgal sigraði Eurovision með hugljúfu lagi.",
    "Mig langar að elda fisk í kvöldmatinn.",
    "Verið er að byggja fjölda hótela á landinu.",
    "Einkaneysla fer hraðvaxandi.",
    "Sláðu inn leitarstreng eða spurningu um mannanöfn, titla, starfsheiti eða sérnöfn hér að ofan.",
]
# while True:
for sentence in sentences:
    ic(model.predict([sentence]))
    # inp = input()
    # print(inp)
    # pred_tree, presult = model.predict([inp])
    # breakpoint()
    # pred_tree = pred_tree.debinarize()
    # pred_tree.pretty_print()
    # input()

# import code; code.interact(local=locals())
