from greynirseq.nicenlp.models.icebert import *
from greynirseq.nicenlp.tasks.wordclassification import *
from greynirseq.nicenlp.criterions.multi_label import *


def predict(lab):

    ib = IcebertModel.from_pretrained(
        "bin/checkpoints_" + lab,
        checkpoint_file="checkpoint_last.pt",
        data_name_or_path="/home/vesteinn/work/pytorch_study/data/MIM/MIM-GOLD-1_0_sets_for_training/{}/bin".format(
            lab
        ),
    )
    ib.eval()

    ib.predict_file(
        "/home/vesteinn/work/pytorch_study/data/MIM/MIM-GOLD-1_0_sets_for_training/{}/valid.input0".format(
            lab
        ),
        "/home/vesteinn/work/pytorch_study/data/MIM/MIM-GOLD-1_0_sets_for_training/{}/valid.label0".format(
            lab
        ),
        device="cpu",
    )


labels = []
for i in range(1, 10):
    labels.append("0{}".format(i))
labels.append("10")

for lab in labels:
    print("------ POS TAG {} -----".format(lab))
    predict(lab)
