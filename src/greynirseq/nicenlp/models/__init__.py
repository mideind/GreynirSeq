# We extend the fairseq BART models to include our pretrained models as well
from typing import Dict

from fairseq.models.bart.model import BARTModel
from fairseq.models.transformer import TransformerModel

# BPE needs to be set on from_pretrained since the default value is gpt2 in BARTModel.
# Note: These keys are also used in greynirseq.py (the cli).
# They are expected to end in '.xxxx', where xxxx is the src_lang and tgt_lang.
MBART_PRETRAINED_MODELS = {
    "mbart25-cont-ees-enis": {"path": "https://data.greynir.is/mbart25-ees-enis.tar.gz"},
    "mbart25-cont-ees-isen": {"path": "https://data.greynir.is/mbart25-ees-isen.tar.gz"},
}
TF_PRETRAINED_MODELS = {
    "transformer-enis": {
        "path": "https://data.greynir.is/eng-isl-base-v6.tar.gz",
        "gpt2_encoder_json": "https://data.greynir.is/eng-isl-bbpe-32k/vocab.json",
        "gpt2_vocab_bpe": "https://data.greynir.is/eng-isl-bbpe-32k/merges.txt",
        "prepend_bos": False,
    },
    "transformer-isen": {
        "path": "https://data.greynir.is/isl-eng-base-v6.tar.gz",
        "gpt2_encoder_json": "https://data.greynir.is/eng-isl-bbpe-32k/vocab.json",
        "gpt2_vocab_bpe": "https://data.greynir.is/eng-isl-bbpe-32k/merges.txt",
        "prepend_bos": True,
    },
}  # type: ignore

hub_models: Dict[str, str] = BARTModel.hub_models()
hub_models.update(MBART_PRETRAINED_MODELS)  # type: ignore
BARTModel.hub_models = lambda: hub_models

tf_hub_models: Dict[str, str] = TransformerModel.hub_models()
tf_hub_models.update(TF_PRETRAINED_MODELS)  # type: ignore
TransformerModel.hub_models = lambda: tf_hub_models
