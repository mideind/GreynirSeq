#
# If needed, override these locally in local_settings.py
#
import os

MODEL_DIR = "/data/models"
DATASET_DIR = "/data/datasets"

IceBERT_NER_PATH = os.path.join(MODEL_DIR, "icebert_ner/ner_slset")
IceBERT_NER_CONFIG = {
    "checkpoint_file": "checkpoint_last.pt",
    "data_name_or_path": os.path.join(DATASET_DIR, "MIM-GOLD-NER/8_entity_types/bin/bin"),
    "gpt2_encoder_json": os.path.join(MODEL_DIR, "icebert-base-36k/icebert-bpe-vocab.json"),
    "gpt2_vocab_bpe": os.path.join(MODEL_DIR, "icebert-base-36k/icebert-bpe-merges.txt"),
}

IceBERT_POS_PATH = os.path.join(MODEL_DIR, "icebert_pos")
IceBERT_POS_CONFIG = {
    "checkpoint_file": "checkpoint_last.pt",
    "gpt2_encoder_json": os.path.join(MODEL_DIR, "icebert-base-36k/icebert-bpe-vocab.json"),
    "gpt2_vocab_bpe": os.path.join(MODEL_DIR, "icebert-base-36k/icebert-bpe-merges.txt"),
}

try:
    from greynirseq.local_settings import *  # noqa
except ImportError:
    pass
