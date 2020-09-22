#
# Override these locally in local_settings.py
# 
MODEL_DIR = '/data/models'
DATASET_DIR = '/data/datasets'


try:
    from greynirseq.local_settings import *
except ImportError:
    pass