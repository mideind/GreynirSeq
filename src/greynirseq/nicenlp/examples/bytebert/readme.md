# ByteBERT #
You first need to generate a vocab with tokenizers from HuggingFace, for this see `greynirseq/utils/train_byte_bpe.py`.

See (and modify) `prepare_data.sh` in order to generate token offsets.

Training data is assumed to be `./data/train.txt` and validation data is assumed to be `./data/valid.txt` .

