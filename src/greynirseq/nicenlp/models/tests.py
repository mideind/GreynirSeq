import unittest

import torch

from greynirseq.nicenlp.models.icebert import MultiLabelClassificationHead


class TestIcebert(unittest.TestCase):
    def test_tokens_to_range_sum(self):
        word_starts = [1, 0, 0, 1, 1, 1, 0, 1]
        bpe_token_features = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        inner_dim = 1
        word_features = MultiLabelClassificationHead.tokens_to_range_sum(
            [word_starts], [[bpe for bpe in bpe_token_features]], 1
        )
        word_start_idxs = MultiLabelClassificationHead.get_word_start_indexes(
            word_starts
        )
        assert word_start_idxs == [0, 3, 4, 5, 7]
        assert word_features.tolist() == [[[6.0], [4.0], [5.0], [13.0], [8.0]]]
