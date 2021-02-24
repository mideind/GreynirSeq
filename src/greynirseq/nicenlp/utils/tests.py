import unittest

import torch

from greynirseq.nicenlp.utils.logits_filter import filter_max_logits, max_tensor_by_bins, word_classes_to_mask


class TestLogitsFilter(unittest.TestCase):
    def test_filter_max_logits(self):
        unique_together = torch.tensor([2, 3])
        logits = torch.tensor([[[1, 2, 3, 4], [4, 3, 2, 1]]])
        expected = [[[1, 2, 0, 4], [4, 3, 2, 0]]]
        filtered_logits = filter_max_logits(logits, unique_together)
        assert torch.eq(filtered_logits, torch.tensor(expected)).all()

    def test_max_tensor_by_bins(self):
        bins = [(0, 1), (1, 3), (3, 7)]
        logits = torch.tensor([range(7), range(6, -1, -1)])
        expected = [(0, 2, 6), (6, 5, 3)]
        expected_ids = [(0, 1, 3), (0, 0, 0)]
        maxed_by_bins, max_id_by_bins = max_tensor_by_bins(logits, bins)
        assert torch.eq(maxed_by_bins, torch.tensor(expected)).all()
        assert torch.eq(max_id_by_bins, torch.tensor(expected_ids)).all()

    def test_word_classes_to_mask(self):
        word_class_tensor = torch.tensor([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        # 3 label groups
        mask_groups = [[0, 1], [1, 2], [0, 2], []]
        wc2m = word_classes_to_mask(word_class_tensor, mask_groups=mask_groups, n_labels_grps=3)
        expected = torch.tensor([[1, 1, 0], [0, 1, 1], [1, 0, 1], [0, 0, 0]])
        assert torch.eq(wc2m, expected).all()
