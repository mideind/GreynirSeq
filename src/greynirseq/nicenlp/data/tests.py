import unittest

import torch
from fairseq.data import ListDataset

from greynirseq.nicenlp.data.lookup_dataset import LookupDataset
from greynirseq.nicenlp.data.mutex_binary_dataset import MutexBinaryDataset
from greynirseq.nicenlp.data.masked_byte_sequence import MaskedByteSequenceDataset


class TestLookupDataset(unittest.TestCase):
    def test_dataset(self):
        data = [range(10), range(10)]
        lookup_dict = {i: i % 2 for i in range(10)}
        dataset = LookupDataset(data, lookup_dict, label_first=True)
        assert dataset[0].tolist() == [1, 1, 0, 1, 0, 1, 0, 1, 0]
        dataset = LookupDataset(data, lookup_dict, label_first=False)
        assert dataset[0].tolist() == [0, 1, 0, 1, 0, 1, 0, 1, 0]


class TestMutexBinaryDataset(unittest.TestCase):
    def test_dataset(self):
        num_mutex_classes = 34

        data = []
        for i in range(10):
            data.append(num_mutex_classes - i - 1)
            data.append(num_mutex_classes + 2 * i)
            data.append(num_mutex_classes - i)

        dataset = MutexBinaryDataset(
            [data], num_mutex_classes=num_mutex_classes, separator=-1
        )

        expected = [
            33.0,
            -1.0,
            34.0,
            -1.0,
            34.0,
            -1.0,
            32.0,
            -1.0,
            36.0,
            -1.0,
            33.0,
            -1.0,
            31.0,
            -1.0,
            38.0,
            -1.0,
            32.0,
            -1.0,
            30.0,
            40.0,
            -1.0,
            31.0,
            -1.0,
            29.0,
            42.0,
            -1.0,
            30.0,
            -1.0,
            28.0,
            44.0,
            -1.0,
            29.0,
            -1.0,
            27.0,
            46.0,
            -1.0,
            28.0,
            -1.0,
            26.0,
            48.0,
            -1.0,
            27.0,
            -1.0,
            25.0,
            50.0,
            -1.0,
            26.0,
            -1.0,
            24.0,
            52.0,
            -1.0,
            25.0,
            -1.0,
        ]
        assert dataset[0].tolist() == expected

# class MaskedByteSequenceDataset(unittest.TestCase):
#     def test_dataset(self):
#         listdataset = ListDataset([
#             "I am a byte sequence"
#         ])
#         byte_seq_masker = MaskedByteSequenceDataset(
#             listdataset,
#         )


if __name__ == "__main__":
    unittest.main()
