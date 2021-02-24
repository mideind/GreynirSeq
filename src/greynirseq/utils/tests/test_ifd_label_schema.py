import unittest

from greynirseq.utils.ifd_label_schema import ifd_tag_to_schema


class IFD_Mapping_Tests(unittest.TestCase):
    faven = ("faven", ["fa", "fem", "sing", "nom"])
    nven = ("nven", ["n", "fem", "sing", "nom"])
    nkeeg = ("nkeeg", ["n", "masc", "sing", "gen", "def"])
    nkeþ_s = ("nkeþ-s", ["n", "masc", "sing", "dat", "proper"])
    lkeove = ("lkeove", ["l", "masc", "sing", "acc", "weak", "superl"])
    label_results = (faven, nven, nkeeg, nkeþ_s, lkeove)

    def test_ifd_tag_to_schema(self):
        for label_result in self.label_results:
            label, correct_result = label_result
            self.assertEqual(ifd_tag_to_schema(label), correct_result)
