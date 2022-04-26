from typing import Any, Optional

import torch
from fairseq.data import BaseWrapperDataset, Dictionary
from torch.utils.data import Dataset


class TextEncodingDataset(BaseWrapperDataset):
    def __init__(
        self,
        dataset: Dataset,
        dictionary: Dictionary,
        bpe: Any,
        prepend_token: Optional[int] = None,
        add_prefix_space=True,
    ):
        super().__init__(dataset)
        self.dictionary = dictionary
        self.bpe = bpe
        self._sizes = None
        self.prepend_tensor = None
        if prepend_token is not None:
            self.prepend_tensor = torch.tensor([prepend_token])
        self.add_prefix_space = add_prefix_space

    def __getitem__(self, index: int):
        text = self.dataset[index]
        if self.add_prefix_space and not text[0] == " ":
            text = " " + text
        hf_ids_string = self.bpe.encode(text)
        output_ids = self.dictionary.encode_line(hf_ids_string)
        if self.prepend_tensor is not None:
            output_ids = torch.cat([self.prepend_tensor, output_ids])
        return output_ids

    def get_sizes(self):
        sizes = torch.zeros(len(self.dataset))
        for idx in range(len(self.dataset)):
            sizes[idx] = len(self[idx])
        return sizes

    @property
    def sizes(self):
        if self._sizes is not None:
            return self._sizes
        sizes = torch.zeros(len(self.dataset), dtype=torch.long)
        for idx in range(len(self.dataset)):
            sizes[idx] = len(self[idx])
        self._sizes = sizes
        return self._sizes
