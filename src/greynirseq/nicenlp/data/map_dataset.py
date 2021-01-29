from typing import List, Tuple, Optional, Callable, Any

from torch.utils.data.dataset import Dataset
from fairseq.data import BaseWrapperDataset


class MapDataset(BaseWrapperDataset):
    def __init__(self, dataset: Dataset, fn: Callable[[int], Any]):
        super().__init__(dataset)
        self.fn = fn
        self._sizes = dataset.sizes

    def __getitem__(self, index: int):
        item = self.dataset[index]
        item = self.fn(item)
        return item
