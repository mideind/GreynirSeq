from fairseq.data import BaseWrapperDataset
from torch.utils.data import Dataset


class LambdaDataset(BaseWrapperDataset):
    def __init__(self, dataset: Dataset, lambda_fn):
        super().__init__(dataset)
        self.lambda_fn = lambda_fn

    def __getitem__(self, index: int):
        return self.lambda_fn(self.dataset[index])
