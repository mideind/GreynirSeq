import mmap

import torch
from fairseq.data import BaseWrapperDataset


class TokenOffsetsDataset(torch.utils.data.Dataset):
    """ Dataset for ByteBert
        Takes in a token span file.
        Returns token offsets.
    """

    def __init__(self, span_path):
        super().__init__()
        self.span_path = span_path
        self.span_fd = None
        self.span_file_buffer = None

        self.open_files()
        self.calculate_indexes()

    def open_files(self):
        self.span_fd = open(self.span_path, "rb")
        self.span_file_buffer = mmap.mmap(self.span_fd.fileno(), 0, prot=mmap.PROT_READ)

    def calculate_indexes(self):
        self.span_indexes = self.calculate_line_indexes(self.span_file_buffer)

    @staticmethod
    def calculate_line_indexes(buf):
        buf.seek(0)
        indexes = [0]
        while True:
            line = buf.readline()
            if not line:
                break
            offset = buf.tell()
            indexes.append(offset)
        buf.seek(0)
        return indexes

    def __getitem__(self, i):
        assert i >= 0 and i < len(self.span_indexes) - 1

        string_nums = self.span_file_buffer[
            self.span_indexes[i] : self.span_indexes[i + 1]
        ].decode()
        return torch.tensor([int(num) for num in string_nums.split()], dtype=torch.long)

    def __len__(self):
        return len(self.span_indexes) - 1

    def __del__(self):
        if hasattr(self, "span_file_buffer") and self.span_file_buffer is not None:
            self.span_file_buffer.close()
            self.span_file_buffer = None
        if hasattr(self, "span_fd") and self.span_fd is not None:
            self.span_fd.close()
            self.span_fd = None


if __name__ == "__main__":
    import sys

    dataset = TokenOffsetsDataset(sys.argv[1])

    d = dataset[9]
    print(d)
    print(len(dataset))
