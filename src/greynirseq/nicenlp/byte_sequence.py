from typing import Optional

from dataclasses import dataclass, asdict
import torch


@dataclass
class ByteSequence:
    str_seq: str
    byte_seq: torch.Tensor
    bpe_ids: Optional[torch.Tensor] = None
    bpe_lens: Optional[torch.Tensor] = None
    bpe_mask: Optional[torch.Tensor] = None
    word_mask: Optional[torch.Tensor] = None
    word_ids: Optional[torch.Tensor] = None
    word_lens: Optional[torch.Tensor] = None
    targets: Optional[torch.Tensor] = None
    # denotes which positions after contraction are targets
    target_mask: Optional[torch.Tensor] = None

    def clone(self):
        """Naive shallow clone"""
        new_dict = {}
        for key, value in asdict(self).items():
            if torch.is_tensor(value):
                new_dict[key] = value.clone()
            elif isinstance(value, list):
                new_dict[key] = list(value)
            else:
                new_dict[key] = value

        return ByteSequence(**(new_dict))

    def add_prefix(self, byte_prefix, string_prefix=None):
        if self.targets is not None or self.target_mask is None:
            # XXX: is this necessary?
            raise ValueError("Cannot add prefix when targets are null")
        new_string = self.str_seq
        if string_prefix is not None and string_prefix:
            new_string = string_prefix + self.str_seq
        new_byte_seq = torch.cat([byte_prefix, self.byte_seq])
        new_seq = ByteSequence(new_string, new_byte_seq)
        if string_prefix is not None and string_prefix:
            new_seq.bpe_lens = self.word_lens.clone()

        if self.bpe_ids is not None:
            new_seq.bpe_ids = self.bpe_ids.clone()
        if self.bpe_lens is not None:
            new_seq.bpe_lens = torch.cat([torch.tensor([len(byte_prefix)]), self.bpe_lens])
        if self.bpe_mask is not None:
            new_seq.bpe_mask = torch.cat([torch.zeros(len(byte_prefix), dtype=self.bpe_mask.type()), self.bpe_mask])
            new_seq.bpe_mask[0] = 1
        if self.word_mask is not None:
             = self.word_mask.clone()
            new_seq.word_mask = torch.cat([torch.zeros(len(byte_prefix), dtype=self.word_mask.type()), self.word_mask])
            new_seq.word_mask[0] = 1
        if self.word_lens is not None:
            new_seq.word_lens = torch.cat([torch.tensor([len(byte_prefix)]), self.word_lens])
        return new_seq
