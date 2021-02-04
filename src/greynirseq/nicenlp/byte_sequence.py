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
        """Deep clone"""
        new_dict = {}
        for key, value in asdict(self).items():
            if torch.is_tensor(value):
                new_dict[key] = value.clone()
            elif isinstance(value, str) or value is None:
                new_dict[key] = value
            else:
                raise ValueError(f"Cannot deep clone contained type {value}")

        return ByteSequence(**(new_dict))

    def add_byte_prefix(self, byte_prefix):
        if self.targets is not None or self.target_mask is not None:
            raise NotImplementedError("Cannot add prefix when sequence has targets")
        new_byte_seq = torch.cat([byte_prefix, self.byte_seq])
        new_seq = ByteSequence(self.str_seq, new_byte_seq)

        if self.bpe_ids is not None:
            new_seq.bpe_ids = self.bpe_ids.clone()
        if self.bpe_lens is not None:
            new_seq.bpe_lens = torch.cat(
                [torch.tensor([len(byte_prefix)]), self.bpe_lens]
            )
        if self.bpe_mask is not None:
            new_seq.bpe_mask = torch.cat(
                [
                    torch.zeros(len(byte_prefix), dtype=self.bpe_mask.dtype),
                    self.bpe_mask,
                ]
            )
            new_seq.bpe_mask[0] = 1
        if self.word_mask is not None:
            new_seq.word_mask = torch.cat(
                [
                    torch.zeros(len(byte_prefix), dtype=self.word_mask.dtype),
                    self.word_mask,
                ]
            )
            new_seq.word_mask[0] = 1
        if self.word_lens is not None:
            new_seq.word_lens = torch.cat(
                [torch.tensor([len(byte_prefix)]), self.word_lens]
            )
        return new_seq
