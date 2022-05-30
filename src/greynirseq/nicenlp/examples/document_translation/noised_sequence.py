from dataclasses import dataclass

from torch import Tensor


@dataclass
class NoisedSequence:
    sequence: Tensor
    noise_allowed_mask: Tensor
