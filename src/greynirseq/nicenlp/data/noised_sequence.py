# Copyright (C) Miðeind ehf.
# This file is part of GreynirSeq <https://github.com/mideind/GreynirSeq>.
# See the LICENSE file in the root of the project for terms of use.

from dataclasses import dataclass

from torch import Tensor


@dataclass
class NoisedSequence:
    sequence: Tensor
    noise_allowed_mask: Tensor
