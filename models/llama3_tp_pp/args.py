# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

from simsuite.device import Device
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class QuantizationScheme(Enum):
    int4_weight_int8_dynamic_activation = "int4_weight_int8_dynamic_activation"


@dataclass
class QuantizationArgs:
    scheme: Optional[QuantizationScheme] = None
    group_size: Optional[int] = None
    spinquant: bool = False

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k == "scheme":
                setattr(self, k, QuantizationScheme(v))
            else:
                if hasattr(self, k):
                    setattr(self, k, v)


@dataclass
class LoRAArgs:
    rank: int
    scale: float


@dataclass
class ModelArgs:
    me: Device
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: Optional[int] = None
    vocab_size: int = -1
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    rope_theta: float = 500000
    use_scaled_rope: bool = False

    max_batch_size: int = 32
    max_seq_len: int = 2048

    use_kv_cache: bool = True
    yield_probability: float = 1.0

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
            elif k == "me":
                setattr(self, k, v)

        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads
        assert self.n_kv_heads <= self.n_heads
        assert self.n_heads % self.n_kv_heads == 0
        assert self.dim % self.n_heads == 0
