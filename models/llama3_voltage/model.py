# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
from torch import Tensor
import torch.nn.functional as F
from fairscale.nn.model_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    VocabParallelEmbedding,
)
from torch import nn

from .args import ModelArgs

# **NOTE**: This code is not runnable without installing `torch` and `fairscale`
# dependencies. These dependencies are not part of the default dependencies
# (requirements.txt) of the `llama-models` package.


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def unstride_torch(shards: list[Tensor], dim: int = 0) -> Tensor:
    """Inverse of x[... , r::total, ...] along `dim` (r = 0..total-1)."""
    if not shards:
        raise ValueError("shards is empty")
    total = len(shards)
    dim = dim % shards[0].ndim

    # All non-sharded dims must match
    ref = shards[0].shape
    for s in shards:
        if s.ndim != len(ref):
            raise ValueError("rank mismatch across shards")
        if any(i != dim and s.shape[i] != ref[i] for i in range(s.ndim)):
            raise ValueError("non-sharded dimensions differ across shards")

    # Output shape
    N = sum(s.shape[dim] for s in shards)
    out_shape = list(ref)
    out_shape[dim] = N
    out = shards[0].new_empty(out_shape)

    # Optional sanity check: each shard must fit the r, r+total, ... slots
    for r, s in enumerate(shards):
        slots = (N - r + total - 1) // total  # ceil((N - r)/total)
        if s.shape[dim] != slots:
            raise ValueError(
                f"shard {r} has length {s.shape[dim]} on dim={dim}, expected {slots} for total={total}, N={N}"
            )
        idx = [slice(None)] * out.ndim
        idx[dim] = slice(r, None, total)
        out[tuple(idx)] = s
    return out


def apply_scaling(freqs: torch.Tensor) -> torch.Tensor:
    # Values obtained from grid search
    scale_factor = 8
    low_freq_factor = 1
    high_freq_factor = 4
    old_context_len = 8192  # original llama3 length

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * torch.pi / freqs
    new_freqs = torch.where(wavelen > low_freq_wavelen, freqs / scale_factor, freqs)
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    return torch.where(
        (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen),
        (1 - smooth) * new_freqs / scale_factor + smooth * new_freqs,
        new_freqs,
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0, use_scaled: bool = False):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if use_scaled:
        freqs = apply_scaling(freqs)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cis_xq: torch.Tensor,
    freqs_cis_xk: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis_xq = reshape_for_broadcast(freqs_cis_xq, xq_)
    freqs_cis_xk = reshape_for_broadcast(freqs_cis_xk, xk_)
    xq_out = torch.view_as_real(xq_ * freqs_cis_xq).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis_xk).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


class Attention(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        world_size = fs_init.get_model_parallel_world_size()
        self.n_local_heads = args.n_heads // world_size
        self.n_local_kv_heads = self.n_kv_heads // world_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.me = args.me
        self.layer_id = layer_id

        self.wq = ColumnParallelLinear(
            args.dim,
            args.n_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinear(
            args.dim,
            self.n_kv_heads * self.head_dim,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinear(
            args.n_heads * self.head_dim,
            args.dim,
            bias=False,
            input_is_parallel=True,
            init_method=lambda x: x,
        )

        self.cache_k = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )
        self.cache_v = torch.zeros(
            (
                args.max_batch_size,
                args.max_seq_len,
                self.n_local_kv_heads,
                self.head_dim,
            )
        )

        self.cache_known = torch.zeros((args.max_seq_len,))

        self.use_kv_cache = args.use_kv_cache

    def insert_cache_value(self, start_pos: int, xk: torch.Tensor, xv: torch.Tensor):
        seqlen = xk.shape[1]
        self.cache_k[:1, start_pos : start_pos + seqlen] = xk
        self.cache_v[:1, start_pos : start_pos + seqlen] = xv
        self.cache_known[start_pos : start_pos + seqlen] = 1

    def clean_cache(self):
        self.cache_known.zero_()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        me = self.me
        world = me.world
        forward_chan = world.chan("forward")
        rank = forward_chan.rank(me)
        total = forward_chan.size()

        # Partition the input matrix x based on rank and total
        xp = torch.tensor_split(x, total, dim=1)[rank]

        bsz, seqlen, _ = x.shape
        bsz_p, seqlen_p, _ = xp.shape
        xq, xk, xv = self.wq(xp), self.wk(x), self.wv(x)

        xq = xq.view(bsz_p, seqlen_p, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        freqs_cis_xq = torch.tensor_split(freqs_cis, total, dim=0)[rank]
        freqs_cis_xk = freqs_cis
        xq, xk = apply_rotary_emb(xq, xk, freqs_cis_xq=freqs_cis_xq, freqs_cis_xk=freqs_cis_xk)

        if self.use_kv_cache:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            self.insert_cache_value(start_pos, xk, xv)

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys = xk
            values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen_p, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask  # (bs, n_local_heads, seqlen_p, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen_p, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz_p, seqlen_p, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)
        self.w2 = RowParallelLinear(hidden_dim, dim, bias=False, input_is_parallel=True, init_method=lambda x: x)
        self.w3 = ColumnParallelLinear(dim, hidden_dim, bias=False, gather_output=False, init_method=lambda x: x)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(layer_id, args)
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.me = args.me

    def sync_kv_cache(self, start_pos: int, xk: Tensor, xv):
        self.attention.insert_cache_value(start_pos, xk, xv)

    def clean_cache(self):
        self.attention.clean_cache()

    def forward(
        self,
        x: torch.Tensor,
        start_pos: int,
        freqs_cis: torch.Tensor,
        mask: Optional[torch.Tensor],
    ):
        me = self.me
        world = me.world
        forward_chan = world.chan("forward")
        rank = forward_chan.rank(me)
        total = forward_chan.size()
        xp = torch.tensor_split(x, total, dim=1)[rank]

        h = xp + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = VocabParallelEmbedding(params.vocab_size, params.dim, init_method=lambda x: x)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(params.dim, params.vocab_size, bias=False, init_method=lambda x: x)
        self.me = params.me

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

        self.use_kv_cache = params.use_kv_cache
        self.me = params.me

    def sync_kv_cache(self, layer_id: int, start_pos: int, xk: Tensor, xv: Tensor):
        layer = self.layers[layer_id]

        if isinstance(layer, TransformerBlock):
            layer.sync_kv_cache(start_pos, xk, xv)
        else:
            raise TypeError(f"Layer {layer_id} is not a TransformerBlock, but {type(layer)}")

    def clean_cache(self):
        for layer in self.layers:
            if isinstance(layer, TransformerBlock):
                layer.clean_cache()

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        _bsz, seqlen = tokens.shape
        h = self.tok_embeddings(tokens)
        self.freqs_cis = self.freqs_cis.to(h.device)

        if self.use_kv_cache:
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        else:
            freqs_cis = self.freqs_cis[:seqlen]

        me = self.me
        world = me.world
        forward_chan = world.chan("forward")
        rank = forward_chan.rank(me)
        total = forward_chan.size()

        mask = None
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device)
            mask = torch.triu(mask, diagonal=1)

            # https://github.com/pytorch/pytorch/issues/100005
            # torch.triu is buggy when the device is mps: filled values are
            # nan instead of 0.
            if mask.device.type == torch.device("mps").type:
                mask = torch.nan_to_num(mask, nan=0.0)

            # When performing key-value caching, we compute the attention scores
            # only for the new sequence. Thus, the matrix of scores is of size
            # (seqlen, cache_len + seqlen), and the only masked entries are (i, j) for
            # j > cache_len + i, since row i corresponds to token cache_len + i.
            if self.use_kv_cache:
                mask = torch.hstack([torch.zeros((seqlen, start_pos), device=tokens.device), mask]).type_as(h)
            else:
                mask = mask.to(tokens.device)

            splits = torch.tensor_split(torch.arange(seqlen), total)
            partition_start = splits[rank][0].item()
            partition_end = splits[rank][-1].item() + 1  # +1 because slicing is exclusive
            mask = mask[partition_start:partition_end, :]

        for i, layer in enumerate(self.layers):
            h = layer(h, start_pos, freqs_cis, mask)
            h = world.chan("forward").all_gather(me, h, f"forward_{i}")
            h = torch.cat(h, dim=1)

        h = self.norm(h)
        output = self.output(h).float()
        return output
