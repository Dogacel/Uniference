# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the terms described in the LICENSE file in
# top-level folder for each specific model found within the models/ directory at
# the top-level of this source tree.

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed in accordance with the terms of the Llama 3 Community License Agreement.

import math
from time import perf_counter
from typing import Optional, Tuple

import fairscale.nn.model_parallel.initialize as fs_init
import torch
from torch import Tensor
import torch.nn.functional as F
from torch import nn

from models.llama3_tp_pp.components import ColumnParallelLinearSim, RowParallelLinearSim, VocabParallelEmbeddingSim
from simsuite.device import Device

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
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
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

        if args.me.world.backend == "pytorch":
            world_size = fs_init.get_model_parallel_world_size()
        else:
            world_size = get_tp_size(args.me)

        self.n_local_heads = args.n_heads // world_size
        self.n_local_kv_heads = self.n_kv_heads // world_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.me = args.me
        self.layer_id = layer_id

        self.wq = ColumnParallelLinearSim(
            args.dim,
            args.n_heads * self.head_dim,
            me=self.me,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wk = ColumnParallelLinearSim(
            args.dim,
            self.n_kv_heads * self.head_dim,
            me=self.me,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wv = ColumnParallelLinearSim(
            args.dim,
            self.n_kv_heads * self.head_dim,
            me=self.me,
            bias=False,
            gather_output=False,
            init_method=lambda x: x,
        )
        self.wo = RowParallelLinearSim(
            args.n_heads * self.head_dim,
            args.dim,
            me=self.me,
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
        batch_size = xk.shape[0]
        self.cache_k[:batch_size, start_pos : start_pos + seqlen] = xk
        self.cache_v[:batch_size, start_pos : start_pos + seqlen] = xv
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
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, freqs_cis=freqs_cis)

        if self.use_kv_cache:
            self.cache_k = self.cache_k.to(xq)
            self.cache_v = self.cache_v.to(xq)

            if self.cache_known[start_pos : start_pos + seqlen].sum() != seqlen:
                self.insert_cache_value(start_pos, xk, xv)

            keys = self.cache_k[:bsz, : start_pos + seqlen]
            values = self.cache_v[:bsz, : start_pos + seqlen]
        else:
            keys = xk
            values = xv

        # repeat k/v heads if n_kv_heads < n_heads
        keys = repeat_kv(keys, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)
        values = repeat_kv(values, self.n_rep)  # (bs, cache_len + seqlen, n_local_heads, head_dim)

        xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
        keys = keys.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        values = values.transpose(1, 2)  # (bs, n_local_heads, cache_len + seqlen, head_dim)
        scores = torch.matmul(xq, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask # (bs, n_local_heads, seqlen, cache_len + seqlen)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, values)  # (bs, n_local_heads, seqlen, head_dim)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        return self.wo(output)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        me: Device,
    ):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ColumnParallelLinearSim(dim, hidden_dim, me=me, bias=False, gather_output=False, init_method=lambda x: x)
        self.w2 = RowParallelLinearSim(hidden_dim, dim, me=me, bias=False, input_is_parallel=True, init_method=lambda x: x)
        self.w3 = ColumnParallelLinearSim(dim, hidden_dim, me=me, bias=False, gather_output=False, init_method=lambda x: x)

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
            me=args.me,
        )
        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

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
        h = x + self.attention(self.attention_norm(x), start_pos, freqs_cis, mask)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


class Transformer(nn.Module):
    def __init__(self, params: ModelArgs):
        super().__init__()

        my_rank = get_pp_rank(params.me)
        world_size = get_pp_size(params.me)

        self.params = params
        self.vocab_size = params.vocab_size

        if my_rank == 0:
            self.tok_embeddings = VocabParallelEmbeddingSim(params.vocab_size, params.dim, init_method=lambda x: x, me=params.me)

        layers_per_rank = params.n_layers // world_size
        start_layer = my_rank * layers_per_rank
        end_layer = start_layer + layers_per_rank

        self.layers = torch.nn.ModuleList()
        for layer_id in range(start_layer, end_layer):
            self.layers.append(TransformerBlock(layer_id, params))

        print(f"Device {params.me.name} has layers {start_layer} to {end_layer - 1}")

        if my_rank == world_size - 1:
            self.norm = RMSNorm(params.dim, eps=params.norm_eps)
            self.output = ColumnParallelLinearSim(params.dim, params.vocab_size, me=params.me, bias=False, init_method=lambda x: x)

        self.freqs_cis = precompute_freqs_cis(
            params.dim // params.n_heads,
            params.max_seq_len * 2,
            params.rope_theta,
            params.use_scaled_rope,
        )

        self.use_kv_cache = params.use_kv_cache

    def load_state_dict(self, state_dict, strict=True, assign=False):
        my_rank = get_pp_rank(self.params.me)
        world_size = get_pp_size(self.params.me)
        layers_per_rank = self.params.n_layers // world_size
        start_layer = my_rank * layers_per_rank
        end_layer = start_layer + layers_per_rank

        filtered = {}
        for key, value in state_dict.items():
            if key.startswith("layers."):
                layer_idx = int(key.split(".")[1])
                if start_layer <= layer_idx < end_layer:
                    new_key = key.replace(f"layers.{layer_idx}.", f"layers.{layer_idx - start_layer}.")
                    filtered[new_key] = value
            elif key.startswith("tok_embeddings."):
                if my_rank == 0:
                    filtered[key] = value
            elif key.startswith("norm.") or key.startswith("output."):
                if my_rank == world_size - 1:
                    filtered[key] = value
            else:
                filtered[key] = value

        return super().load_state_dict(filtered, strict=strict, assign=assign)


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
        my_rank = get_pp_rank(self.params.me)
        world_size = get_pp_size(self.params.me)

        # print(f"Device {self.params.me.name} starting forward pass at PP rank {my_rank}/{world_size}...")

        _bsz, seqlen = tokens.shape

        if my_rank == 0:
            h = self.tok_embeddings(tokens)
        else:
            # Receive hidden states from previous rank
            # print(f"Device {self.params.me.name} waiting to receive input from previous PP rank {my_rank - 1}...")
            h = pp_recv(
                self.params.me, 
                source_rank=my_rank - 1,
                tokens=tokens,
                params=self.params,
            )
            # print(f"Device {self.params.me.name} received input from previous PP rank {my_rank - 1}.")

        self.freqs_cis = self.freqs_cis.to(h.device)

        if self.use_kv_cache:
            freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        else:
            freqs_cis = self.freqs_cis[:seqlen]

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

        for i, layer in enumerate(self.layers):
            # print(f"Device {self.params.me.name} processing layer {my_rank * (self.params.n_layers // world_size) + i}/{self.params.n_layers}...")
            h = layer(h, start_pos, freqs_cis, mask)

        # Only last rank does norm + output projection
        if my_rank == world_size - 1:
            h = self.norm(h)
            output = self.output(h).float()
            # result = pp_broadcast(self.params.me, output, source_rank=my_rank)
            # print(f"Device {self.params.me.name} broadcasting final output from PP rank {my_rank}.")
            return output
        else:
            # Send to next rank
            pp_send(self.params.me, h, target_rank=my_rank + 1)
            # print(f"Device {self.params.me.name} sent output to next PP rank {my_rank + 1}.")

        data = torch.empty(
            (h.shape[0], h.shape[1], self.params.vocab_size),
            dtype=torch.float32,
            device=h.device
        )
        # result = pp_broadcast(self.params.me, data, source_rank=world_size - 1)
        # print(f"Device {self.params.me.name} received final output from PP rank {world_size - 1}.")
        return None

from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_world_size, 
    get_model_parallel_rank,
    get_pipeline_parallel_group,
    get_pipeline_parallel_ranks,
)
import torch.distributed as dist


def pp_send(me: Device, data, target_rank: int):
    if me.world.backend == "pytorch":
        group = get_pipeline_parallel_group()
        pp_ranks = get_pipeline_parallel_ranks()

        # print(f"Device {me.name} sending data to PP rank {target_rank}...")
        # print(f"SEND: my global rank={dist.get_rank()}, pp_ranks={pp_ranks}, target_rank={target_rank}, dst_global={pp_ranks[target_rank]}")
        # print(f"Sent data shape: {data.shape}, dtype={data.dtype}, device={data.device}")

        me.world.event_logger.log_event(
            {
                "time": me.state.sync_clock(),
                "action": "transmit_start",
                "op": "pp_send",
                "size": data.element_size() * data.nelement(),
            }
        )

        start = perf_counter()
        dist.send(data.cpu(), dst=get_pipeline_parallel_ranks()[target_rank], group=group)
        end = perf_counter()

        me.world.event_logger.log_event(
            {
                "time": me.state.sync_clock(),
                "action": "transmit_end",
                "op": "pp_send",
                "duration": end - start,
            }
        )
        return

    rank = get_pp_rank(me)
    tp_rank = get_tp_rank(me)

    target_device = me.world.chan(f"pp_{target_rank}").subscribers[tp_rank]

    me.pp_chan().send(me, data, f"pp_send_{rank}_{target_rank}_{tp_rank}", target_device)

def pp_recv(me: Device, source_rank: int, tokens: torch.Tensor, params):
    if me.world.backend == "pytorch":
        group = get_pipeline_parallel_group()
        pp_ranks = get_pipeline_parallel_ranks()

        shape = (tokens.shape[0], tokens.shape[1], params.dim)
        data = torch.empty(shape, dtype=torch.bfloat16, device="cpu")

        # print(f"Device {me.name} receiving data from PP rank {source_rank}...")
        # print(f"RECV: my global rank={dist.get_rank()}, pp_ranks={pp_ranks}, source_rank={source_rank}, src_global={pp_ranks[source_rank]}")
        # print(f"RECV: group={group}, group.size()={group.size()}")
        # print(f"Receiving data shape: {data.shape}, dtype={data.dtype}, device={data.device}")

        me.world.event_logger.log_event(
            {
                "time": me.state.sync_clock(),
                "action": "transmit_start",
                "op": "pp_recv",
                "size": data.element_size() * data.nelement(),
            }
        )

        start = perf_counter()
        dist.recv(data, src=get_pipeline_parallel_ranks()[source_rank], group=group)
        end = perf_counter()

        me.world.event_logger.log_event(
            {
                "time": me.state.sync_clock(),
                "action": "transmit_end",
                "op": "pp_recv",
                "duration": end - start,
            }
        )

        data = data.to(tokens.device)
        return data

    rank = get_pp_rank(me)
    tp_rank = get_tp_rank(me)

    data = me.pp_chan().receive(me, f"pp_send_{source_rank}_{rank}_{tp_rank}")
    return data

def pp_broadcast(me: Device, data, source_rank: int):
    if me.world.backend == "pytorch":
        group = get_pipeline_parallel_group()
        rank = get_pp_rank(me)

        # print(f"Device {me.name} broadcasting data from PP rank {source_rank}...")
        # print(f"Shape: {data.shape}, dtype={data.dtype}, device={data.device}")

        me.world.event_logger.log_event(
            {
                "time": me.state.sync_clock(),
                "action": "transmit_start",
                "op": "pp_broadcast",
                "size": data.element_size() * data.nelement(),
            }
        )

        start = perf_counter()
        dist.broadcast(data, src=get_pipeline_parallel_ranks()[source_rank], group=group)
        end = perf_counter()

        me.world.event_logger.log_event(
            {
                "time": me.state.sync_clock(),
                "action": "transmit_end",
                "op": "pp_broadcast",
                "duration": end - start,
            }
        )
        return data

    rank = get_pp_rank(me)
    if rank == source_rank:
        tp_rank = get_tp_rank(me)
        for i in range(me.pp_size):
            if i == source_rank:
                continue
            target_device = me.pp_chan().subscribers[tp_rank]
            # print(f"Device {me.name} broadcasting to PP rank {i}...")
            me.world.chan(f"pp_{i}").send(me, data, f"pp_broadcast_{source_rank}_{i}", target_device)
            # print(f"Device {me.name} broadcasted to PP rank {i}.")

    if rank != source_rank:
        # print(f"Device {me.name} waiting to receive broadcast from PP rank {source_rank}...")
        data = me.pp_chan().receive(me, f"pp_broadcast_{source_rank}_{rank}")
        # print(f"Device {me.name} received broadcast from PP rank {source_rank}.")
    return data

def get_pp_size(me: Device) -> int:
    if me.world.backend == "pytorch":
        # print(f"Device {me.name} size: {len(get_pipeline_parallel_ranks())}")
        return len(get_pipeline_parallel_ranks())

    return me.pp_size

def get_pp_rank(me: Device) -> int:
    if me.world.backend == "pytorch":
        # print(f"Device {me.name} rank: {get_pipeline_parallel_group().rank()}")
        return get_pipeline_parallel_group().rank()

    return me.pp_rank

def get_tp_size(me: Device) -> int:
    if me.world.backend == "pytorch":
        # print(f"Device {me.name} TP size: {fs_init.get_model_parallel_world_size()}")
        return get_model_parallel_world_size()

    return len(me.tp_chan().subscribers)

def get_tp_rank(me: Device) -> int:
    if me.world.backend == "pytorch":
        # print(f"Device {me.name} TP rank: {fs_init.get_model_parallel_rank()}")
        return get_model_parallel_rank()

    return me.tp_chan().subscribers.index(me)
