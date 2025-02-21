# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from typing import Optional

import torch
import triton
import triton.language as tl
from einops import rearrange

from fla.ops.common.utils import prepare_sequence_indices
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, contiguous

class Rotary(torch.nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None

    def forward(self, x):
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        return self.cos_cached, self.sin_cached


@triton.heuristics({
    'USE_OFFSETS': lambda args: args['offsets'] is not None
})
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16]
    ],
    key=["BS", "BK", "BV"],
)
@triton.jit
def parallel_nsa_pe_fwd_kernel(
    q,
    k,
    v,
    o,
    cos,
    sin,
    scale,
    block_indices,
    offsets,
    indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    HQ: tl.constexpr,
    G: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    S: tl.constexpr,
    BS: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_OFFSETS: tl.constexpr
):
    i_v, i_t, i_bh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H

    if USE_OFFSETS:
        i_n, i_t = tl.load(indices + i_t * 2).to(tl.int32), tl.load(indices + i_t * 2 + 1).to(tl.int32)
        bos, eos = tl.load(offsets + i_n).to(tl.int32), tl.load(offsets + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    k += (bos * H + i_h) * K
    v += (bos * H + i_h) * V
    block_indices += (bos + i_t) * H*S + i_h * S

    p_q1 = tl.make_block_ptr(q + (bos + i_t) * HQ*K, (HQ, K), (K, 1), (i_h * G, 0), (G, BK // 2), (1, 0))
    p_q2 = tl.make_block_ptr(q + (bos + i_t) * HQ*K, (HQ, K), (K, 1), (i_h * G, BK // 2), (G, BK // 2), (1, 0))
    p_o = tl.make_block_ptr(o + (bos + i_t) * HQ*V, (HQ, V), (V, 1), (i_h * G, i_v * BV), (G, BV), (1, 0))

    # the Q block is kept in the shared memory throughout the whole kernel
    # [G, K // 2]
    b_q1 = tl.load(p_q1, boundary_check=(0, 1))
    b_q2 = tl.load(p_q2, boundary_check=(0, 1))

    flag_q = 0
    #b_q = (b_q * scale).to(b_q.dtype)
    # [G, BV]
    b_o = tl.zeros([G, BV], dtype=tl.float32)

    b_m = tl.full([G], float('-inf'), dtype=tl.float32)
    b_acc = tl.zeros([G], dtype=tl.float32)
    for i in range(S-1, -1, -1):
        i_s = tl.load(block_indices + i).to(tl.int32) * BS
        if i_s <= i_t:
            p_cos_k = tl.make_block_ptr(cos, (K // 2, T), (1, K // 2), (0, i * BS), (BK // 2, BS), (0, 1))
            p_sin_k = tl.make_block_ptr(sin, (K // 2, T), (1, K // 2), (0, i * BS), (BK // 2, BS), (0, 1))
            # [d, BS]
            b_cos_k = tl.load(p_cos_k, boundary_check=(0, 1))
            # [d, BS]
            b_sin_k = tl.load(p_sin_k, boundary_check=(0, 1))

            p_k1 = tl.make_block_ptr(k, (K, T), (1, H*K), (0, i_s), (BK // 2, BS), (0, 1))
            p_k2 = tl.make_block_ptr(k, (K, T), (1, H*K), (K // 2, i_s), (BK // 2, BS), (0, 1))
            p_v = tl.make_block_ptr(v, (T, V), (H*V, 1), (i_s, i_v * BV), (BS, BV), (1, 0))
            # [K, BS]
            b_k1 = tl.load(p_k1, boundary_check=(0, 1))
            b_k2 = tl.load(p_k2, boundary_check=(0, 1))

            c_k1 = b_k1
            b_k1 = b_k1 * b_cos_k + b_k2 * b_sin_k
            b_k2 = b_k2 * b_cos_k - c_k1 * b_sin_k

            if flag_q == 0:
                p_cos_q = tl.make_block_ptr(cos, (T, K // 2), (K // 2, 1), (i * BS + i_t - i_s, 0), (1, BK // 2), (1, 0))
                p_sin_q = tl.make_block_ptr(sin, (T, K // 2), (K // 2, 1), (i * BS + i_t - i_s, 0), (1, BK // 2), (1, 0))
                # [d, 1]
                b_cos_q = tl.load(p_cos_q, boundary_check=(0, 1))
                # [d, 1]
                b_sin_q = tl.load(p_sin_q, boundary_check=(0, 1))
                c_q1 = b_q1
                b_q1 = b_q1 * b_cos_q + b_q2 * b_sin_q
                b_q2 = b_q2 * b_cos_q - c_q1 * b_sin_q
                b_q1 = (b_q1 * scale).to(b_q1.dtype)
                b_q2 = (b_q2 * scale).to(b_q2.dtype)
                flag_q = 1

            # [BS, BV]
            b_v = tl.load(p_v, boundary_check=(0, 1))
            # [G, BS]
            b_s = tl.dot(b_q1, b_k1) + tl.dot(b_q2, b_k2)
            b_s = tl.where((i_t >= (i_s + tl.arange(0, BS)))[None, :], b_s, float('-inf'))

            # [G]
            b_m, b_mp = tl.maximum(b_m, tl.max(b_s, 1)), b_m
            b_r = tl.exp(b_mp - b_m)
            # [G, BS]
            b_p = tl.exp(b_s - b_m[:, None])
            # [G]
            b_acc = b_acc * b_r + tl.sum(b_p, 1)
            # [G, BV]
            b_o = b_o * b_r[:, None] + tl.dot(b_p.to(b_q1.dtype), b_v)

            b_mp = b_m
    b_o = b_o / b_acc[:, None]

    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


def parallel_nsa_pe_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    block_indices: torch.Tensor,
    block_size: int,
    scale: float,
    offsets: Optional[torch.LongTensor] = None,
    indices: Optional[torch.LongTensor] = None,
):
    B, T, H, K, V, S = *k.shape, v.shape[-1], block_indices.shape[-1]
    HQ = q.shape[2]
    G = HQ // H
    BS = block_size
    if torch.cuda.get_device_capability()[0] >= 9:
        BK = min(256, triton.next_power_of_2(K))
        BV = min(256, triton.next_power_of_2(V))
    else:
        BK = min(128, triton.next_power_of_2(K))
        BV = min(128, triton.next_power_of_2(V))
    NK = triton.cdiv(K, BK)
    NV = triton.cdiv(V, BV)
    assert NK == 1, "The key dimension can not be larger than 128/256"

    grid = (NV, T, B * H)
    o = torch.empty(B, T, HQ, V, dtype=v.dtype, device=q.device)

    parallel_nsa_pe_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        cos=cos,
        sin=sin,
        scale=scale,
        block_indices=block_indices,
        offsets=offsets,
        indices=indices,
        B=B,
        H=H,
        HQ=HQ,
        G=G,
        T=T,
        K=K,
        V=V,
        S=S,
        BS=BS,
        BK=BK,
        BV=BV,
    )
    return o


class ParallelNSAPEFunction(torch.autograd.Function):

    @staticmethod
    @contiguous
    @autocast_custom_fwd
    def forward(ctx, q, k, v, cos, sin, block_indices, block_size, scale, offsets):
        ctx.dtype = q.dtype

        # 2-d indices denoting the offsets of tokens in each sequence
        # for example, if the passed `offsets` is [0, 2, 6],
        # then there are 2 and 4 tokens in the 1st and 2nd sequences respectively, and `indices` will be
        # [[0, 0], [0, 1], [1, 0], [1, 1], [1, 2], [1, 3]]
        indices = prepare_sequence_indices(offsets) if offsets is not None else None

        o = parallel_nsa_pe_fwd(
            q=q,
            k=k,
            v=v,
            cos=cos,
            sin=sin,
            block_indices=block_indices,
            block_size=block_size,
            scale=scale,
            offsets=offsets,
            indices=indices)
        ctx.save_for_backward(q, k, v, cos, sin, block_indices, offsets, indices)
        ctx.block_size = block_size
        ctx.scale = scale
        return o.to(q.dtype)

    @staticmethod
    @contiguous
    @autocast_custom_bwd
    def backward(ctx, do):
        raise NotImplementedError


def parallel_nsa_pe(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.LongTensor,
    block_size: int,
    scale: Optional[float] = None,
    cu_seqlens: Optional[torch.LongTensor] = None,
    head_first: bool = False
) -> torch.Tensor:
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, HQ, T, K]` if `head_first=True` else `[B, T, HQ, K]`.
        k (torch.Tensor):
            keys of shape `[B, H, T, K]` if `head_first=True` else `[B, T, H, K]`.
            GQA is enforced here. The ratio of query heads (HQ) to key/value heads (H) must be a power of 2 and >=16.
        v (torch.Tensor):
            values of shape `[B, H, T, V]` if `head_first=True` else `[B, T, H, V]`.
        indices (torch.LongTensor):
            Block indices of shape `[B, T, H, S]` if `head_first=True` else `[B, T, H, S]`.
            `S` is the number of selected blocks for each query token, which is set to 16 in the paper.
        block_size (int):
            Selected block size. Default: 64.
        scale (Optional[int]):
            Scale factor for attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        head_first (Optional[bool]):
            Whether the inputs are in the head-first format. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.

    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, HQ, T, V]` if `head_first=True` else `[B, T, HQ, V]`.
    """
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if cu_seqlens is not None:
        assert q.shape[0] == 1, "batch size must be 1 when cu_seqlens are provided"
        assert not head_first, "head_first must be False when cu_seqlens are provided"
    if head_first:
        q, k, v, indices = map(lambda x: rearrange(x, 'b h t d -> b t h d'), (q, k, v, indices))
    rotary = Rotary(dim=q.shape[-1], base=10000)
    cos, sin = rotary(k)
    o = ParallelNSAPEFunction.apply(q, k, v, cos, sin, indices, block_size, scale, cu_seqlens)
    if head_first:
        o = rearrange(o, 'b t h d -> b h t d')
    return o
