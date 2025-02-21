# -*- coding: utf-8 -*-

import os

import pytest
import torch
import triton

from fla.ops.nsa_pe.naive import naive_nsa_pe
from fla.ops.nsa_pe.parallel import parallel_nsa_pe
from utils import assert_close


@pytest.mark.parametrize("B", [1])
@pytest.mark.parametrize("T", [256, 1000])
@pytest.mark.parametrize("H", [1])
@pytest.mark.parametrize("HQ", [16])
@pytest.mark.parametrize("D", [128, 64])
@pytest.mark.parametrize("S", [16])
@pytest.mark.parametrize("block_size", [64])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("scale", [0.1])
def test_parallel(
    B: int,
    H: int,
    HQ: int,
    T: int,
    D: int,
    S: int,
    block_size: int,
    dtype: torch.dtype,
    scale: float
):
    torch.manual_seed(42)
    os.environ['TRITON_F32_DEFAULT'] = 'ieee'

    q = torch.randn((B, T, HQ, D), dtype=dtype, device='cuda').requires_grad_(True)
    k = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)
    v = torch.randn((B, T, H, D), dtype=dtype, device='cuda').requires_grad_(True)

    S = min(triton.cdiv(T, block_size), S)

    indices = torch.stack([torch.randperm(S) for _ in range(B * T * H)]).sort(-1)[0]
    indices = indices.view(B, T, H, S).long().cuda()

    ref = naive_nsa_pe(q=q, k=k, v=v, indices=indices, block_size=block_size, scale=scale)

    tri = parallel_nsa_pe(q=q, k=k, v=v, indices=indices, block_size=block_size, scale=scale)
    assert_close(" o", ref, tri, 0.005)
