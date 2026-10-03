"""Isolated C128 attention-to-projection fusion for the 720p fast backend.

The installed MLP and QKV producer remain unchanged. Each program processes
one window's query rows, keeping one head's attended half values on chip while
accumulating the four output-projection heads in their original order.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from fused_swin_core_native_half_v1 import _exp, _weights_pair
from nr_backend.triton_fp8 import _round_fp8_half
from spill_preflight_v1 import select


@triton.jit
def _fused(Q, K, V, BIAS, WEIGHT, RESIDUAL, SCALE, ORDER, OUT,
           HP: tl.constexpr, WP: tl.constexpr, H: tl.constexpr,
           WIDTH: tl.constexpr, SY: tl.constexpr, SX: tl.constexpr,
           BM: tl.constexpr, ROUND_WEIGHTS: tl.constexpr):
    window = tl.program_id(1)
    local = tl.program_id(0) * BM + tl.arange(0, BM)
    lane = tl.arange(0, 32)
    cols = tl.arange(0, 128)
    windows_per_head: tl.constexpr = (HP // 8) * (WP // 8)
    total = tl.full((BM, 128), 0, tl.float32)

    for head in range(4):
        offset = (head * windows_per_head + window) * 2048
        q = tl.load(Q + offset + local[:, None] * 32 + lane[None, :])
        k0 = tl.load(K + offset + lane[None, :] * 32 + lane[:, None])
        k1 = tl.load(K + offset + (lane[None, :] + 32) * 32 + lane[:, None])
        score0 = tl.dot(q, k0, out_dtype=tl.float32)
        score1 = tl.dot(q, k1, out_dtype=tl.float32)
        b0 = tl.load(BIAS + head * 4096 + local[:, None] * 64 + lane[None, :])
        b1 = tl.load(BIAS + head * 4096 + local[:, None] * 64 + lane[None, :] + 32)
        e0 = _exp((score0 + b0.to(tl.float32)).to(tl.float16))
        e1 = _exp((score1 + b1.to(tl.float32)).to(tl.float16))
        w0, w1 = _weights_pair(e0, e1, BM, ROUND_WEIGHTS=ROUND_WEIGHTS)
        v0 = tl.load(V + offset + lane[:, None] * 32 + lane[None, :])
        v1 = tl.load(V + offset + (lane[:, None] + 32) * 32 + lane[None, :])
        attended = tl.dot(w0, v0, out_dtype=tl.float32)
        attended = tl.dot(w1, v1, attended, out_dtype=tl.float32).to(tl.float16)
        if ROUND_WEIGHTS:
            attended = _round_fp8_half(attended)
        projection = tl.load(
            WEIGHT + (head * 32 + lane[:, None]) * 128 + cols[None, :])
        total = tl.dot(attended, projection, total, out_dtype=tl.float32)

    physical = tl.load(ORDER + local).to(tl.int32)
    y = (window // (WP // 8)) * 8 + physical // 8
    x = (window % (WP // 8)) * 8 + physical % 8
    pixel = y * WP + x
    residual = tl.load(RESIDUAL + pixel[:, None] * 128 + cols[None, :])
    scale = tl.load(SCALE + cols)
    initial = (residual.to(tl.float32) * scale[None, :].to(tl.float32)).to(tl.float16)
    result = (total + initial.to(tl.float32)).to(tl.float16)
    oy = y - SY
    ox = x - SX
    valid = ((oy[:, None] >= 0) & (oy[:, None] < H) &
             (ox[:, None] >= 0) & (ox[:, None] < WIDTH))
    tl.store(OUT + (oy[:, None] * WIDTH + ox[:, None]) * 128 + cols[None, :],
             result, valid)


def forward(query, key, value, bias, weight, residual, scale, order, *,
            height: int, width: int, shift: tuple[int, int],
            round_weights: bool = False):
    if (query.ndim != 5 or query.shape != key.shape or query.shape != value.shape or
            query.shape[0] != 4 or query.shape[-2:] != (64, 32)):
        raise ValueError("Expected four-head C128 Q/K/V windows")
    hp, wp = query.shape[1] * 8, query.shape[2] * 8
    sy, sx = shift
    if (bias.shape != (4, 64, 64) or weight.shape != (128, 128) or
            residual.shape != (hp, wp, 128) or scale.shape != (128,) or
            order.shape != (64,) or sy not in (0, 4) or sx not in (0, 4) or
            height + sy > hp or width + sx > wp):
        raise ValueError("Unexpected C128 output boundary")
    for tensor in (query, key, value, bias, weight, residual, scale):
        if (tensor.device.type != "xpu" or tensor.device != query.device or
                tensor.dtype != torch.float16 or not tensor.is_contiguous()):
            raise ValueError("Expected contiguous FP16 XPU inputs")
    if order.device != query.device or order.dtype != torch.int64 or not order.is_contiguous():
        raise ValueError("Expected contiguous XPU pixel order")
    output = torch.empty((height, width, 128), dtype=torch.float16, device=query.device)
    args_for = lambda config: (query, key, value, bias, weight, residual, scale,
                               order, output, hp, wp, height, width, sy, sx,
                               config[0], round_weights)
    grid_for = lambda config: (64 // config[0], (hp // 8) * (wp // 8))
    chosen, compiled, selection = select(
        _fused, [(16,)], args_for, grid_for,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    kernel = _fused[grid_for(chosen)](*args_for(chosen), num_warps=4,
                                      num_stages=1, enable_fp_fusion=False)
    if kernel is not compiled:
        raise RuntimeError("Fused C128 kernel changed after preflight")
    return output, selection
