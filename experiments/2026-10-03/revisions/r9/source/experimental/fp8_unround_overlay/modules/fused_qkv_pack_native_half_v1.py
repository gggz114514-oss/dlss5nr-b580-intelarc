# Explicit native-half attention version; see native_half_attention_fma_v1.py.
"""Exact C32 normalization, head scale, FP8 boundaries and window gather.

Internal helper for an owned model's projected contiguous H,W,head,3,32 half
tensor. Normalization retains the established half FMA and XOR reduction order.
The source projection, head/window attention and inverse gather are unchanged.
"""
import torch
import triton
import triton.language as tl
from nr_backend.triton_attention_normalize import _nan_left, rsqrt_half_clamped
from native_half_attention_fma_v1 import half_fma_attention as _half_fma_value
from nr_backend.triton_fp8 import _round_fp8_half


@triton.jit
def _pack(Z, SCALE, ORDER, Q, K, V, COUNT: tl.constexpr, W: tl.constexpr,
          HEADS: tl.constexpr, ROWS: tl.constexpr, COLS: tl.constexpr,
          BR: tl.constexpr, ROUND_QKV: tl.constexpr=True):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    family = tl.program_id(1)
    lane = tl.arange(0, 8)
    pixel = r % 64
    window = r // 64
    col = window % COLS
    row = (window // COLS) % ROWS
    head = window // (COLS * ROWS)
    physical = tl.load(ORDER + pixel).to(tl.int32)
    y, x = row * 8 + physical // 8, col * 8 + physical % 8
    offset = (((y * W + x) * HEADS + head) * 3 + family) * 32
    off = offset[:, None] + lane[None, :]
    valid = r[:, None] < COUNT
    x0 = tl.load(Z + off, valid, other=0)
    x8 = tl.load(Z + off + 8, valid, other=0)
    x16 = tl.load(Z + off + 16, valid, other=0)
    x24 = tl.load(Z + off + 24, valid, other=0)
    if family < 2:
        a = (x16.to(tl.float32) * x16.to(tl.float32)).to(tl.float16)
        a = _nan_left(_half_fma_value(x0, x0, a), x0, a)
        b = (x24.to(tl.float32) * x24.to(tl.float32)).to(tl.float16)
        b = _nan_left(_half_fma_value(x8, x8, b), x8, b)
        total = _nan_left((a.to(tl.float32) + b.to(tl.float32)).to(tl.float16), a, b)
        for mask in tl.static_range(3):
            other = tl.gather(total, tl.broadcast_to((lane ^ (4 >> mask))[None, :], (BR, 8)), 1)
            total = _nan_left((total.to(tl.float32) + other.to(tl.float32)).to(tl.float16), total, other)
        denominator = tl.gather(total, tl.full((BR, 1), 0, tl.int32), 1)
        scale = rsqrt_half_clamped(denominator)
        x0 = _nan_left((x0.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x0, scale)
        x8 = _nan_left((x8.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x8, scale)
        x16 = _nan_left((x16.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x16, scale)
        x24 = _nan_left((x24.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x24, scale)
        if family == 0:
            scale = tl.load(SCALE + head, r < COUNT, other=0)[:, None]
            x0 = _nan_left((x0.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x0, scale)
            x8 = _nan_left((x8.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x8, scale)
            x16 = _nan_left((x16.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x16, scale)
            x24 = _nan_left((x24.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x24, scale)
    output = tl.where(family == 0, Q, tl.where(family == 1, K, V))
    dest = r[:, None] * 32 + lane[None, :]
    if ROUND_QKV:
        x0=_round_fp8_half(x0);x8=_round_fp8_half(x8)
        x16=_round_fp8_half(x16);x24=_round_fp8_half(x24)
    tl.store(output + dest, x0, valid)
    tl.store(output + dest + 8, x8, valid)
    tl.store(output + dest + 16, x16, valid)
    tl.store(output + dest + 24, x24, valid)


def forward(z, scale, order, *, rows=16, round_qkv=True):
    if z.ndim != 5 or z.shape[-2:] != (3, 32) or min(z.shape) <= 0 or z.shape[0] % 8 or z.shape[1] % 8:
        raise ValueError('Expected whole H,W,head,3,32 projected windows')
    h, w, heads = z.shape[:3]
    if z.device.type != 'xpu' or z.dtype != torch.float16 or not z.is_contiguous():
        raise ValueError('Expected contiguous half XPU projection')
    if scale.device != z.device or scale.dtype != torch.float16 or scale.shape != (heads,) or not scale.is_contiguous():
        raise ValueError('Expected owned contiguous half head scales')
    if order.device != z.device or order.dtype != torch.int64 or order.shape != (64,) or not order.is_contiguous():
        raise ValueError('Expected owned int64 window permutation')
    if rows not in (8, 16, 32, 64):
        raise ValueError('Unsupported row tile')
    shape = (heads, h // 8, w // 8, 64, 32)
    q, k, v = [torch.empty(shape, dtype=z.dtype, device=z.device) for _ in range(3)]
    kernel = _pack[(triton.cdiv(h * w * heads, rows), 3)](
        z, scale, order, q, k, v, h * w * heads, w, heads, h // 8, w // 8, rows,
        ROUND_QKV=round_qkv,
        num_warps=4, enable_fp_fusion=False)
    return (q, k, v), kernel
