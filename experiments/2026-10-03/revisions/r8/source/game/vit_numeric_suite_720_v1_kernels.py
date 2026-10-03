"""Default-off kernels for the owned 240-token ViT numeric suite.

CPU source handoff only. No compilation, GPU execution or quality claim.
QKV uses FP16 matrix operands, one full-K FP32 accumulator, one half store.
Ordered denominator retains the exact reference 64-key half tree, four
successive half totals and left-NaN selection. FP32 mode sums all 256 keys
in FP32, then rounds its total to half before the original half padding
correction/subtraction and clamp/reciprocal. Neither mode masks padding
before summation.
"""
from __future__ import annotations

import triton
import triton.language as tl
from nr_backend.triton_attention_weights import _add_half


TOKENS, WIDTH, QKV_WIDTH, ROWS = 240, 1024, 3072, 32 * 240


@triton.jit
def _qkv_full_k(X, W, OUT, BM: tl.constexpr, BN: tl.constexpr):
    TOKENS: tl.constexpr = 240
    WIDTH: tl.constexpr = 1024
    QKV_WIDTH: tl.constexpr = 3072
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    col = tl.program_id(1) * BN + tl.arange(0, BN)
    lane = tl.arange(0, 32)
    total = tl.full((BM, BN), 0.0, tl.float32)
    for block in range(WIDTH // 32):
        kk = block * 32 + lane
        x = tl.load(X + row[:, None] * WIDTH + kk[None, :],
                    row[:, None] < TOKENS, other=0)
        w = tl.load(W + kk[:, None] * QKV_WIDTH + col[None, :],
                    col[None, :] < QKV_WIDTH, other=0)
        total = tl.dot(x, w, total, out_dtype=tl.float32)
    tl.store(OUT + row[:, None] * QKV_WIDTH + col[None, :],
             total.to(tl.float16),
             (row[:, None] < TOKENS) & (col[None, :] < QKV_WIDTH))


@triton.jit
def _ordered_sum64(X, rows, lanes, segment: tl.constexpr,
                   COUNT: tl.constexpr, BR: tl.constexpr):
    offsets = rows[:, None] * 256 + segment * 64 + lanes[None, :]
    valid = rows[:, None] < COUNT
    v0 = tl.load(X + offsets, valid, other=0)
    v8 = tl.load(X + offsets + 8, valid, other=0)
    v16 = tl.load(X + offsets + 16, valid, other=0)
    v24 = tl.load(X + offsets + 24, valid, other=0)
    v32 = tl.load(X + offsets + 32, valid, other=0)
    v40 = tl.load(X + offsets + 40, valid, other=0)
    v48 = tl.load(X + offsets + 48, valid, other=0)
    v56 = tl.load(X + offsets + 56, valid, other=0)
    p0 = _add_half(v0, v8)
    p1 = _add_half(v16, v24)
    p2 = _add_half(v32, v40)
    p3 = _add_half(v48, v56)
    partial = _add_half(_add_half(_add_half(p0, p1), p2), p3)
    small = tl.arange(0, 2)
    t0 = tl.gather(partial, tl.broadcast_to(small[None, :], (BR, 2)), 1)
    t2 = tl.gather(partial, tl.broadcast_to((small + 2)[None, :], (BR, 2)), 1)
    t4 = tl.gather(partial, tl.broadcast_to((small + 4)[None, :], (BR, 2)), 1)
    t6 = tl.gather(partial, tl.broadcast_to((small + 6)[None, :], (BR, 2)), 1)
    pair = _add_half(_add_half(_add_half(t0, t2), t4), t6)
    a = tl.gather(pair, tl.full((BR, 1), 0, tl.int32), 1)
    b = tl.gather(pair, tl.full((BR, 1), 1, tl.int32), 1)
    return _add_half(a, b)


@triton.jit
def _denominator(X, OUT, COUNT: tl.constexpr, BR: tl.constexpr,
                 ORDERED: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    if ORDERED:
        lanes = tl.arange(0, 8)
        total = _ordered_sum64(X, rows, lanes, 0, COUNT, BR)
        for segment in tl.static_range(1, 4):
            part = _ordered_sum64(X, rows, lanes, segment, COUNT, BR)
            total = _add_half(total, part)
        tl.store(OUT + rows[:, None], total, rows[:, None] < COUNT)
    else:
        lanes = tl.arange(0, 256)
        values = tl.load(X + rows[:, None] * 256 + lanes[None, :],
                         rows[:, None] < COUNT, other=0)
        total = tl.sum(values.to(tl.float32), axis=1)
        tl.store(OUT + rows, total.to(tl.float16), rows < COUNT)
