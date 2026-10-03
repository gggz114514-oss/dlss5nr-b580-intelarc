"""Whole FP16 branched MLP in one kernel; preserve the original half/FP8 boundaries.

Each of 2/4/8 branches performs C->128, exact cubic+E4M3, 128->32,
E4M3, 32->C and half-rounded accumulation. Intermediates remain in kernel-local
values. FP16 matrix reductions still use consecutive K32 blocks and FP32 sums.
"""
import torch
import triton
import triton.language as tl
from nr_backend.triton_fp8 import _round_fp8_half
from nr_backend.triton_cubic_fp8 import _half_fma_value


@triton.jit
def _cubic(x, ROUND_OUTPUT:tl.constexpr=True):
    t = tl.minimum(tl.maximum(x.to(tl.float32), -4.), 4.).to(tl.float16)
    p = _half_fma_value(-tl.abs(t).to(tl.float32), tl.full((), .055908203125, tl.float32), tl.full((), .447265625, tl.float32))
    v = _half_fma_value(t, p, tl.full((), .89453125, tl.float32))
    activated=(x.to(tl.float32) * v.to(tl.float32)).to(tl.float16)
    if ROUND_OUTPUT:
        return _round_fp8_half(activated)
    return activated


@triton.jit
def _kernel(X, EXPAND, REDUCE, PROJECT, SCALE, OUT, M:tl.constexpr,
            C:tl.constexpr, BM:tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    channels = tl.arange(0, C)
    hidden_columns = tl.arange(0, 128)
    small_columns = tl.arange(0, 32)
    x = tl.load(X + rows[:, None] * C + channels[None, :], rows[:, None] < M, other=0)
    x = _round_fp8_half(x)
    scale = tl.load(SCALE + channels)
    result = (x.to(tl.float32) * scale[None, :].to(tl.float32)).to(tl.float16)
    for branch in range(C // 32):
        expanded = tl.full((BM, 128), 0., tl.float32)
        for start in range(C // 32):
            kk = start * 32 + small_columns
            av = tl.gather(x, tl.broadcast_to(kk[None, :], (BM, 32)), 1)
            weight = tl.load(EXPAND + branch * C * 128 + kk[:, None] * 128 + hidden_columns[None, :])
            expanded = tl.dot(av, weight, expanded, out_dtype=tl.float32)
        hidden = _cubic(expanded.to(tl.float16))
        reduced = tl.full((BM, 32), 0., tl.float32)
        for start in range(4):
            kk = start * 32 + small_columns
            hv = tl.gather(hidden, tl.broadcast_to(kk[None, :], (BM, 32)), 1)
            weight = tl.load(REDUCE + branch * 128 * 32 + kk[:, None] * 32 + small_columns[None, :])
            reduced = tl.dot(hv, weight, reduced, out_dtype=tl.float32)
        small = _round_fp8_half(reduced.to(tl.float16))
        weight = tl.load(PROJECT + branch * 32 * C + small_columns[:, None] * C + channels[None, :])
        projected = tl.dot(small, weight, out_dtype=tl.float32)
        result = (projected + result.to(tl.float32)).to(tl.float16)
    tl.store(OUT + rows[:, None] * C + channels[None, :], result, rows[:, None] < M)


def forward(features, expand, reduce, project, skip_scale, *, bm=16):
    c = features.shape[-1]
    if c not in (64, 128, 256) or bm not in (16, 32):
        raise ValueError('Expected C64/C128/C256 and BM16/32')
    b = c // 32
    expected = [(b,c,128), (b,128,32), (b,32,c), (c,)]
    for value, shape in zip((expand, reduce, project, skip_scale), expected):
        if value.shape != shape or value.dtype != torch.float16 or value.device != features.device or not value.is_contiguous():
            raise ValueError('Invalid registered MLP weights')
    if features.device.type != 'xpu' or features.numel() == 0:
        raise ValueError('Expected nonempty XPU features')
    x = features.half().contiguous()
    out = torch.empty_like(x)
    m = x.numel() // c
    kernel = _kernel[(triton.cdiv(m, bm),)](x, expand, reduce, project, skip_scale, out,
        m, c, bm, num_warps=4, enable_fp_fusion=False)
    return out, kernel
