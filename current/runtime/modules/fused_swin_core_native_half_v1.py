# Explicit native-half attention version; see native_half_attention_fma_v1.py.
"""Split/reshape FP16 window attention, avoiding gather layout conversion.

Keep the original half reduction order and FP8 weight boundary, without global
64x64 score/exponential/weight tensors. Not ordinary softmax/FlashAttention.
"""
import torch
import triton
import triton.language as tl
from native_half_attention_fma_v1 import half_fma_attention as _half_fma_value
from nr_backend.triton_fp8 import _round_fp8_half
from nr_backend.triton_attention_weights import _add_half, reciprocal_half_clamped
from nr_backend.triton_attention_normalize import _nan_left


@triton.jit
def _exp(scores):
    affine = _half_fma_value(scores, tl.full((), 0.044921875, tl.float32), tl.full((), 1.30078125, tl.float32))
    affine = tl.minimum(tl.maximum(affine.to(tl.float32), 1.03125), 1.5693359375).to(tl.float16)
    bits = affine.to(tl.uint16, bitcast=True).to(tl.int32)
    inp = scores.to(tl.uint16, bitcast=True).to(tl.int32)
    bits = tl.where((inp & 0x7fff) > 0x7c00, inp | 0x200, bits)
    return (((bits << 5) + 0x8000) & 0xffff).to(tl.uint16).to(tl.float16, bitcast=True)


@triton.jit
def _halves(x, BM: tl.constexpr, N: tl.constexpr):
    return tl.split(tl.permute(tl.reshape(x, (BM, 2, N // 2)), (0, 2, 1)))


@triton.jit
def _sum4(x, BM: tl.constexpr):
    a, b = _halves(x, BM, 4)
    x0, x1 = tl.split(a)
    x2, x3 = tl.split(b)
    return _add_half(_add_half(_add_half(x0, x1), x2), x3)


@triton.jit
def _weights_pair(e0, e1, BM: tl.constexpr):
    a0, a1 = _halves(e0, BM, 32)
    a2, a3 = _halves(e1, BM, 32)
    v0, v8 = _halves(a0, BM, 16)
    v16, v24 = _halves(a1, BM, 16)
    v32, v40 = _halves(a2, BM, 16)
    v48, v56 = _halves(a3, BM, 16)
    partial = _add_half(_add_half(_add_half(_add_half(v0, v8), _add_half(v16, v24)), _add_half(v32, v40)), _add_half(v48, v56))
    even, odd = tl.split(tl.reshape(partial, (BM, 4, 2)))
    denominator = _add_half(_sum4(even, BM), _sum4(odd, BM))
    reciprocal = reciprocal_half_clamped(denominator)[:, None]
    p0 = _nan_left((e0.to(tl.float32) * reciprocal.to(tl.float32)).to(tl.float16), e0, reciprocal)
    p1 = _nan_left((e1.to(tl.float32) * reciprocal.to(tl.float32)).to(tl.float16), e1, reciprocal)
    return _round_fp8_half(p0), _round_fp8_half(p1)


@triton.jit
def _kernel(Q, K, V, BIAS, OUT, BM: tl.constexpr):
    window = tl.program_id(1)
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    lane = tl.arange(0, 32)
    q = tl.load(Q + window * 2048 + row[:, None] * 32 + lane[None, :])
    k0 = tl.load(K + window * 2048 + lane[None, :] * 32 + lane[:, None])
    k1 = tl.load(K + window * 2048 + (lane[None, :] + 32) * 32 + lane[:, None])
    score0 = tl.dot(q, k0, out_dtype=tl.float32)
    score1 = tl.dot(q, k1, out_dtype=tl.float32)
    b0 = tl.load(BIAS + row[:, None] * 64 + lane[None, :])
    b1 = tl.load(BIAS + row[:, None] * 64 + lane[None, :] + 32)
    e0 = _exp((score0 + b0.to(tl.float32)).to(tl.float16))
    e1 = _exp((score1 + b1.to(tl.float32)).to(tl.float16))
    w0, w1 = _weights_pair(e0, e1, BM)
    v0 = tl.load(V + window * 2048 + lane[:, None] * 32 + lane[None, :])
    v1 = tl.load(V + window * 2048 + (lane[:, None] + 32) * 32 + lane[None, :])
    result = tl.dot(w0, v0, out_dtype=tl.float32)
    result = tl.dot(w1, v1, result, out_dtype=tl.float32)
    tl.store(OUT + window * 2048 + row[:, None] * 32 + lane[None, :], result.to(tl.float16))


def forward(query, key, value, bias, *, bm=16, warps=4, stages=1):
    if query.shape != key.shape or query.shape != value.shape or query.shape[-2:] != (64, 32) or bias.shape != (64, 64):
        raise ValueError('Expected matching complete 64x32 windows and a 64x64 bias')
    if bm not in (16, 32, 64) or warps not in (4, 8) or stages not in (1, 2):
        raise ValueError('Unsupported window launch configuration')
    for t in (query, key, value, bias):
        if t.device.type != 'xpu' or t.device != query.device or t.dtype != torch.float16 or not t.is_contiguous() or t.numel() == 0:
            raise ValueError('Expected contiguous half XPU window tensors')
    out = torch.empty_like(query)
    compiled = _kernel[(64 // bm, query.numel() // 2048)](query, key, value, bias, out, bm,
        num_warps=warps, num_stages=stages, enable_fp_fusion=False)
    return out, compiled
