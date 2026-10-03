"""Experimental FP16 window attention: QK, old exponential/weights, then V.

Keep the original half reduction order and FP8 weight boundary, without global
64x64 score/exponential/weight tensors. Not ordinary softmax/FlashAttention.
"""
import torch
import triton
import triton.language as tl
from nr_backend.triton_cubic_fp8 import _half_fma_value
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
def _weights(e, BM: tl.constexpr):
    lane = tl.arange(0, 8)
    v0 = tl.gather(e, tl.broadcast_to(lane[None, :], (BM, 8)), 1)
    v8 = tl.gather(e, tl.broadcast_to((lane + 8)[None, :], (BM, 8)), 1)
    v16 = tl.gather(e, tl.broadcast_to((lane + 16)[None, :], (BM, 8)), 1)
    v24 = tl.gather(e, tl.broadcast_to((lane + 24)[None, :], (BM, 8)), 1)
    v32 = tl.gather(e, tl.broadcast_to((lane + 32)[None, :], (BM, 8)), 1)
    v40 = tl.gather(e, tl.broadcast_to((lane + 40)[None, :], (BM, 8)), 1)
    v48 = tl.gather(e, tl.broadcast_to((lane + 48)[None, :], (BM, 8)), 1)
    v56 = tl.gather(e, tl.broadcast_to((lane + 56)[None, :], (BM, 8)), 1)
    partial = _add_half(_add_half(_add_half(_add_half(v0, v8), _add_half(v16, v24)), _add_half(v32, v40)), _add_half(v48, v56))
    small = tl.arange(0, 2)
    t0 = tl.gather(partial, tl.broadcast_to(small[None, :], (BM, 2)), 1)
    t2 = tl.gather(partial, tl.broadcast_to((small + 2)[None, :], (BM, 2)), 1)
    t4 = tl.gather(partial, tl.broadcast_to((small + 4)[None, :], (BM, 2)), 1)
    t6 = tl.gather(partial, tl.broadcast_to((small + 6)[None, :], (BM, 2)), 1)
    total = _add_half(_add_half(_add_half(t0, t2), t4), t6)
    a = tl.gather(total, tl.full((BM, 1), 0, tl.int32), 1)
    b = tl.gather(total, tl.full((BM, 1), 1, tl.int32), 1)
    reciprocal = reciprocal_half_clamped(_add_half(a, b))
    product = _nan_left((e.to(tl.float32) * reciprocal.to(tl.float32)).to(tl.float16), e, reciprocal)
    return _round_fp8_half(product)


@triton.jit
def _kernel(Q, K, V, BIAS, OUT, BM: tl.constexpr):
    window = tl.program_id(1)
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    lane = tl.arange(0, 32)
    key = tl.arange(0, 64)
    q = tl.load(Q + window * 2048 + row[:, None] * 32 + lane[None, :])
    k = tl.load(K + window * 2048 + key[None, :] * 32 + lane[:, None])
    score = tl.dot(q, k, out_dtype=tl.float32)
    bias = tl.load(BIAS + row[:, None] * 64 + key[None, :])
    e = _exp((score + bias.to(tl.float32)).to(tl.float16))
    w = _weights(e, BM)
    result = tl.full((BM, 32), 0., tl.float32)
    for block in tl.static_range(2):
        ii = block * 32 + lane
        weight = tl.gather(w, tl.broadcast_to(ii[None, :], (BM, 32)), 1)
        value = tl.load(V + window * 2048 + ii[:, None] * 32 + lane[None, :])
        result = tl.dot(weight, value, result, out_dtype=tl.float32)
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
