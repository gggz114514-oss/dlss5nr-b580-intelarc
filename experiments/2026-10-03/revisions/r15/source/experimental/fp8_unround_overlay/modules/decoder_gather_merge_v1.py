"""Exact nearest-neighbour gather + compensated half FMA, with explicit outputs.

Inputs are the original unquantized projected half image and the already FP8
rounded skip. No contraction/reassociation, new quantization, or input crop.
Separate entry points keep the raw C32 residual out of the FP8-domain contract.
"""
import torch
import triton
import triton.language as tl
from nr_backend.triton_cubic_fp8 import _half_fma_value
from nr_backend.triton_fp8 import _round_fp8_half
from spill_preflight_v1 import select


@triton.jit
def _value(P, S, SCALE, y, x, c, valid,
           PS: tl.constexpr, SS: tl.constexpr, SC: tl.constexpr):
    p = tl.load(P + (y // 2) * PS[0] + (x // 2) * PS[1] + c * PS[2], valid, other=0)
    s = tl.load(S + y * SS[0] + x * SS[1] + c * SS[2], valid, other=0)
    scale = tl.load(SCALE + c * SC, valid, other=0)
    return _half_fma_value(s, scale, p)


@triton.jit
def _quantized(P, S, SCALE, OUT, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
               PS: tl.constexpr, SS: tl.constexpr, SC: tl.constexpr, B: tl.constexpr,
               ROUND_OUTPUT: tl.constexpr=True):
    i = tl.program_id(0) * B + tl.arange(0, B)
    c, x, y = i % C, (i // C) % W, i // (C * W)
    merged = _value(P, S, SCALE, y, x, c, i < H * W * C, PS, SS, SC)
    if ROUND_OUTPUT:
        merged = _round_fp8_half(merged)
    tl.store(OUT + i, merged, i < H * W * C)


@triton.jit
def _raw_padded(P, S, SCALE, OUT, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
                PH: tl.constexpr, PW: tl.constexpr, SY: tl.constexpr, SX: tl.constexpr,
                PS: tl.constexpr, SS: tl.constexpr, SC: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    c, x, y = i % C, (i // C) % PW - SX, i // (C * PW) - SY
    valid = (i < PH * PW * C) & (y >= 0) & (y < H) & (x >= 0) & (x < W)
    merged = _value(P, S, SCALE, y, x, c, valid, PS, SS, SC)
    tl.store(OUT + i, tl.where(valid, merged, tl.full((), 0., tl.float16)), i < PH * PW * C)


def forward(projected, skip, scale, *, shift=None, round_output=True):
    if projected.ndim != 3 or skip.ndim != 3:
        raise ValueError('Expected HWC tensors')
    h, w, c = skip.shape
    if projected.shape[2] != c or scale.shape != (c,) or min(h, w) <= 0:
        raise ValueError('Channel/skip geometry mismatch')
    if h > 2 * projected.shape[0] or w > 2 * projected.shape[1]:
        raise ValueError('Skip exceeds expanded projection')
    values = (projected, skip, scale)
    if any(t.dtype != torch.float16 or t.device.type != 'xpu' or t.device != skip.device for t in values):
        raise ValueError('Expected half operands on one XPU')
    if any(s < 0 for t in values for s in t.stride()):
        raise ValueError('Negative strides are not supported')
    ps, ss, sc = tuple(projected.stride()), tuple(skip.stride()), scale.stride(0)
    if shift is None:
        out = torch.empty((h, w, c), dtype=skip.dtype, device=skip.device)
        jit, args = _quantized, (projected, skip, scale, out, h, w, c, ps, ss, sc, 256, round_output)
    else:
        sy, sx = shift
        if c != 32 or not (0 <= sy < 8 and 0 <= sx < 8):
            raise ValueError('Raw merge is only reviewed for C32 window padding')
        ph, pw = triton.cdiv(h + sy, 8) * 8, triton.cdiv(w + sx, 8) * 8
        out = torch.empty((ph, pw, c), dtype=skip.dtype, device=skip.device)
        jit, args = _raw_padded, (projected, skip, scale, out, h, w, c, ph, pw, sy, sx, ps, ss, sc, 256)
    grid = (triton.cdiv(out.numel(), 256),)
    options = dict(num_warps=4, enable_fp_fusion=False)
    _, kernel, decision = select(jit, [(256,)], lambda _: args, lambda _: grid, **options)
    assert jit[grid](*args, **options) is kernel
    return out, dict(hash=kernel.hash, spills=kernel.n_spills, registers=kernel.n_regs,
                     shared_bytes=kernel.metadata.shared, selection=decision)
