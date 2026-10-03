"""Fast-only decoder merges with one native half FMA.

Nearest sampling, padded zero context and the independently selected output
FP8 boundary stay unchanged. Replaces the compensated half-FMA calculation;
does not change the projection matrix, skip quantization or data layout.
"""
import torch
import triton
import triton.language as tl

from nr_backend.triton_fp8 import _round_fp8_half
from spill_preflight_v1 import select


@triton.jit
def _value(P, S, SCALE, y, x, c, valid,
           PS: tl.constexpr, SS: tl.constexpr, SC: tl.constexpr):
    p = tl.load(P + (y // 2) * PS[0] + (x // 2) * PS[1] + c * PS[2], valid, other=0)
    s = tl.load(S + y * SS[0] + x * SS[1] + c * SS[2], valid, other=0)
    scale = tl.load(SCALE + c * SC, valid, other=0)
    return tl.fma(s.to(tl.float16), scale.to(tl.float16), p.to(tl.float16)).to(tl.float16)


@triton.jit
def _quantized(P, S, SCALE, OUT, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
               PS: tl.constexpr, SS: tl.constexpr, SC: tl.constexpr, B: tl.constexpr,
               ROUND_OUTPUT: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    c, x, y = i % C, (i // C) % W, i // (C * W)
    value = _value(P, S, SCALE, y, x, c, i < H * W * C, PS, SS, SC)
    if ROUND_OUTPUT:
        value = _round_fp8_half(value)
    tl.store(OUT + i, value, i < H * W * C)


@triton.jit
def _raw_padded(P, S, SCALE, OUT, H: tl.constexpr, W: tl.constexpr, C: tl.constexpr,
                PH: tl.constexpr, PW: tl.constexpr, SY: tl.constexpr, SX: tl.constexpr,
                PS: tl.constexpr, SS: tl.constexpr, SC: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    c, x, y = i % C, (i // C) % PW - SX, i // (C * PW) - SY
    valid = (i < PH * PW * C) & (y >= 0) & (y < H) & (x >= 0) & (x < W)
    value = _value(P, S, SCALE, y, x, c, valid, PS, SS, SC)
    tl.store(OUT + i, tl.where(valid, value, tl.full((), 0., tl.float16)), i < PH * PW * C)


def parameters(projected, skip, scale, *, shift=None, round_output=True):
    if projected.ndim != 3 or skip.ndim != 3:
        raise ValueError("Expected HWC decoder operands")
    h, w, c = skip.shape
    if (projected.shape[2] != c or scale.shape != (c,) or min(h, w) <= 0 or
            h > 2 * projected.shape[0] or w > 2 * projected.shape[1]):
        raise ValueError("Decoder merge geometry changed")
    values = (projected, skip, scale)
    if any(t.dtype != torch.float16 or t.device.type != "xpu" or
           t.device != skip.device or any(s < 0 for s in t.stride()) for t in values):
        raise ValueError("Expected half operands with nonnegative strides on one XPU")
    ps, ss, sc = projected.stride(), skip.stride(), scale.stride(0)
    if shift is None:
        out = torch.empty_like(skip, memory_format=torch.contiguous_format)
        jit = _quantized
        args = (projected, skip, scale, out, h, w, c, ps, ss, sc, 256, round_output)
    else:
        sy, sx = shift
        if c != 32 or not (0 <= sy < 8 and 0 <= sx < 8):
            raise ValueError("Raw decoder merge requires the existing C32 padding contract")
        ph, pw = triton.cdiv(h + sy, 8) * 8, triton.cdiv(w + sx, 8) * 8
        out = torch.empty((ph, pw, c), device=skip.device, dtype=skip.dtype)
        jit = _raw_padded
        args = (projected, skip, scale, out, h, w, c, ph, pw, sy, sx, ps, ss, sc, 256)
    return jit, args, out, (triton.cdiv(out.numel(), 256),)


def preflight(projected, skip, scale, *, shift=None, round_output=True):
    jit, args, out, grid = parameters(projected, skip, scale, shift=shift,
                                     round_output=round_output)
    _, kernel, report = select(jit, [(256,)], lambda _: args, lambda _: grid,
                               num_warps=4, enable_fp_fusion=False)
    return kernel, report


def forward(projected, skip, scale, *, shift=None, round_output=True):
    jit, args, out, grid = parameters(projected, skip, scale, shift=shift,
                                     round_output=round_output)
    _, kernel, report = select(jit, [(256,)], lambda _: args, lambda _: grid,
                               num_warps=4, enable_fp_fusion=False)
    if jit[grid](*args, num_warps=4, enable_fp_fusion=False) is not kernel:
        raise RuntimeError("Native decoder FMA missed its screened binary")
    return out, dict(hash=kernel.hash, spills=kernel.n_spills, registers=kernel.n_regs,
                     shared_bytes=kernel.metadata.shared, selection=report)
