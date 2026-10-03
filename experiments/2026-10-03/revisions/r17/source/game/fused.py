"""Isolated fractional-motion candidate. GPU correctness is intentionally unproven.

The kernel mirrors backend/nr_backend/sampling.py, including its measured
half-texture integer arithmetic. It must pass probe.py's stage and byte gates
before it can be considered for any session. No product code imports this file.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

START_BITS = 0x3F780000
END_BITS = 0x3F940000


@triton.jit
def _axis(norm, extent: tl.constexpr, TABLE, p, base, A, BAD, N: tl.constexpr):
    start_bits: tl.constexpr = 0x3F780000
    table_size: tl.constexpr = 0x3F940000 - 0x3F780000
    center = tl.floor(tl.fma(norm, extent, -0.5)) + 0.5
    t = tl.minimum(1.0, tl.maximum(0.0, tl.fma(norm, extent, -center)))
    t2 = t * t
    t3 = t * t2
    w0 = tl.fma(t + t3, -0.5, t2)
    w1 = tl.fma(t3, 1.5, -(t2 * 2.5)) + 1.0
    w3 = (t3 - t2) * 0.5
    w2 = ((1.0 - w0) - w1) - w3
    group = w1 + w2
    index = group.to(tl.int32, bitcast=True) - start_bits
    good = (index >= 0) & (index < table_size)
    inverse = tl.load(TABLE + tl.minimum(table_size - 1, tl.maximum(0, index)))
    middle = tl.fma(w2, inverse, center)
    tl.store(A + (base + 0) * N + p, center, p < N)
    tl.store(A + (base + 1) * N + p, w0, p < N)
    tl.store(A + (base + 2) * N + p, group, p < N)
    tl.store(A + (base + 3) * N + p, w3, p < N)
    tl.store(A + (base + 4) * N + p, middle, p < N)
    return ~good


@triton.jit
def _prepare_axes(MOTION, IW, IH, TABLE, AXES, INVALID,
                  H: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = p < H * W
    x = p % W
    y = p // W
    mx = tl.load(MOTION + p * 2, valid, 0).to(tl.float16).to(tl.float32)
    my = tl.load(MOTION + p * 2 + 1, valid, 0).to(tl.float16).to(tl.float32)
    iw = tl.load(IW)
    ih = tl.load(IH)
    u = tl.fma(mx, 1.0 / W, (x.to(tl.float32) + 0.5) * iw)
    v = tl.fma(my, 1.0 / H, (y.to(tl.float32) + 0.5) * ih)
    badx = _axis(u, W, TABLE, p, 0, AXES, INVALID, H * W)
    bady = _axis(v, H, TABLE, p, 5, AXES, INVALID, H * W)
    tl.store(INVALID + p, (badx | bady).to(tl.int32), valid)


@triton.jit
def _exp32(value):
    bits = value.to(tl.int32, bitcast=True)
    return tl.where(value == 0.0, -1000, ((bits >> 23) & 255) - 127)


@triton.jit
def _half_texture(IMAGE, px, py, ch, IW, IH,
                  H: tl.constexpr, W: tl.constexpr):
    # Same model-normalized -> region -> backing-normalized round trip.
    iw = tl.load(IW)
    ih = tl.load(IH)
    xc = tl.minimum(W - 0.5, tl.maximum(0.5, px))
    yc = tl.minimum(H - 0.5, tl.maximum(0.5, py))
    u = tl.fma(xc * iw, W, 0.0) * (1.0 / W)
    v = tl.fma(yc * ih, H, 0.0) * (1.0 / H)
    nx = tl.floor(u * 2097152.0).to(tl.int64)
    ny = tl.floor(v * 2097152.0).to(tl.int64)
    cx = ((nx * W + 4096) >> 13) - 128
    cy = ((ny * H + 4096) >> 13) - 128
    cx = tl.minimum((W - 1) * 256, tl.maximum(0, cx))
    cy = tl.minimum((H - 1) * 256, tl.maximum(0, cy))
    ix = (cx >> 8).to(tl.int32)
    iy = (cy >> 8).to(tl.int32)
    jx = tl.minimum(ix + 1, W - 1)
    jy = tl.minimum(iy + 1, H - 1)
    ax = (cx & 255).to(tl.int32)
    ay = (cy & 255).to(tl.int32)
    cross = (ax * ay + 128) >> 8
    k0 = 256 - ax - ay + cross
    k1 = ax - cross
    k2 = ay - cross
    k3 = cross
    a0 = tl.load(IMAGE + (iy * W + ix) * 3 + ch).to(tl.float16)
    a1 = tl.load(IMAGE + (iy * W + jx) * 3 + ch).to(tl.float16)
    a2 = tl.load(IMAGE + (jy * W + ix) * 3 + ch).to(tl.float16)
    a3 = tl.load(IMAGE + (jy * W + jx) * 3 + ch).to(tl.float16)
    f0 = a0.to(tl.float32)
    f1 = a1.to(tl.float32)
    f2 = a2.to(tl.float32)
    f3 = a3.to(tl.float32)
    e0 = tl.where((f0 != 0.0) & (k0 != 0), _exp32(tl.abs(f0)), -1000)
    e1 = tl.where((f1 != 0.0) & (k1 != 0), _exp32(tl.abs(f1)), -1000)
    e2 = tl.where((f2 != 0.0) & (k2 != 0), _exp32(tl.abs(f2)), -1000)
    e3 = tl.where((f3 != 0.0) & (k3 != 0), _exp32(tl.abs(f3)), -1000)
    exponent = tl.maximum(-24, tl.maximum(tl.maximum(e0, e1), tl.maximum(e2, e3)))
    scale_bits = ((14 - exponent + 127) << 23).to(tl.int32)
    scale = scale_bits.to(tl.float32, bitcast=True)
    summed = ((f0 * scale).to(tl.int64) * k0.to(tl.int64) +
              (f1 * scale).to(tl.int64) * k1.to(tl.int64) +
              (f2 * scale).to(tl.int64) * k2.to(tl.int64) +
              (f3 * scale).to(tl.int64) * k3.to(tl.int64))
    magnitude = tl.abs(summed)
    lead = tl.minimum(62, tl.maximum(0, _exp32(magnitude.to(tl.float32))))
    lead = lead - (magnitude < (tl.full(magnitude.shape, 1, tl.int64) << lead)).to(tl.int32)
    lead = tl.where(magnitude == 0, 0, lead)
    scale_exponent = exponent - 22
    result_exponent = lead + scale_exponent
    shift = tl.maximum(result_exponent - 10, -24) - scale_exponent
    rs = tl.minimum(62, tl.maximum(0, shift))
    quotient = magnitude >> rs
    remainder = magnitude - (quotient << rs)
    midpoint = tl.full(magnitude.shape, 1, tl.int64) << tl.minimum(61, tl.maximum(0, rs - 1))
    rounded = quotient + ((rs > 0) & (remainder >= midpoint)).to(tl.int64)
    rounded = tl.where(shift < 0, magnitude << tl.minimum(62, tl.maximum(0, -shift)), rounded)
    bits = tl.minimum(0x7c00, tl.maximum(0, result_exponent + 14).to(tl.int64) * 1024 + rounded)
    bits = tl.where(magnitude == 0, 0, bits) | ((summed < 0).to(tl.int64) << 15)
    negzero = (summed == 0) & ((k0 == 0) | ((a0.to(tl.uint16, bitcast=True) & 0x8000) != 0)) & ((k1 == 0) | ((a1.to(tl.uint16, bitcast=True) & 0x8000) != 0)) & ((k2 == 0) | ((a2.to(tl.uint16, bitcast=True) & 0x8000) != 0)) & ((k3 == 0) | ((a3.to(tl.uint16, bitcast=True) & 0x8000) != 0))
    bits = bits.to(tl.uint16) | (negzero.to(tl.uint16) << 15)
    fractional = bits.to(tl.float16, bitcast=True).to(tl.float32)
    # Backend's integer-coordinate branch returns the original half texel.
    return tl.where(((ax == 0) & (ay == 0)), f0, fractional)


@triton.jit
def _five_tap(IMAGE, AXES, IW, IH, TABLE, NUMERATOR, RECIPROCAL, NORMALIZED,
              INVALID, TAPS, H: tl.constexpr, W: tl.constexpr,
              DEBUG: tl.constexpr, BLOCK: tl.constexpr):
    start_bits: tl.constexpr = 0x3F780000
    table_size: tl.constexpr = 0x3F940000 - 0x3F780000
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = q < H * W * 3
    p = q // 3
    ch = q % 3
    n = H * W
    bx = tl.load(AXES + p, valid, 0)
    x0 = tl.load(AXES + n + p, valid, 0)
    gx = tl.load(AXES + 2 * n + p, valid, 0)
    x3 = tl.load(AXES + 3 * n + p, valid, 0)
    mx = tl.load(AXES + 4 * n + p, valid, 0)
    by = tl.load(AXES + 5 * n + p, valid, 0)
    y0 = tl.load(AXES + 6 * n + p, valid, 0)
    gy = tl.load(AXES + 7 * n + p, valid, 0)
    y3 = tl.load(AXES + 8 * n + p, valid, 0)
    my = tl.load(AXES + 9 * n + p, valid, 0)
    k0, k1, k2, k3, k4 = x0 * gy, y0 * gx, gx * gy, y3 * gx, x3 * gy
    total = (((k0 + k1) + k2) + k3) + k4
    index = total.to(tl.int32, bitcast=True) - start_bits
    good = (index >= 0) & (index < table_size)
    inverse = tl.load(TABLE + tl.minimum(table_size - 1, tl.maximum(0, index)))
    s1 = _half_texture(IMAGE, mx, by - 1.0, ch, IW, IH, H, W)
    s0 = _half_texture(IMAGE, bx - 1.0, my, ch, IW, IH, H, W)
    s2 = _half_texture(IMAGE, mx, my, ch, IW, IH, H, W)
    s3 = _half_texture(IMAGE, mx, by + 2.0, ch, IW, IH, H, W)
    s4 = _half_texture(IMAGE, bx + 2.0, my, ch, IW, IH, H, W)
    if DEBUG:
        tl.store(TAPS + q, s0, valid)
        tl.store(TAPS + n * 3 + q, s1, valid)
        tl.store(TAPS + n * 6 + q, s2, valid)
        tl.store(TAPS + n * 9 + q, s3, valid)
        tl.store(TAPS + n * 12 + q, s4, valid)
    value = s1 * k1
    value = tl.fma(s0, k0, value)
    value = tl.fma(s2, k2, value)
    value = tl.fma(s3, k3, value)
    value = tl.fma(s4, k4, value)
    tl.store(NUMERATOR + q, value, valid)
    tl.store(NORMALIZED + q, value * inverse, valid)
    tl.store(RECIPROCAL + p, inverse, valid & (ch == 0))
    tl.store(INVALID + p, (~good).to(tl.int32), valid & (ch == 0))


def prepare_axes(motion: torch.Tensor, reciprocal_values: torch.Tensor,
                 iw: torch.Tensor, ih: torch.Tensor):
    if motion.ndim != 3 or motion.shape[-1] != 2 or motion.device.type != 'xpu':
        raise ValueError('Expected HWC2 XPU motion')
    if motion.dtype not in (torch.float16, torch.float32):
        raise ValueError('Expected FP16/FP32 motion')
    h, w = motion.shape[:2]
    if not (1 <= h <= 4096 and 1 <= w <= 4096):
        raise ValueError('Native dimension table is defined through 4096')
    if not motion.is_contiguous():
        motion = motion.contiguous()
    if reciprocal_values.dtype != torch.float32 or reciprocal_values.numel() != END_BITS - START_BITS:
        raise ValueError('Expected authenticated native reciprocal table')
    if iw.dtype != torch.float32 or ih.dtype != torch.float32:
        raise ValueError('Expected native FP32 dimension reciprocals')
    if any(t.device != motion.device for t in (reciprocal_values, iw, ih)):
        raise ValueError('All tensors must share the XPU device')
    axes = torch.empty((10, h, w), device=motion.device, dtype=torch.float32)
    invalid = torch.empty((h, w), device=motion.device, dtype=torch.int32)
    _prepare_axes[(triton.cdiv(h * w, 128),)](
        motion, iw, ih, reciprocal_values, axes, invalid, h, w, 128,
        num_warps=4, enable_fp_fusion=False)
    return axes, invalid


def sample_five(image: torch.Tensor, axes: torch.Tensor, reciprocal_values: torch.Tensor,
                iw: torch.Tensor, ih: torch.Tensor, *, debug: bool = False):
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != torch.float16 or image.device.type != 'xpu':
        raise ValueError('Expected contiguous HWC3 FP16 XPU history')
    h, w = image.shape[:2]
    if tuple(axes.shape) != (10, h, w) or axes.dtype != torch.float32:
        raise ValueError('Axis tensor must match history')
    if any(not t.is_contiguous() for t in (image, axes, reciprocal_values)):
        raise ValueError('Inputs must be contiguous')
    if any(t.device != image.device for t in (axes, reciprocal_values, iw, ih)):
        raise ValueError('All tensors must share the XPU device')
    numerator = torch.empty((h, w, 3), device=image.device, dtype=torch.float32)
    normalized = torch.empty_like(numerator)
    reciprocal = torch.empty((h, w), device=image.device, dtype=torch.float32)
    invalid = torch.empty((h, w), device=image.device, dtype=torch.int32)
    taps = torch.empty((5, h, w, 3), device=image.device, dtype=torch.float32) if debug else numerator
    _five_tap[(triton.cdiv(h * w * 3, 64),)](
        image, axes, iw, ih, reciprocal_values, numerator, reciprocal, normalized,
        invalid, taps, h, w, debug, 64, num_warps=4, enable_fp_fusion=False)
    return numerator, reciprocal, normalized, invalid, taps if debug else None
