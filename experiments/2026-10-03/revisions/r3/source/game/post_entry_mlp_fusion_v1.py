"""Isolated post entry / complete C32 MLP fusion for reviewed fast-path sizes.

Stage A writes the exact padded HWC32 boundary. Stage B consumes the same
register-resident value in the installed LUT MLP and writes only its output.
Neither stage changes the model's weights, history, motion, or output head.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from nr_backend.triton_fp8 import _round_fp8_half
from nr_backend.unround_policy import ENABLED
from nr_backend.execution import record_arithmetic_dispatch
from fused_c32_mlp_lut_v1 import lookup

_PREFLIGHT = {}


@triton.jit
def _finite_half_fma(a, b, c):
    """Match the installed finite-half FMA's midpoint correction."""
    product = a.to(tl.float32) * b.to(tl.float32)
    addend = c.to(tl.float32)
    summed = product + addend
    virtual = summed - product
    residual = (product - (summed - virtual)) + (addend - virtual)
    rounded = summed.to(tl.float16)
    bits = rounded.to(tl.uint16, bitcast=True).to(tl.int32)
    negative = (bits & 0x8000) != 0
    virtual_overflow = tl.where(negative, -65536.0, 65536.0)
    rounded_float = tl.where((bits & 0x7fff) == 0x7c00,
                             virtual_overflow, rounded.to(tl.float32))
    adjacent_bits = bits + tl.where((summed > rounded_float) != negative, 1, -1)
    adjacent = adjacent_bits.to(tl.uint16).to(tl.float16, bitcast=True)
    adjacent_float = tl.where((adjacent_bits & 0x7fff) == 0x7c00,
                              virtual_overflow, adjacent.to(tl.float32))
    midpoint = (rounded_float + adjacent_float) * 0.5
    correction = ((summed == midpoint) & (residual != 0) &
                  ((summed > rounded_float) == (residual > 0)))
    return tl.where(correction, adjacent, rounded)


@triton.jit
def _merged(FEATURES, SKIP, INPUT_WEIGHT, INPUT_SKIP_SCALE,
            H: tl.constexpr, W: tl.constexpr, BM: tl.constexpr,
            ROUND_INPUT: tl.constexpr):
    pw: tl.constexpr = W + 8
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    lane = tl.arange(0, 32)
    y = row // pw - 4
    x = row % pw - 4
    valid = (row < (H + 8) * pw) & (y >= 0) & (y < H) & (x >= 0) & (x < W)
    feature_index = ((y // 2) * (W // 2) + x // 2)[:, None] * 32 + lane[None, :]
    skip_index = (y * W + x)[:, None] * 32 + lane[None, :]
    feature = tl.load(FEATURES + feature_index, valid[:, None], other=0)
    skip = tl.load(SKIP + skip_index, valid[:, None], other=0)
    weight = tl.load(INPUT_WEIGHT + lane)
    scale = tl.load(INPUT_SKIP_SCALE + lane)
    if ROUND_INPUT:
        feature = _round_fp8_half(feature)
        skip = _round_fp8_half(skip)
    expanded = (feature.to(tl.float32) * weight[None, :].to(tl.float32)).to(tl.float16)
    merged = _finite_half_fma(skip, scale[None, :], expanded)
    return tl.where(valid[:, None], merged, tl.full((BM, 32), 0, tl.float16))


@triton.jit
def _entry_kernel(FEATURES, SKIP, INPUT_WEIGHT, INPUT_SKIP_SCALE, OUT,
                  H: tl.constexpr, W: tl.constexpr, BM: tl.constexpr,
                  ROUND_INPUT: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    lane = tl.arange(0, 32)
    merged = _merged(FEATURES, SKIP, INPUT_WEIGHT, INPUT_SKIP_SCALE,
                     H, W, BM, ROUND_INPUT)
    tl.store(OUT + row[:, None] * 32 + lane[None, :], merged,
             row[:, None] < (H + 8) * (W + 8))


@triton.jit
def _entry_mlp_kernel(FEATURES, SKIP, INPUT_WEIGHT, INPUT_SKIP_SCALE,
                      EXPAND, CONTRACT, SCALE, LUT, OUT,
                      H: tl.constexpr, W: tl.constexpr, BM: tl.constexpr,
                      ROUND_INPUT: tl.constexpr):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    lane = tl.arange(0, 32)
    raw = _merged(FEATURES, SKIP, INPUT_WEIGHT, INPUT_SKIP_SCALE,
                  H, W, BM, ROUND_INPUT)
    x = _round_fp8_half(raw)
    contracted = tl.full((BM, 32), 0., tl.float32)
    for part in range(4):
        weight = tl.load(EXPAND + lane[:, None] * 128 +
                         part * 32 + lane[None, :])
        expanded = tl.dot(x, weight, out_dtype=tl.float32)
        hidden = lookup(expanded.to(tl.float16), LUT)
        weight = tl.load(CONTRACT + (part * 32 + lane[:, None]) * 32 +
                         lane[None, :])
        contracted = tl.dot(hidden, weight, contracted, out_dtype=tl.float32)
    scale = tl.load(SCALE + lane)
    initial = (raw.to(tl.float32) * scale[None, :].to(tl.float32)).to(tl.float16)
    value = (contracted + initial.to(tl.float32)).to(tl.float16)
    tl.store(OUT + row[:, None] * 32 + lane[None, :], value,
             row[:, None] < (H + 8) * (W + 8))


def _validate(module, features, skip, *, lut=None):
    geometries = {
        (512, 896): (256, 448),
        (640, 1024): (320, 512),
        (768, 1280): (384, 640),
    }
    if (tuple(skip.shape) not in {(h, w, 32) for h, w in geometries} or
            tuple(features.shape) != (*geometries[tuple(skip.shape[:2])], 32)):
        raise ValueError("Unsupported post skip/features canvas pair")
    if (features.device.type != "xpu" or skip.device != features.device or
            features.dtype != torch.float16 or skip.dtype != torch.float16 or
            not features.is_contiguous() or not skip.is_contiguous()):
        raise ValueError("Expected contiguous same-device XPU FP16 HWC32 inputs")
    weights = ((module.input_weight, (32,)),
               (module.input_skip_scale, (32,)))
    if lut is not None:
        weights += ((module.body.mlp.expansion, (32, 128)),
                    (module.body.mlp.contraction, (128, 32)),
                    (module.body.mlp.skip_scale, (32,)))
    for value, shape in weights:
        if (tuple(value.shape) != shape or value.device != skip.device or
                value.dtype != torch.float16 or not value.is_contiguous()):
            raise ValueError("Unexpected post weight layout")
    if lut is not None and (lut.device != skip.device or
                            lut.dtype != torch.int16 or lut.shape != (65536,) or
                            not lut.is_contiguous()):
        raise ValueError("Expected the installed cubic/FP8 half-bit LUT")


def preflight(module, features, skip, *, stage, lut_guard=None, bm=32):
    """Compile/load one reviewed shape and reject a spilling kernel before launch."""
    from spill_preflight_v1 import select

    if stage not in ("padded", "mlp") or bm not in (16, 32):
        raise ValueError("Expected post padded/MLP stage and BM16/BM32")
    lut = lut_guard.require() if stage == "mlp" else None
    _validate(module, features, skip, lut=lut)
    h, w = skip.shape[:2]
    round_input = "post" not in ENABLED
    out = torch.empty((h + 8, w + 8, 32), device=skip.device, dtype=torch.float16)
    if stage == "padded":
        jit = _entry_kernel
        args = (features, skip, module.input_weight, module.input_skip_scale,
                out, h, w, bm, round_input)
    else:
        block = module.body.mlp
        jit = _entry_mlp_kernel
        args = (features, skip, module.input_weight, module.input_skip_scale,
                block.expansion, block.contraction, block.skip_scale, lut,
                out, h, w, bm, round_input)
    _, compiled, report = select(
        jit, [(bm,)], lambda _config: args,
        lambda _config: (triton.cdiv((h + 8) * (w + 8), bm),),
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    _PREFLIGHT[(stage, h, w, bm, round_input)] = compiled
    return report


def padded(module, features, skip, *, bm=32):
    """A: replace expansion, merge and pad; keep the installed MLP."""
    _validate(module, features, skip)
    if bm not in (16, 32):
        raise ValueError("BM must be 16 or 32")
    h, w = skip.shape[:2]
    round_input = "post" not in ENABLED
    compiled = _PREFLIGHT.get(("padded", h, w, bm, round_input))
    if compiled is None:
        raise RuntimeError("Post padded kernel requires no-spill preflight")
    out = torch.empty((h + 8, w + 8, 32), device=skip.device, dtype=torch.float16)
    launched = _entry_kernel[(triton.cdiv((h + 8) * (w + 8), bm),)](
        features, skip, module.input_weight, module.input_skip_scale, out,
        h, w, bm, round_input,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    if launched is not compiled:
        raise RuntimeError("Post padded launch did not reuse its screened binary")
    for kind in (("fp8", "fp8", "half_fma") if round_input else ("half_fma",)):
        record_arithmetic_dispatch(kind)
    return out


def mlp(module, features, skip, lut_guard, *, bm=32):
    """B: do A and the complete installed LUT C32 MLP in one kernel."""
    lut = lut_guard.require()
    _validate(module, features, skip, lut=lut)
    if bm not in (16, 32):
        raise ValueError("BM must be 16 or 32")
    h, w = skip.shape[:2]
    round_input = "post" not in ENABLED
    compiled = _PREFLIGHT.get(("mlp", h, w, bm, round_input))
    if compiled is None:
        raise RuntimeError("Post MLP kernel requires no-spill preflight")
    out = torch.empty((h + 8, w + 8, 32), device=skip.device, dtype=torch.float16)
    block = module.body.mlp
    launched = _entry_mlp_kernel[(triton.cdiv((h + 8) * (w + 8), bm),)](
        features, skip, module.input_weight, module.input_skip_scale,
        block.expansion, block.contraction, block.skip_scale, lut, out,
        h, w, bm, round_input,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    if launched is not compiled:
        raise RuntimeError("Post MLP launch did not reuse its screened binary")
    for kind in (("fp8", "fp8", "half_fma") if round_input else ("half_fma",)):
        record_arithmetic_dispatch(kind)
    for _ in range(triton.cdiv((h + 8) * (w + 8), 32768)):
        for kind in ("fp8", "cubic_fp8", "dense", "dense"):
            record_arithmetic_dispatch(kind)
    return out
