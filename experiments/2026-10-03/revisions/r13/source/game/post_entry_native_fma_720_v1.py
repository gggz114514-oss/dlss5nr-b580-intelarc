"""Isolated lossy 720p fast-post padded-entry FMA candidates.

Only the installed midpoint-corrected merge is changed. The FP8 input policy,
half-rounded feature expansion, 4-pixel halo, shape, and output dtype match
post_entry_mlp_fusion_v1.padded. Nothing here installs into the game runtime.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from nr_backend.triton_fp8 import _round_fp8_half
from nr_backend.unround_policy import ENABLED


_COMPILED = {}


@triton.jit
def _native_entry(FEATURES, SKIP, INPUT_WEIGHT, INPUT_SKIP_SCALE, OUT,
                  BM: tl.constexpr, ROUND_INPUT: tl.constexpr,
                  HALF_FMA: tl.constexpr):
    padded_width: tl.constexpr = 1288
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    lane = tl.arange(0, 32)
    y = row // padded_width - 4
    x = row % padded_width - 4
    valid = ((row < 776 * padded_width) &
             (y >= 0) & (y < 768) & (x >= 0) & (x < 1280))
    feature_index = ((y // 2) * 640 + x // 2)[:, None] * 32 + lane[None, :]
    skip_index = (y * 1280 + x)[:, None] * 32 + lane[None, :]
    feature = tl.load(FEATURES + feature_index, valid[:, None], other=0)
    skip = tl.load(SKIP + skip_index, valid[:, None], other=0)
    weight = tl.load(INPUT_WEIGHT + lane)
    scale = tl.load(INPUT_SKIP_SCALE + lane)
    if ROUND_INPUT:
        feature = _round_fp8_half(feature)
        skip = _round_fp8_half(skip)
    expanded = (feature.to(tl.float32) * weight[None, :].to(tl.float32)).to(tl.float16)
    if HALF_FMA:
        # Request one native FP16 fused operation. Inspect the compiled ISA
        # before claiming that this lowers to a particular B580 instruction.
        merged = tl.fma(skip, scale[None, :], expanded).to(tl.float16)
    else:
        # FP16 operands with one native FP32 FMA and a final FP16 store.
        # This is intentionally a separate numerical candidate.
        merged = tl.fma(skip.to(tl.float32), scale[None, :].to(tl.float32),
                        expanded.to(tl.float32)).to(tl.float16)
    value = tl.where(valid[:, None], merged, tl.full((BM, 32), 0, tl.float16))
    tl.store(OUT + row[:, None] * 32 + lane[None, :], value,
             row[:, None] < 776 * padded_width)


def _validate(module, features, skip, *, bm, mode):
    if mode not in ("fp16_fma", "fp32_fma") or bm not in (16, 32):
        raise ValueError("Expected fp16_fma/fp32_fma and BM16/BM32")
    if (tuple(features.shape) != (384, 640, 32) or
            tuple(skip.shape) != (768, 1280, 32) or
            features.device.type != "xpu" or skip.device != features.device or
            features.dtype != torch.float16 or skip.dtype != torch.float16 or
            not features.is_contiguous() or not skip.is_contiguous()):
        raise ValueError("Expected actual contiguous 720p post XPU FP16 HWC32 inputs")
    for value in (module.input_weight, module.input_skip_scale):
        if (tuple(value.shape) != (32,) or value.device != skip.device or
                value.dtype != torch.float16 or not value.is_contiguous()):
            raise ValueError("Unexpected post-entry weight layout")


def _params(module, features, skip, output, *, bm, mode):
    return (features, skip, module.input_weight, module.input_skip_scale,
            output, bm, "post" not in ENABLED, mode == "fp16_fma")


def preflight(module, features, skip, *, mode="fp16_fma", bm=32):
    """Compile the real geometry and reject a spilling binary before timing."""
    from spill_preflight_v1 import select

    _validate(module, features, skip, bm=bm, mode=mode)
    output = torch.empty((776, 1288, 32), device=skip.device, dtype=torch.float16)
    params = _params(module, features, skip, output, bm=bm, mode=mode)
    _, compiled, report = select(
        _native_entry, [(bm,)], lambda _: params,
        lambda _: (triton.cdiv(776 * 1288, bm),),
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    _COMPILED[(skip.device, mode, bm, "post" not in ENABLED)] = compiled
    return report


def padded(module, features, skip, *, mode="fp16_fma", bm=32):
    """Return the same padded HWC32 boundary as the installed stage A."""
    _validate(module, features, skip, bm=bm, mode=mode)
    round_input = "post" not in ENABLED
    compiled = _COMPILED.get((skip.device, mode, bm, round_input))
    if compiled is None:
        raise RuntimeError("Native-FMA post entry requires no-spill preflight")
    output = torch.empty((776, 1288, 32), device=skip.device, dtype=torch.float16)
    launched = _native_entry[(triton.cdiv(776 * 1288, bm),)](
        *_params(module, features, skip, output, bm=bm, mode=mode),
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    if launched is not compiled:
        raise RuntimeError("Native-FMA post launch missed its screened binary")
    return output
