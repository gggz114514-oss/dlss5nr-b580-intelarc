"""Isolated C128 QKV producer for encoder[2][1] in the game fast chain.

The original half XMX dot's stored-half boundary, head normalization, FP8
rounding and attention consumer layout are retained. A scoped instance method
replaces only one producer on a fresh capture; game and exact code stay intact.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch
import triton
import triton.language as tl

from nr_backend.execution import record_arithmetic_dispatch
from nr_backend.triton_attention_normalize import _nan_left, rsqrt_half_clamped
from nr_backend.triton_fp8 import _round_fp8_half
from native_half_attention_fma_v1 import half_fma_attention as _half_fma_value
from fused_swin_core_native_half_v1 import _halves
from window_block_attention_v3 import forward as attend
from window_blocks_v3 import WindowBlocks
import nr_backend.multihead_block as blocks


_SHAPES = {360: (56, 104), 480: (72, 120), 540: (88, 136)}


@triton.jit
def _direct_pack(X, WEIGHT, SCALE, INVERSE, Q, K, V,
                 COUNT: tl.constexpr, WIDTH: tl.constexpr,
                 WINDOWS_PER_HEAD: tl.constexpr,
                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                 ROUND_QKV: tl.constexpr = True):
    row = tl.program_id(0) * BM + tl.arange(0, BM)
    segment = tl.program_id(1)
    head = segment // 3
    family = segment % 3
    col = tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(0, 128, BK):
        k = start + kk
        x = tl.load(X + row[:, None] * 128 + k[None, :],
                    row[:, None] < COUNT, other=0)
        w = tl.load(WEIGHT + k[:, None] * 384 +
                    (segment * 32 + col[None, :]))
        acc = tl.dot(x, w, acc, out_dtype=tl.float32)

    # Existing matrix kernel writes FP16 before the pack reloads it.
    z = acc.to(tl.float16)
    lane = tl.arange(0, 8)
    lo, hi = _halves(z, BM, 32)
    x0, x8 = _halves(lo, BM, 16)
    x16, x24 = _halves(hi, BM, 16)
    if family < 2:
        a = (x16.to(tl.float32) * x16.to(tl.float32)).to(tl.float16)
        a = _nan_left(_half_fma_value(x0, x0, a), x0, a)
        b = (x24.to(tl.float32) * x24.to(tl.float32)).to(tl.float16)
        b = _nan_left(_half_fma_value(x8, x8, b), x8, b)
        total = _nan_left((a.to(tl.float32) + b.to(tl.float32)).to(tl.float16), a, b)
        for mask in tl.static_range(3):
            other = tl.gather(total, tl.broadcast_to(
                (lane ^ (4 >> mask))[None, :], (BM, 8)), 1)
            total = _nan_left(
                (total.to(tl.float32) + other.to(tl.float32)).to(tl.float16),
                total, other)
        denominator = tl.gather(total, tl.full((BM, 1), 0, tl.int32), 1)
        scale = rsqrt_half_clamped(denominator)
        x0 = _nan_left((x0.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x0, scale)
        x8 = _nan_left((x8.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x8, scale)
        x16 = _nan_left((x16.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x16, scale)
        x24 = _nan_left((x24.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x24, scale)
        if family == 0:
            head_scale = tl.load(SCALE + head)
            x0 = _nan_left((x0.to(tl.float32) * head_scale.to(tl.float32)).to(tl.float16), x0, head_scale)
            x8 = _nan_left((x8.to(tl.float32) * head_scale.to(tl.float32)).to(tl.float16), x8, head_scale)
            x16 = _nan_left((x16.to(tl.float32) * head_scale.to(tl.float32)).to(tl.float16), x16, head_scale)
            x24 = _nan_left((x24.to(tl.float32) * head_scale.to(tl.float32)).to(tl.float16), x24, head_scale)

    y = row // WIDTH
    x = row % WIDTH
    window = (y // 8) * (WIDTH // 8) + x // 8
    pixel = (y % 8) * 8 + x % 8
    token = tl.load(INVERSE + pixel, row < COUNT, other=0).to(tl.int32)
    output = tl.where(family == 0, Q, tl.where(family == 1, K, V))
    offset = ((head * WINDOWS_PER_HEAD + window) * 64 + token) * 32
    valid = row[:, None] < COUNT
    lane_offset = lane[None, :]
    if ROUND_QKV:
        x0 = _round_fp8_half(x0)
        x8 = _round_fp8_half(x8)
        x16 = _round_fp8_half(x16)
        x24 = _round_fp8_half(x24)
    tl.store(output + offset[:, None] + lane_offset,
             x0, valid)
    tl.store(output + offset[:, None] + lane_offset + 8,
             x8, valid)
    tl.store(output + offset[:, None] + lane_offset + 16,
             x16, valid)
    tl.store(output + offset[:, None] + lane_offset + 24,
             x24, valid)


def direct_pack(features: torch.Tensor, module, *, height: int):
    if (height not in _SHAPES or features.device.type != "xpu" or
            features.dtype != torch.float16 or not features.is_contiguous() or
            tuple(features.shape) != (*_SHAPES[height], 128) or
            module.heads != 4 or tuple(module.qkv.shape) != (128, 384) or
            module.qkv.device != features.device or
            module.qkv.dtype != torch.float16 or not module.qkv.is_contiguous() or
            tuple(module.scale.shape) != (4,) or module.scale.dtype != torch.float16 or
            not module.scale.is_contiguous() or
            tuple(module.pixel_inverse.shape) != (64,) or
            module.pixel_inverse.dtype != torch.int64 or
            not module.pixel_inverse.is_contiguous()):
        raise ValueError("Unexpected installed C128 QKV boundary")
    h, w = _SHAPES[height]
    shape = (4, h // 8, w // 8, 64, 32)
    q, k, v = [torch.empty(shape, dtype=torch.float16, device=features.device)
               for _ in range(3)]
    _direct_pack[(triton.cdiv(h * w, 16), 12)](
        features, module.qkv, module.scale, module.pixel_inverse, q, k, v,
        h * w, w, h * w // 64, 16, 32, 32,
        num_warps=4, enable_fp_fusion=False)
    return q, k, v


@contextmanager
def installed(window_blocks: WindowBlocks, model, *, height: int):
    if not isinstance(window_blocks, WindowBlocks) or height not in _SHAPES:
        raise ValueError("Expected installed WindowBlocks and known game mode")
    block = model.encoder[2][1]
    target = block.attention
    if block.channels != 128 or block.window_shift != (4, 4) or target.heads != 4:
        raise ValueError("Unexpected C128 encoder block 1")
    had_instance = "windows" in window_blocks.__dict__
    original = window_blocks.windows
    calls = {"direct_pack": 0}

    def replacement(self, module, features):
        if module is not target:
            return original(module, features)
        h, w = _SHAPES[height]
        if tuple(features.shape) != (h, w, 128):
            raise ValueError("Unexpected padded C128 target shape")
        q, k, v = direct_pack(blocks.quantize_fp8(features), module, height=height)
        for _ in range(2):
            record_arithmetic_dispatch("attention_normalize_c32")
        for _ in range(3):
            record_arithmetic_dispatch("fp8")
        result, self.last_attention_kernel, self.last_attention_selection = attend(
            q, k, v, module.bias)
        for _ in range(module.heads * ((h // 8 * (w // 8) + 1023) // 1024)):
            for key in ("batched", "attention_exp_swin", "attention_weights", "batched"):
                record_arithmetic_dispatch(key)
        for key, n in (("multi", 1), ("batched_heads", module.heads), ("qkv_pack", 1)):
            self.layout.calls[key] = self.layout.calls.get(key, 0) + n
        calls["direct_pack"] += 1
        return result

    bound = MethodType(replacement, window_blocks)
    window_blocks.windows = bound
    try:
        yield calls
    finally:
        if window_blocks.__dict__.get("windows") is not bound:
            raise RuntimeError("C128 QKV producer replaced during its scope")
        if had_instance:
            window_blocks.windows = original
        else:
            del window_blocks.windows
