"""Isolated C64 block-6 QKV producer with static post-dot channel splits.

Only encoder[1][1]'s QKV dot and native-half pack are replaced. The existing
window attention, output projection, FP8 boundary and all other blocks remain
on the installed implementation. Install before a new graph capture and do not
switch this scope on a graph captured with a different producer.
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


_SHAPES = {(104, 200), (136, 232), (168, 264)}


@triton.jit
def _qkv_direct_pack(X, WEIGHT, SCALE, INVERSE, Q, K, V,
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
    for start in range(0, 64, BK):
        k = start + kk
        x = tl.load(X + row[:, None] * 64 + k[None, :],
                    row[:, None] < COUNT, other=0)
        w = tl.load(WEIGHT + k[:, None] * 192 +
                    (segment * 32 + col[None, :]))
        acc = tl.dot(x, w, acc, out_dtype=tl.float32)

    # The original XMX producer stores this tensor in half before _pack reads it.
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
            total = _nan_left((total.to(tl.float32) + other.to(tl.float32)).to(tl.float16),
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


def direct_pack(features: torch.Tensor, module) -> tuple[torch.Tensor, ...]:
    """Produce the installed attention's native window layout for one C64 block."""
    if (features.device.type != "xpu" or features.dtype != torch.float16 or
            not features.is_contiguous() or features.ndim != 3 or
            features.shape[-1] != 64 or tuple(features.shape[:2]) not in _SHAPES or
            module.heads != 2 or module.qkv.device != features.device or
            module.qkv.dtype != torch.float16 or not module.qkv.is_contiguous() or
            tuple(module.qkv.shape) != (64, 192) or
            module.scale.shape != (2,) or module.scale.dtype != torch.float16 or
            not module.scale.is_contiguous() or
            module.pixel_inverse.shape != (64,) or
            module.pixel_inverse.dtype != torch.int64 or
            not module.pixel_inverse.is_contiguous()):
        raise ValueError("Unexpected installed C64 QKV/attention boundary")
    h, w = features.shape[:2]
    shape = (2, h // 8, w // 8, 64, 32)
    q, k, v = [torch.empty(shape, dtype=torch.float16, device=features.device)
               for _ in range(3)]
    _qkv_direct_pack[(triton.cdiv(h * w, 16), 6)](
        features, module.qkv, module.scale, module.pixel_inverse, q, k, v,
        h * w, w, h * w // 64, 16, 32, 32,
        num_warps=4, enable_fp_fusion=False)
    return q, k, v


@contextmanager
def installed(window_blocks: WindowBlocks, model):
    """Scope the one block's producer; a fresh session/graph is still required."""
    if not isinstance(window_blocks, WindowBlocks) or "windows" in window_blocks.__dict__:
        raise ValueError("Expected one unmodified WindowBlocks instance")
    target = model.encoder[1][1].attention
    if target.channels != 64 or target.heads != 2 or model.encoder[1][1].window_shift != (4, 4):
        raise ValueError("Expected shifted C64 encoder block 6")
    original = type(window_blocks).windows
    calls = {"direct_pack": 0}

    def replacement(self, module, features):
        if module is not target:
            return original(self, module, features)
        h, w, c = features.shape
        if (h, w) not in _SHAPES or c != 64:
            raise ValueError("Unexpected target C64 padded geometry")
        q, k, v = direct_pack(blocks.quantize_fp8(features), module)
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
            raise RuntimeError("C64 producer replaced during its scope")
        del window_blocks.windows
