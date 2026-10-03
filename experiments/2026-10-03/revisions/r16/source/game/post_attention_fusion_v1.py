"""Isolated post attention/projection candidate; never enabled by default.

Reuse the validated C32 window-tail arithmetic with the post block's own weights.
The MLP, padded input, RGB head, and temporal output remain on their current paths.
Install only around an offline capture/paired comparison; no protected backend edit.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch
import triton
import triton.language as tl

from nr_backend.post import ResetPostBlock, dot, half_fma, q
from nr_backend.unround_policy import ENABLED
from nr_backend.triton_fp8 import _round_fp8_half
from fused_c32_projection_native_half_v1 import forward as project_window_qkv
from fused_swin_core_native_half_v1 import _exp, _weights_pair


@triton.jit
def _attention_project(MLP, Q, K, V, BIAS, PROJECT, SKIPSCALE, ORDER, OUT,
                       H: tl.constexpr, W: tl.constexpr,
                       SY: tl.constexpr, SX: tl.constexpr,
                       ROUND_ATTENDED: tl.constexpr):
    """C32 native window attention, FP8 publish, projection and half residual."""
    window = tl.program_id(1)
    row = tl.program_id(0) * 32 + tl.arange(0, 32)
    lane = tl.arange(0, 32)
    query = tl.load(Q + window * 2048 + row[:, None] * 32 + lane[None, :])
    key0 = tl.load(K + window * 2048 + lane[None, :] * 32 + lane[:, None])
    key1 = tl.load(K + window * 2048 + (lane[None, :] + 32) * 32 + lane[:, None])
    score0 = tl.dot(query, key0, out_dtype=tl.float32)
    score1 = tl.dot(query, key1, out_dtype=tl.float32)
    bias0 = tl.load(BIAS + row[:, None] * 64 + lane[None, :])
    bias1 = tl.load(BIAS + row[:, None] * 64 + 32 + lane[None, :])
    exp0 = _exp((score0 + bias0.to(tl.float32)).to(tl.float16))
    exp1 = _exp((score1 + bias1.to(tl.float32)).to(tl.float16))
    probability0, probability1 = _weights_pair(exp0, exp1, 32,
                                               ROUND_WEIGHTS=ROUND_ATTENDED)
    value0 = tl.load(V + window * 2048 + lane[:, None] * 32 + lane[None, :])
    value1 = tl.load(V + window * 2048 + (lane[:, None] + 32) * 32 + lane[None, :])
    attended = tl.dot(probability0, value0, out_dtype=tl.float32)
    attended = tl.dot(probability1, value1, attended, out_dtype=tl.float32)
    if ROUND_ATTENDED:
        attended = _round_fp8_half(attended.to(tl.float16))
    else:
        attended = attended.to(tl.float16)
    projection = tl.load(PROJECT + lane[:, None] * 32 + lane[None, :])
    projected = tl.dot(attended, projection, out_dtype=tl.float32)
    pixel = tl.load(ORDER + row).to(tl.int32)
    columns = (W + 2 * SX) // 8
    y = window // columns * 8 + pixel // 8 - SY
    x = window % columns * 8 + pixel % 8 - SX
    original = tl.load(MLP + ((y + SY) * (W + 2 * SX) + (x + SX))[:, None] * 32 + lane[None, :])
    skip_scale = tl.load(SKIPSCALE + lane)
    initial = (original.to(tl.float32) * skip_scale[None, :].to(tl.float32)).to(tl.float16)
    projected = (projected + initial.to(tl.float32)).to(tl.float16)
    valid = (y >= 0) & (y < H) & (x >= 0) & (x < W)
    tl.store(OUT + (y * W + x)[:, None] * 32 + lane[None, :], projected, valid[:, None])


def fused_forward_head(module: ResetPostBlock, features: torch.Tensor,
                       skip: torch.Tensor, *, head_dot=None,
                       entry_fn=None, attention_fn=None) -> torch.Tensor:
    if features.ndim != 3 or skip.ndim != 3 or features.shape[-1] != 32 or skip.shape[-1] != 32:
        raise ValueError("Expected HWC32 post features and skip")
    height, width = skip.shape[:2]
    geometries = {
        (512, 896): (256, 448),
        (640, 1024): (320, 512),
        (768, 1280): (384, 640),
    }
    if tuple(features.shape[:2]) != geometries.get((height, width)):
        raise ValueError("Unsupported post skip/features canvas pair")
    if skip.device.type != "xpu" or features.device != skip.device:
        raise ValueError("Expected same-device XPU post tensors")
    shift_y, shift_x = module.body.window_shift
    if (shift_y, shift_x) != (4, 4):
        raise ValueError("Unexpected post window shift")

    if entry_fn is None:
        expanded = (q(features).repeat_interleave(2, 0).repeat_interleave(2, 1)
                    * module.input_weight).half()
        merged = half_fma(q(skip), module.input_skip_scale, expanded)
        padded = torch.nn.functional.pad(merged, (0, 0, 4, 4, 4, 4))
        mlp = module.body.mlp.forward_unquantized(padded)
    else:
        mlp = entry_fn(module, features, skip)
        if (tuple(mlp.shape) != (height + 8, width + 8, 32) or
                mlp.device != skip.device or mlp.dtype != torch.float16 or
                not mlp.is_contiguous()):
            raise RuntimeError("Post entry candidate changed the MLP interface")
    if attention_fn is None:
        (query, key, value), _ = project_window_qkv(
            q(mlp), module.body.attention.front.qkv, module.body.attention.front.scale,
            module.body.attention.pixel_order, bm=32, warps=4, stages=1,
            round_qkv="post" not in ENABLED)
        output = torch.empty((height, width, 32), device=skip.device, dtype=torch.float16)
        windows = (height + 8) * (width + 8) // 64
        _attention_project[(2, windows)](
            mlp, query, key, value, module.body.attention.bias,
            module.body.output_weight, module.body.skip_scale,
            module.body.attention.pixel_order, output,
            height, width, shift_y, shift_x,
            "post" not in ENABLED,
            num_warps=4, num_stages=1, enable_fp_fusion=False)
    else:
        output = attention_fn(module, mlp, height, width)
        if (output.shape != (height, width, 32) or output.dtype != torch.float16
                or output.device != skip.device or not output.is_contiguous()):
            raise RuntimeError("Experimental post attention changed its output interface")
    if head_dot is None:
        return dot(output, module.head_weight, chunk_k=8)
    return head_dot(output, module.head_weight)


@contextmanager
def installed(module: ResetPostBlock):
    if not isinstance(module, ResetPostBlock) or "forward_head" in module.__dict__:
        raise ValueError("Expected one unmodified post block")
    replacement = MethodType(fused_forward_head, module)
    module.forward_head = replacement
    try:
        yield
    finally:
        if module.forward_head is not replacement:
            raise RuntimeError("Post fusion was replaced during its scope")
        del module.forward_head
