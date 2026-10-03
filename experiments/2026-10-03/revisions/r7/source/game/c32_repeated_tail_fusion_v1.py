"""Isolated C32 encoder/decoder attention-to-projection game candidate.

The seven ordinary C32 blocks retain their MLP, FP8 input/QKV boundaries,
downsampling, and callers. Only attention, attended FP8, output projection,
residual and crop use the previously validated post window-tail kernel.
Install around an offline capture; this module is never a default game option.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch

from nr_backend.c32_block import C32SwinBlock
from nr_backend.unround_policy import round_c32 as q, ENABLED
from nr_backend.executor import ResetNR
from fused_c32_projection_native_half_v1 import forward as project_window_qkv
from post_attention_fusion_v1 import _attention_project


_SHAPES = {(192, 384, 32), (256, 448, 32), (320, 512, 32),
           (384, 640, 32)}


def ordinary_blocks(model: ResetNR) -> tuple[C32SwinBlock, ...]:
    """The four encoder blocks and three plain decoder blocks, never pre/post."""
    blocks = tuple(model.encoder[0]) + tuple(model.decoder[-1][1:])
    if len(blocks) != 7 or any(type(block) is not C32SwinBlock for block in blocks):
        raise ValueError("Expected the seven ordinary C32 blocks")
    return blocks


def fused_forward_unquantized(module: C32SwinBlock, features: torch.Tensor) -> torch.Tensor:
    if tuple(features.shape) not in _SHAPES or features.dtype != torch.float16 or \
            features.device.type != "xpu" or not features.is_contiguous():
        raise ValueError("Expected contiguous XPU FP16 HWC32 on a validated game feature canvas; "
                         f"actual shape={tuple(features.shape)} dtype={features.dtype} "
                         f"device={features.device} stride={features.stride()}")
    height, width, _ = features.shape
    shift_y, shift_x = module.window_shift
    if shift_y not in (0, 4) or shift_x not in (0, 4):
        raise ValueError("Unsupported C32 shift")

    x = q(features)
    if shift_y or shift_x:
        x = torch.nn.functional.pad(x, (0, 0, shift_x, shift_x, shift_y, shift_y))
    mlp = module.mlp.forward_unquantized(x)
    (query, key, value), _ = project_window_qkv(
        q(mlp), module.attention.front.qkv, module.attention.front.scale,
        module.attention.pixel_order, bm=32, warps=4, stages=1,
        round_qkv="c32" not in ENABLED)
    out = torch.empty((height, width, 32), device=features.device, dtype=torch.float16)
    windows = mlp.shape[0] * mlp.shape[1] // 64
    _attention_project[(2, windows)](
        mlp, query, key, value, module.attention.bias,
        module.output_weight, module.skip_scale,
        module.attention.pixel_order, out,
        height, width, shift_y, shift_x,
        "c32" not in ENABLED,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    return out


@contextmanager
def installed(model: ResetNR):
    blocks = ordinary_blocks(model)
    if any("forward_unquantized" in block.__dict__ for block in blocks):
        raise ValueError("A C32 instance already has a forward override")
    replacements = []
    primary = None
    try:
        for block in blocks:
            replacement = MethodType(fused_forward_unquantized, block)
            block.forward_unquantized = replacement
            replacements.append((block, replacement))
        yield
    except BaseException as exc:
        primary = exc
        raise
    finally:
        violations = []
        for block, replacement in reversed(replacements):
            if block.forward_unquantized is not replacement:
                violations.append(id(block))
            else:
                del block.forward_unquantized
        if violations:
            message = f"C32 fusion foreign overrides retained; other owned slots restored: {violations}"
            if primary is not None:
                primary.add_note(message)
            else:
                raise RuntimeError(message)
