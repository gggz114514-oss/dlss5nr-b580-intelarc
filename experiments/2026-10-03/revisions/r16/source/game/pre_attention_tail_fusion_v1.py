"""One-instance pre attention-to-projection experiment for the game fast chain.

The unrounded full projection remains available for the original downsample.
Only the attention result, attended FP8 and output projection are fused; this
candidate is never enabled by default.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch

from nr_backend.front import project_front_features
from nr_backend.execution import current_arithmetic_backend
from nr_backend.pre_block import PreBlock
from nr_backend.unround_policy import round_pre as q, ENABLED
from fused_c32_projection_native_half_v1 import forward as project_window_qkv
from post_attention_fusion_v1 import _attention_project


_SHAPES = {480: (512, 896, 16), 540: (640, 1024, 16),
           720: (768, 1280, 16)}


def fused_forward_features_unquantized(module: PreBlock,
                                       features: torch.Tensor) -> torch.Tensor:
    shape = tuple(features.shape)
    if (shape not in _SHAPES.values() or features.device.type != "xpu" or
            features.dtype != torch.float16 or not features.is_contiguous()):
        raise ValueError(f"Unexpected game pre canvas: {shape}, {features.dtype}")
    height, width, _ = shape
    projected = project_front_features(features, module.front_weight)
    mlp = module.mlp.forward_unquantized(projected)
    (query, key, value), _ = project_window_qkv(
        q(mlp), module.attention.front.qkv, module.attention.front.scale,
        module.attention.pixel_order, bm=32, warps=4, stages=1,
        round_qkv="pre" not in ENABLED)
    full = torch.empty((height, width, 32), device=features.device,
                       dtype=torch.float16)
    _attention_project[(2, height * width // 64)](
        mlp, query, key, value, module.attention.bias,
        module.output_weight, module.skip_scale,
        module.attention.pixel_order, full,
        height, width, 0, 0,
        "pre" not in ENABLED,
        num_warps=4, num_stages=1, enable_fp_fusion=False)
    return full


@contextmanager
def installed(model, *, height: int):
    if height not in _SHAPES or not isinstance(model.pre, PreBlock):
        raise ValueError("Expected supported game mode and pre block")
    module = model.pre
    if "forward_features_unquantized" in module.__dict__:
        raise ValueError("Pre block already has a forward override")
    calls = {"pre_fused": 0}

    def scoped(self, features):
        if tuple(features.shape) != _SHAPES[height]:
            raise ValueError("Pre input shape differs from selected game mode")
        if current_arithmetic_backend() != "triton":
            raise ValueError("Pre fusion requires the pinned Triton arithmetic backend")
        output = fused_forward_features_unquantized(self, features)
        calls["pre_fused"] += 1
        return output

    replacement = MethodType(scoped, module)
    module.forward_features_unquantized = replacement
    try:
        yield calls
    finally:
        if module.forward_features_unquantized is not replacement:
            raise RuntimeError("Pre override changed during its scope")
        del module.forward_features_unquantized
