"""Scoped twelve-block C128 QKV direct-write experiment for RE8 fast modes.

This sits inside the installed C64/C32/post combo, delegates non-C128 calls to
its bound method and restores that method before the outer combo closes.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch
import triton

from nr_backend.execution import record_arithmetic_dispatch
from window_block_attention_v3 import forward as attend
from window_blocks_v3 import WindowBlocks
import nr_backend.multihead_block as blocks
from nr_backend.unround_policy import ENABLED, round_multihead
from c128_qkv_direct_pack_one_v1 import _direct_pack


_BASE = {360: (48, 96), 480: (64, 112), 540: (80, 128),
         720: (96, 160)}
_SHIFTS = ((0, 0), (4, 4), (0, 4), (4, 0))


def _shape(height, shift):
    if height not in _BASE or shift not in _SHIFTS:
        raise ValueError("Unexpected C128 game geometry or shift")
    h, w = _BASE[height]
    sy, sx = shift
    return h + (8 if sy else 0), w + (8 if sx else 0)


_AUDIT_LOCAL = None


def direct_pack(features, module, *, height: int, shift: tuple[int, int]):
    if _AUDIT_LOCAL is not None and id(module) in _AUDIT_LOCAL.by_attention:
        if height != 720 or tuple(shift) != tuple(_AUDIT_LOCAL.by_attention[id(module)][0].shift):
            raise RuntimeError("Owned C128 QKV requires its fixed720 site shift")
        result = _AUDIT_LOCAL.qkv_override(features, module)
        if result is not None:
            return result
    h, w = _shape(height, shift)
    if h % 8 or w % 8:
        raise ValueError("C128 QKV requires complete 8x8 windows")
    if (features.device.type != "xpu" or features.dtype != torch.float16 or
            not features.is_contiguous() or tuple(features.shape) != (h, w, 128) or
            module.heads != 4 or tuple(module.qkv.shape) != (128, 384) or
            module.qkv.device != features.device or
            module.qkv.dtype != torch.float16 or not module.qkv.is_contiguous() or
            tuple(module.scale.shape) != (4,) or module.scale.device != features.device or
            module.scale.dtype != torch.float16 or not module.scale.is_contiguous() or
            tuple(module.pixel_inverse.shape) != (64,) or
            module.pixel_inverse.device != features.device or
            module.pixel_inverse.dtype != torch.int64 or
            not module.pixel_inverse.is_contiguous()):
        raise ValueError("Unexpected installed C128 QKV/attention boundary")
    output_shape = (4, h // 8, w // 8, 64, 32)
    q, k, v = [torch.empty(output_shape, dtype=torch.float16, device=features.device)
               for _ in range(3)]
    _direct_pack[(triton.cdiv(h * w, 16), module.heads * 3)](
        features, module.qkv, module.scale, module.pixel_inverse, q, k, v,
        h * w, w, (h // 8) * (w // 8), 16, 32, 32,
        ROUND_QKV="c128" not in ENABLED,
        num_warps=4, enable_fp_fusion=False)
    return q, k, v


@contextmanager
def installed(window_blocks: WindowBlocks, model, *, height: int):
    if not isinstance(window_blocks, WindowBlocks) or height not in _BASE:
        raise ValueError("Expected installed WindowBlocks and known game mode")
    targets = {}
    for index in range(6):
        for prefix, block in (("encoder", model.encoder[2][index]),
                              ("decoder", model.decoder[1][index])):
            body = block.body if prefix == "decoder" and index == 0 else block
            expected_shift = _SHIFTS[(index + (2 if prefix == "decoder" else 0)) % 4]
            if (not isinstance(body, blocks.MultiHeadSwinBlock) or
                    body.channels != 128 or body.attention.heads != 4 or
                    body.window_shift != expected_shift):
                raise ValueError(f"Unexpected {prefix} C128 block {index}")
            name = f"{prefix}[{index}]"
            shape = _shape(height, body.window_shift)
            targets[id(body.attention)] = (body.attention, name,
                                            body.window_shift, shape)
    if len(targets) != 12:
        raise ValueError("Expected twelve unique C128 attention modules")
    had_instance = "windows" in window_blocks.__dict__
    original = window_blocks.windows
    calls = {"direct_pack": 0,
             "by_block": {row[1]: 0 for row in targets.values()}}

    def replacement(self, module, features):
        row = targets.get(id(module))
        if row is None:
            return original(module, features)
        target, name, shift, shape = row
        if module is not target or tuple(features.shape) != (*shape, 128):
            raise ValueError(f"Unexpected {name} padded C128 geometry")
        h, w = shape
        q, k, v = direct_pack(round_multihead(128,features), module,
                              height=height, shift=shift)
        for _ in range(2):
            record_arithmetic_dispatch("attention_normalize_c32")
        for _ in range(3):
            record_arithmetic_dispatch("fp8")
        result, self.last_attention_kernel, self.last_attention_selection = attend(
            q, k, v, module.bias, round_weights="c128" not in ENABLED)
        for _ in range(module.heads * ((h // 8 * (w // 8) + 1023) // 1024)):
            for key in ("batched", "attention_exp_swin", "attention_weights", "batched"):
                record_arithmetic_dispatch(key)
        for key, n in (("multi", 1), ("batched_heads", module.heads), ("qkv_pack", 1)):
            self.layout.calls[key] = self.layout.calls.get(key, 0) + n
        calls["direct_pack"] += 1
        calls["by_block"][name] += 1
        return result

    bound = MethodType(replacement, window_blocks)
    window_blocks.windows = bound
    try:
        yield calls
    finally:
        if window_blocks.__dict__.get("windows") is not bound:
            raise RuntimeError("C128 producer replaced during its scope")
        if had_instance:
            window_blocks.windows = original
        else:
            del window_blocks.windows
