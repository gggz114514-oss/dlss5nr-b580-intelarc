"""Scoped sixteen-block C256 QKV direct-write experiment for RE8 fast modes.

This sits inside the installed C64/C128/C32/post combo, delegates other calls to
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
from c256_qkv_direct_pack_one_v1 import _direct_pack


_BASE = {360: (24, 48), 480: (32, 56), 540: (40, 64),
         720: (48, 80)}
_SHIFTS = ((0, 0), (4, 4), (0, 4), (4, 0))


def _shape(height, shift):
    if height not in _BASE or shift not in _SHIFTS:
        raise ValueError("Unexpected C256 game geometry or shift")
    h, w = _BASE[height]
    sy, sx = shift
    return h + (8 if sy else 0), w + (8 if sx else 0)


_AUDIT_LOCAL = None


def direct_pack(features, module, *, height: int, shift: tuple[int, int],
                block_m: int = 16):
    if _AUDIT_LOCAL is not None and id(module) in _AUDIT_LOCAL.by_attention:
        if height != 720 or tuple(shift) != tuple(_AUDIT_LOCAL.by_attention[id(module)][0].shift):
            raise RuntimeError("Owned C256 QKV requires its fixed720 site shift")
        result = _AUDIT_LOCAL.qkv_override(features, module)
        if result is not None:
            return result
    h, w = _shape(height, shift)
    if h % 8 or w % 8:
        raise ValueError("C256 QKV requires complete 8x8 windows")
    if (features.device.type != "xpu" or features.dtype != torch.float16 or
            not features.is_contiguous() or tuple(features.shape) != (h, w, 256) or
            block_m not in (16, 32) or
            module.heads != 8 or tuple(module.qkv.shape) != (256, 768) or
            module.qkv.device != features.device or
            module.qkv.dtype != torch.float16 or not module.qkv.is_contiguous() or
            tuple(module.scale.shape) != (8,) or module.scale.device != features.device or
            module.scale.dtype != torch.float16 or not module.scale.is_contiguous() or
            tuple(module.pixel_inverse.shape) != (64,) or
            module.pixel_inverse.device != features.device or
            module.pixel_inverse.dtype != torch.int64 or
            not module.pixel_inverse.is_contiguous()):
        raise ValueError("Unexpected installed C256 QKV/attention boundary")
    output_shape = (8, h // 8, w // 8, 64, 32)
    q, k, v = [torch.empty(output_shape, dtype=torch.float16, device=features.device)
               for _ in range(3)]
    _direct_pack[(triton.cdiv(h * w, block_m), module.heads * 3)](
        features, module.qkv, module.scale, module.pixel_inverse, q, k, v,
        h * w, w, (h // 8) * (w // 8), block_m, 32, 32,
        ROUND_QKV="c256" not in ENABLED,
        num_warps=4, enable_fp_fusion=False)
    return q, k, v


@contextmanager
def installed(window_blocks: WindowBlocks, model, *, height: int,
              block_m: int = 16):
    if (not isinstance(window_blocks, WindowBlocks) or height not in _BASE or
            block_m not in (16, 32)):
        raise ValueError("Expected installed WindowBlocks and known game mode")
    targets = {}
    for index in range(8):
        for prefix, block in (("encoder", model.encoder[3][index]),
                              ("decoder", model.decoder[0][index])):
            body = block.body if prefix == "decoder" and index == 0 else block
            expected_shift = _SHIFTS[index % 4]
            if (not isinstance(body, blocks.MultiHeadSwinBlock) or
                    body.channels != 256 or body.attention.heads != 8 or
                    body.window_shift != expected_shift):
                raise ValueError(f"Unexpected {prefix} C256 block {index}")
            name = f"{prefix}[{index}]"
            shape = _shape(height, body.window_shift)
            targets[id(body.attention)] = (body.attention, name,
                                            body.window_shift, shape)
    if len(targets) != 16:
        raise ValueError("Expected sixteen unique C256 attention modules")
    had_instance = "windows" in window_blocks.__dict__
    original = window_blocks.windows
    calls = {"direct_pack": 0,
             "by_block": {row[1]: 0 for row in targets.values()}}

    def replacement(self, module, features):
        row = targets.get(id(module))
        if row is None:
            return original(module, features)
        target, name, shift, shape = row
        if module is not target or tuple(features.shape) != (*shape, 256):
            raise ValueError(f"Unexpected {name} padded C256 geometry")
        h, w = shape
        q, k, v = direct_pack(round_multihead(256,features), module,
                              height=height, shift=shift, block_m=block_m)
        for _ in range(2):
            record_arithmetic_dispatch("attention_normalize_c32")
        for _ in range(3):
            record_arithmetic_dispatch("fp8")
        result, self.last_attention_kernel, self.last_attention_selection = attend(
            q, k, v, module.bias, round_weights="c256" not in ENABLED)
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
            raise RuntimeError("C256 producer replaced during its scope")
        if had_instance:
            window_blocks.windows = original
        else:
            del window_blocks.windows
