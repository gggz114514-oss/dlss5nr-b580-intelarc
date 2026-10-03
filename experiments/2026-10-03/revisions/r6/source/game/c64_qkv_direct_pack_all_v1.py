"""Isolated eight-block C64 producer experiment for supported game modes.

Reuses the byte-gated v2 arithmetic kernel, restricting replacement to the
four C64 encoder blocks and their four corresponding decoder blocks. Each
installation is tied to one geometry and a new graph capture.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import MethodType

import torch
import triton

from nr_backend.execution import record_arithmetic_dispatch
from nr_backend.multihead_block import MultiHeadSwinBlock
from window_block_attention_v3 import forward as attend
from window_blocks_v3 import WindowBlocks
import nr_backend.multihead_block as blocks
from nr_backend.unround_policy import ENABLED, round_multihead
from c64_qkv_direct_pack_v2 import _qkv_direct_pack


# Padded QKV H×W in encoder/decoder block order, derived from current G code.
_SIZES = {
    360: ((96, 192), (104, 200), (96, 200), (104, 192)),
    480: ((128, 224), (136, 232), (128, 232), (136, 224)),
    540: ((160, 256), (168, 264), (160, 264), (168, 256)),
    720: ((192, 320), (200, 328), (192, 328), (200, 320)),
}
_SHIFTS = ((0, 0), (4, 4), (0, 4), (4, 0))


def _padded_shape(height: int, shift: tuple[int, int]) -> tuple[int, int]:
    if height not in _SIZES or shift not in _SHIFTS:
        raise ValueError("Unexpected C64 game geometry or shift")
    base_h, base_w = _SIZES[height][0]
    sy, sx = shift
    return base_h + (8 if sy else 0), base_w + (8 if sx else 0)


_AUDIT_LOCAL = None


def direct_pack(features: torch.Tensor, module) -> tuple[torch.Tensor, ...]:
    if _AUDIT_LOCAL is not None:
        result = _AUDIT_LOCAL.qkv_override(features, module)
        if result is not None:
            return result
    if (features.device.type != "xpu" or features.dtype != torch.float16 or
            not features.is_contiguous() or features.ndim != 3 or
            features.shape[-1] != 64 or
            tuple(features.shape[:2]) not in {shape for sizes in _SIZES.values()
                                              for shape in sizes} or
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
    if h % 8 or w % 8:
        raise ValueError("C64 QKV requires complete 8x8 windows")
    shape = (2, h // 8, w // 8, 64, 32)
    q, k, v = [torch.empty(shape, dtype=torch.float16, device=features.device)
               for _ in range(3)]
    _qkv_direct_pack[(triton.cdiv(h * w, 16), module.heads * 3)](
        features, module.qkv, module.scale, module.pixel_inverse, q, k, v,
        h * w, w, (h // 8) * (w // 8), 16, 32, 32,
        ROUND_QKV="c64" not in ENABLED,
        num_warps=4, enable_fp_fusion=False)
    return q, k, v


@contextmanager
def installed(window_blocks: WindowBlocks, model, *, height: int):
    if (not isinstance(window_blocks, WindowBlocks) or
            "windows" in window_blocks.__dict__ or height not in _SIZES):
        raise ValueError("Expected one unmodified WindowBlocks and known mode")
    targets = {}
    for index, expected_shape in enumerate(_SIZES[height]):
        for prefix, block in (("encoder", model.encoder[1][index]),
                              ("decoder", model.decoder[2][index])):
            body = block.body if prefix == "decoder" and index == 0 else block
            if (type(body) is not MultiHeadSwinBlock or
                    body.window_shift != _SHIFTS[index] or
                    body.channels != 64 or body.attention.heads != 2):
                raise ValueError(f"Unexpected {prefix} C64 block {index}")
            shape = _padded_shape(height, body.window_shift)
            if shape != expected_shape:
                raise ValueError(f"Unexpected {prefix} C64 padded shape {index}")
            name = f"{prefix}[{index}]"
            targets[id(body.attention)] = (body.attention, name, shape)
    if len(targets) != 8:
        raise ValueError("Expected eight unique C64 attention modules")
    original = type(window_blocks).windows
    calls = {"direct_pack": 0, "by_block": {row[1]: 0 for row in targets.values()}}

    def replacement(self, module, features):
        row = targets.get(id(module))
        if row is None:
            return original(self, module, features)
        target, name, shape = row
        if module is not target or tuple(features.shape) != (*shape, 64):
            raise ValueError(f"Unexpected {name} padded C64 geometry")
        h, w = shape
        q, k, v = direct_pack(round_multihead(64,features), module)
        for _ in range(2):
            record_arithmetic_dispatch("attention_normalize_c32")
        for _ in range(3):
            record_arithmetic_dispatch("fp8")
        result, self.last_attention_kernel, self.last_attention_selection = attend(
            q, k, v, module.bias, round_weights="c64" not in ENABLED)
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
            raise RuntimeError("C64 producer replaced during its scope")
        del window_blocks.windows
