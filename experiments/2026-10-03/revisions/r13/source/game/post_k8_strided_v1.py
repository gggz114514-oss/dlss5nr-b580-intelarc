"""Isolated post-head K8 stride reader for the installed RE8 Triton backend.

Only a single ResetPostBlock.forward_head instance is scoped. Its original
Python function and every other arithmetic call retain their existing code.
The kernel changes the activation address calculation in `_tiled` and keeps
the K8 shared-exponent/INT32/half-rounding expressions in their original order.
"""
from __future__ import annotations

from contextlib import contextmanager
from types import FunctionType, MethodType

import torch
import triton
import triton.language as tl

from nr_backend.post import ResetPostBlock
from nr_backend.triton_math import _exponent, _scaled_integer_to_half


_INTERNAL_SHAPES = {(384, 768), (512, 896), (640, 1024), (768, 1280)}


@triton.jit
def _post_k8_strided(A, WEIGHTS, OUT, M: tl.constexpr, OUTPUT_WIDTH: tl.constexpr,
                     ROW_STRIDE: tl.constexpr, OUTPUT_CHANNELS: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    group = tl.arange(0, 8)
    valid = (rows[:, None] < M) & (cols[None, :] < OUTPUT_CHANNELS)
    offset = rows[:, None] * OUTPUT_CHANNELS + cols[None, :]
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(0, 32, 8):
        # A already points at the cropped view's storage offset. The row pitch
        # still includes the eight border columns in the K16 result.
        address = (rows // OUTPUT_WIDTH)[:, None] * ROW_STRIDE + \
                  (rows % OUTPUT_WIDTH)[:, None] * 32 + start + group[None, :]
        av = tl.load(A + address, rows[:, None] < M, other=0).to(tl.float32)
        wv = tl.load(WEIGHTS + (start + group[:, None]) * 8 + cols[None, :],
                     cols[None, :] < OUTPUT_CHANNELS, other=0).to(tl.float32)
        ea, ew = _exponent(av), _exponent(wv)
        exponent = tl.minimum(tl.maximum(
            tl.maximum(tl.max(ea[:, :, None] + ew[None, :, :], 1),
                       _exponent(acc)), -50), 50)
        scale = ((127 + 24 - exponent) << 23).to(tl.float32, bitcast=True)
        product = av[:, :, None] * wv[None, :, :]
        summed = tl.sum((product * scale[:, None, :]).to(tl.int32), 1) + \
                 (acc * scale).to(tl.int32)
        acc = _scaled_integer_to_half(summed, exponent - 24).to(tl.float32)
    tl.store(OUT + offset, acc.to(tl.float16), valid)


def strided_head_dot(a: torch.Tensor, weight: torch.Tensor,
                     output_size: tuple[int, int] | None = None,
                     output_channels: int = 8) -> torch.Tensor:
    """Read the K16 crop with its native pitch; optionally prune unused outputs."""
    if (a.device.type != "xpu" or a.device != weight.device or
            a.dtype != torch.float16 or weight.dtype != torch.float16 or
            a.ndim != 3 or tuple(a.shape[:2]) not in _INTERNAL_SHAPES or
            a.shape[2] != 32 or tuple(weight.shape) != (32, 8) or
            tuple(weight.stride()) != (8, 1)):
        raise ValueError("Expected an RE8 XPU post-head K8 operand and weight")
    height, width = a.shape[:2]
    if (tuple(a.stride()) != ((width + 8) * 32, 32, 1) or
            a.storage_offset() != (4 * (width + 8) + 4) * 32):
        raise ValueError("Expected the cropped, non-contiguous post K16 result")
    out_h, out_w = (height, width) if output_size is None else tuple(output_size)
    if not (0 < out_h <= height and 0 < out_w <= width):
        raise ValueError("The visible post-head output must fit its source canvas")
    if output_channels not in (4, 8):
        raise ValueError("Only the four consumed or all eight head channels are supported")
    output = torch.empty((out_h, out_w, output_channels),
                         dtype=torch.float16, device=a.device)
    _post_k8_strided[(triton.cdiv(out_h * out_w, 4), 1)](
        a, weight, output, out_h * out_w, out_w, a.stride(0),
        output_channels, 4, output_channels,
        num_warps=1, enable_fp_fusion=False)
    return output


@contextmanager
def installed(module: ResetPostBlock,
              output_size: tuple[int, int] | None = None,
              output_channels: int = 8):
    """Replace only this post instance's K8 dispatch; restore on exit.

    The copied globals dictionary leaves the installed module's `dot` binding
    untouched, including for concurrent post instances and graph capture.
    `calls` counts eager/capture dispatch; replay executes the captured kernel.
    """
    if not isinstance(module, ResetPostBlock) or "forward_head" in module.__dict__:
        raise ValueError("Expected one unmodified post block")
    original = type(module).forward_head
    original_dot = original.__globals__["dot"]
    if original.__module__ != "nr_backend.post":
        raise ValueError("Unexpected post.forward_head implementation")
    calls = {"k8": 0}

    def scoped_dot(a, weight, *, chunk_k, rows_per_batch=2048, initial=None):
        if weight is not module.head_weight:
            return original_dot(a, weight, chunk_k=chunk_k,
                                rows_per_batch=rows_per_batch, initial=initial)
        if chunk_k != 8 or initial is not None or rows_per_batch != 2048:
            raise ValueError("Unexpected post.head_weight dispatch")
        result = strided_head_dot(a, weight, output_size=output_size,
                                  output_channels=output_channels)
        calls["k8"] += 1
        return result

    namespace = original.__globals__.copy()
    namespace["dot"] = scoped_dot
    replacement_function = FunctionType(original.__code__, namespace,
                                        original.__name__, original.__defaults__,
                                        original.__closure__)
    replacement_function.__kwdefaults__ = original.__kwdefaults__
    replacement = MethodType(replacement_function, module)
    module.forward_head = replacement
    try:
        yield calls
    finally:
        if module.__dict__.get("forward_head") is not replacement:
            raise RuntimeError("Post forward_head was replaced during its scope")
        del module.forward_head
