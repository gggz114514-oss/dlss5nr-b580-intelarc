"""Fuse the original Swin/ViT score transform, preserving every half boundary."""
import torch
import triton
import triton.language as tl
from .triton_cubic_fp8 import _half_fma_value


@triton.jit
def _kernel(X, Y, N: tl.constexpr, VIT: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, other=0)
    if VIT:
        affine = _half_fma_value(x, tl.full((), 0.08953857421875, tl.float32), tl.full((), 1.708984375, tl.float32))
        affine = tl.minimum(tl.maximum(affine.to(tl.float32), 1.439453125), 1.9775390625).to(tl.float16)
        bits = affine.to(tl.uint16, bitcast=True).to(tl.int32)
        # The current XPU half FMA quiets a NaN while retaining its payload;
        # clamp retains that NaN. Integer selection preserves those exact bits.
        input_bits = x.to(tl.uint16, bitcast=True).to(tl.int32)
        bits = tl.where((input_bits & 0x7fff) > 0x7c00, input_bits | 0x200, bits)
        result = ((bits << 4) + 0x4000) & 0xffff
    else:
        affine = _half_fma_value(x, tl.full((), 0.044921875, tl.float32), tl.full((), 1.30078125, tl.float32))
        affine = tl.minimum(tl.maximum(affine.to(tl.float32), 1.03125), 1.5693359375).to(tl.float16)
        bits = affine.to(tl.uint16, bitcast=True).to(tl.int32)
        # The current XPU half FMA quiets a NaN while retaining its payload;
        # clamp retains that NaN. Integer selection preserves those exact bits.
        input_bits = x.to(tl.uint16, bitcast=True).to(tl.int32)
        bits = tl.where((input_bits & 0x7fff) > 0x7c00, input_bits | 0x200, bits)
        result = ((bits << 5) + 0x8000) & 0xffff
    tl.store(Y + i, result.to(tl.uint16).to(tl.float16, bitcast=True), i < N)


def exponential(x, *, vit=False):
    if x.device.type != 'xpu':
        raise ValueError('Attention exponential fusion requires XPU')
    source = x.half().contiguous()
    result = torch.empty_like(source)
    if source.numel():
        _kernel[(triton.cdiv(source.numel(), 512),)](source, result, source.numel(), vit, 512, enable_fp_fusion=False)
    return result
