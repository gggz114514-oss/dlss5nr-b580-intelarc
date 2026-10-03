"""Experimental exact finite-half FMA fusion; independent of the main backend."""
import triton
import triton.language as tl
import torch


@triton.jit
def _half_fma(A, B, C, OUT, N: tl.constexpr, SHAPE: tl.constexpr,
              AS: tl.constexpr, BS: tl.constexpr, CS: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    rest = index
    ai = tl.full((BLOCK,), 0, tl.int32)
    bi = tl.full((BLOCK,), 0, tl.int32)
    ci = tl.full((BLOCK,), 0, tl.int32)
    for axis in tl.static_range(len(SHAPE) - 1, -1, -1):
        coord = rest % SHAPE[axis]
        rest = rest // SHAPE[axis]
        ai += coord * AS[axis]
        bi += coord * BS[axis]
        ci += coord * CS[axis]
    av = tl.load(A + ai, index < N, other=0).to(tl.float32)
    bv = tl.load(B + bi, index < N, other=0).to(tl.float32)
    cv = tl.load(C + ci, index < N, other=0).to(tl.float32)
    product = av * bv
    summed = product + cv
    virtual = summed - product
    residual = (product - (summed - virtual)) + (cv - virtual)
    rounded = summed.to(tl.float16)
    bits = rounded.to(tl.uint16, bitcast=True).to(tl.int32)
    negative = (bits & 0x8000) != 0
    virtual_overflow = tl.where(negative, -65536.0, 65536.0)
    rounded_float = tl.where((bits & 0x7fff) == 0x7c00, virtual_overflow, rounded.to(tl.float32))
    adjacent_bits = bits + tl.where((summed > rounded_float) != negative, 1, -1)
    adjacent = adjacent_bits.to(tl.uint16).to(tl.float16, bitcast=True)
    adjacent_float = tl.where((adjacent_bits & 0x7fff) == 0x7c00, virtual_overflow, adjacent.to(tl.float32))
    midpoint = (rounded_float + adjacent_float) * 0.5
    correction = (summed == midpoint) & (residual != 0) & ((summed > rounded_float) == (residual > 0))
    tl.store(OUT + index, tl.where(correction, adjacent, rounded), index < N)


def fused_half_fma(a, b, c, *, block=256):
    """Match the reference's half conversion, broadcasting and signed-zero result."""
    a, b, c = torch.broadcast_tensors(a.half(),
        torch.as_tensor(b, device=a.device, dtype=torch.float16),
        torch.as_tensor(c, device=a.device, dtype=torch.float16))
    output = torch.empty(a.shape, dtype=torch.float16, device=a.device)
    if output.numel():
        _half_fma[(triton.cdiv(output.numel(), block),)](a, b, c, output, output.numel(),
            tuple(a.shape), tuple(a.stride()), tuple(b.stride()), tuple(c.stride()), block,
            enable_fp_fusion=False)
    return output
