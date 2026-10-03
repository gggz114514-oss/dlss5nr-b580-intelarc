"""Shorter E4M3 encode/decode from half, including signed zeros and NaN policy.

Below 1/64, adding half bias 2 gives spacing 1/512. Rounding to half then
subtracting the bias gives the E4M3 subnormal lattice. Near each decision
boundary the half input spacing is at least 2^-21, whereas float32 spacing
near 2 is 2^-22; the float32 addition cannot erase a decisive half-input bit.
All 65536 bit patterns must be tested against the existing integer helper.
"""
import triton
import triton.language as tl
from nr_backend.triton_fp8 import _round_fp8_half


@triton.jit
def round_half(x):
    bits = x.to(tl.uint16, bitcast=True).to(tl.int32)
    sign = bits & 0x8000
    raw = bits & 0x7fff
    mag = tl.minimum(raw, 0x5f00)
    positive = mag.to(tl.uint16).to(tl.float16, bitcast=True)
    biased = (positive.to(tl.float32) + 2.0).to(tl.float16)
    sub = (biased.to(tl.float32) - 2.0).to(tl.float16).to(tl.uint16, bitcast=True).to(tl.int32)
    normal = (mag + 63 + ((mag >> 7) & 1)) & 0x7f80
    result = tl.where(mag < 0x2400, sub, normal) | sign
    result = tl.where(raw > 0x7c00, 0x7f80 | sign, result)
    return result.to(tl.uint16).to(tl.float16, bitcast=True)


@triton.jit
def compare_kernel(X, OLD, NEW, N: tl.constexpr, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = tl.load(X + i, i < N, other=0)
    tl.store(OLD + i, _round_fp8_half(x), i < N)
    tl.store(NEW + i, round_half(x), i < N)
