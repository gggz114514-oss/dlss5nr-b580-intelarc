"""Offline half-bit table enumeration for the original cubic arithmetic."""
import triton
import triton.language as tl
from nr_backend.triton_cubic_fp8 import _half_fma_value
from nr_backend.triton_fp8 import _round_fp8_half


@triton.jit
def build(Y, R, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    x = i.to(tl.uint16).to(tl.float16, bitcast=True)
    t = tl.minimum(tl.maximum(x.to(tl.float32), -4.0), 4.0).to(tl.float16)
    p = _half_fma_value(-tl.abs(t).to(tl.float32),
                        tl.full((), .055908203125, tl.float32),
                        tl.full((), .447265625, tl.float32))
    v = _half_fma_value(t, p, tl.full((), .89453125, tl.float32))
    activated = (x.to(tl.float32) * v.to(tl.float32)).to(tl.float16)
    rounded = _round_fp8_half(activated)
    tl.store(Y + i, activated.to(tl.int16, bitcast=True))
    tl.store(R + i, rounded.to(tl.int16, bitcast=True))
