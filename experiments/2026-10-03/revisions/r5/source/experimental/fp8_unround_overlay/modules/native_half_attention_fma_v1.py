"""Half FMA for the validated attention callers, retaining their NaN selection.

Callers supply half values or constants exactly representable in half. They must
retain normalization's _nan_left or the exponential's input-NaN bit selection.
This is not a blanket replacement for the general compensated half FMA helper.
"""
import triton
import triton.language as tl

@triton.jit
def half_fma_attention(a,b,c):
    return tl.fma(a.to(tl.float16),b.to(tl.float16),c.to(tl.float16)).to(tl.float16)
