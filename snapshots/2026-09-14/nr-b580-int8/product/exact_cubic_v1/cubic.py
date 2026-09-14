"""Native half FMA for the specific unary cubic+FP8 domain only.

Keep the clamp, two independently rounded half FMAs, final half product and
original FP8 conversion. Do not replace the generic compensated FMA helper.
"""
import triton
import triton.language as tl
from nr_backend.triton_fp8 import _round_fp8_half


@triton.jit
def cubic(x):
    t = tl.minimum(tl.maximum(x.to(tl.float32), -4.), 4.).to(tl.float16)
    p = tl.fma(-tl.abs(t), tl.full((), .055908203125, tl.float16),
               tl.full((), .447265625, tl.float16)).to(tl.float16)
    v = tl.fma(t, p, tl.full((), .89453125, tl.float16)).to(tl.float16)
    return _round_fp8_half((x.to(tl.float32) * v.to(tl.float32)).to(tl.float16))

import torch

@triton.jit
def _direct(X,Y,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+i,i<N,other=0)
    tl.store(Y+i,cubic(x),i<N)

def direct_cubic_fp8(x):
    if x.device.type != 'xpu':
        raise ValueError('Cubic FP8 requires XPU')
    source=x.half().contiguous()
    out=torch.empty_like(source)
    if source.numel():
        _direct[(triton.cdiv(source.numel(),512),)](source,out,source.numel(),512,enable_fp_fusion=False)
    return out
