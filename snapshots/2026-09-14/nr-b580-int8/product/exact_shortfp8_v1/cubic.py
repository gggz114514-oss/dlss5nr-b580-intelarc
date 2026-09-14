"""Exact cubic activation and the existing E4M3 boundary in one XPU call.

Retains every original half rounding boundary. This changes execution only;
it does not change the model or introduce a new quantization scheme.
"""
import torch
import triton
import triton.language as tl
from .fp8 import _round_fp8_half


@triton.jit
def _half_fma_value(av,bv,cv):
    av=av.to(tl.float32);bv=bv.to(tl.float32);cv=cv.to(tl.float32)
    product=av*bv
    summed=product+cv
    virtual=summed-product
    residual=(product-(summed-virtual))+(cv-virtual)
    rounded=summed.to(tl.float16)
    bits=rounded.to(tl.uint16,bitcast=True).to(tl.int32)
    negative=(bits&0x8000)!=0
    virtual_overflow=tl.where(negative,-65536.0,65536.0)
    rounded_float=tl.where((bits&0x7fff)==0x7c00,virtual_overflow,rounded.to(tl.float32))
    adjacent_bits=bits+tl.where((summed>rounded_float)!=negative,1,-1)
    adjacent=adjacent_bits.to(tl.uint16).to(tl.float16,bitcast=True)
    adjacent_float=tl.where((adjacent_bits&0x7fff)==0x7c00,virtual_overflow,adjacent.to(tl.float32))
    midpoint=(rounded_float+adjacent_float)*0.5
    correction=(summed==midpoint)&(residual!=0)&((summed>rounded_float)==(residual>0))
    return tl.where(correction,adjacent,rounded)


@triton.jit
def _direct(X,Y,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+i,i<N,other=0)
    t=tl.minimum(tl.maximum(x.to(tl.float32),-4.0),4.0).to(tl.float16)
    p=_half_fma_value(-tl.abs(t).to(tl.float32),tl.full((),0.055908203125,tl.float32),tl.full((),0.447265625,tl.float32))
    v=_half_fma_value(t,p,tl.full((),0.89453125,tl.float32))
    activated=(x.to(tl.float32)*v.to(tl.float32)).to(tl.float16)
    result=_round_fp8_half(activated)
    tl.store(Y+i,result,i<N)


def direct_cubic_fp8(x):
    if x.device.type!='xpu':raise ValueError('Cubic FP8 fusion requires XPU')
    source=x.half().contiguous();out=torch.empty_like(source)
    if source.numel():
        _direct[(triton.cdiv(source.numel(),512),)](source,out,source.numel(),512,enable_fp_fusion=False)
    return out
