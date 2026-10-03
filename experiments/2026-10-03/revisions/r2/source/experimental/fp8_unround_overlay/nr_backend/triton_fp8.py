"""Single-pass SATFINITE E4M3 encode/decode, with explicit FP16 bit rounding.

No model or weight quantization change: the model already has this exact FP8
boundary. This candidate only combines its clamp, conversions and zero-sign fix.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _round_fp8_half(x):
    bits=x.to(tl.uint16,bitcast=True).to(tl.int32)
    sign=bits&0x8000
    raw=bits&0x7fff
    mag=tl.minimum(raw,0x5f00)  # 448, maximum finite E4M3FN magnitude.
    exponent=mag>>10
    sig=(mag&1023)+tl.where(exponent!=0,1024,0)
    shift=tl.maximum(16-tl.maximum(exponent,1),1)
    quotient=sig>>shift
    remainder=sig-(quotient<<shift)
    midpoint=1<<(shift-1)
    q=quotient+((remainder>midpoint)|((remainder==midpoint)&((quotient&1)!=0))).to(tl.int32)
    sub=(q.to(tl.float32)*0.001953125).to(tl.float16).to(tl.uint16,bitcast=True).to(tl.int32)
    normal=(mag+63+((mag>>7)&1))&0x7f80
    result=tl.where(mag<0x2400,sub,normal)|sign
    # Canonical NaNs follow the tested encode/decode behavior. Full NR native
    # fidelity remains scoped separately to its established finite contract.
    result=tl.where(raw>0x7c00,0x7f80|sign,result)
    return result.to(tl.uint16).to(tl.float16,bitcast=True)


@triton.jit
def _kernel(X,Y,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+i,i<N,other=0)
    tl.store(Y+i,_round_fp8_half(x),i<N)


def quantize_fp8(x):
    if x.device.type!='xpu':raise ValueError('Fused FP8 conversion requires XPU')
    source=x.half().contiguous()
    output=torch.empty_like(source)
    if source.numel():
        _kernel[(triton.cdiv(source.numel(),512),)](source,output,source.numel(),512,enable_fp_fusion=False)
    return output
