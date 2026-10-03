"""C32 expansion/cubic/contraction/raw-input skip in one streaming FP16 kernel.

The original C32 skip uses raw projected half, not FP8 projected half. Retain
that distinction and ordered four K32 contraction sums. No C128 image allocation.
"""
from contextlib import contextmanager
import torch
import triton
import triton.language as tl
from nr_backend.triton_fp8 import _round_fp8_half
from nr_backend.execution import current_arithmetic_backend,record_arithmetic_dispatch
from nr_backend.pre_mlp import C32MLP,PreMLP
from fused_branched_mlp_v1 import _cubic


@triton.jit
def _kernel(X,EXPAND,CONTRACT,SCALE,OUT,M:tl.constexpr,BM:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    lane=tl.arange(0,32)
    raw=tl.load(X+row[:,None]*32+lane[None,:],row[:,None]<M,other=0)
    x=_round_fp8_half(raw)
    contracted=tl.full((BM,32),0.,tl.float32)
    for part in range(4):
        w=tl.load(EXPAND+lane[:,None]*128+part*32+lane[None,:])
        expanded=tl.dot(x,w,out_dtype=tl.float32)
        hidden=_cubic(expanded.to(tl.float16))
        weight=tl.load(CONTRACT+(part*32+lane[:,None])*32+lane[None,:])
        contracted=tl.dot(hidden,weight,contracted,out_dtype=tl.float32)
    scale=tl.load(SCALE+lane)
    initial=(raw.to(tl.float32)*scale[None,:].to(tl.float32)).to(tl.float16)
    value=(contracted+initial.to(tl.float32)).to(tl.float16)
    tl.store(OUT+row[:,None]*32+lane[None,:],value,row[:,None]<M)


def forward(features,expansion,contraction,skip_scale,*,bm=16,warps=4,stages=1):
    if features.device.type!='xpu' or features.shape[-1]!=32 or features.numel()==0:
        raise ValueError('Expected projected XPU features with C32')
    if bm not in (16,32) or warps not in (4,8) or stages not in (1,2):
        raise ValueError('Unsupported C32 launch configuration')
    for value,shape in ((expansion,(32,128)),(contraction,(128,32)),(skip_scale,(32,))):
        if value.shape!=shape or value.device!=features.device or value.dtype!=torch.float16 or not value.is_contiguous():
            raise ValueError('Invalid C32 MLP weights')
    x=features.half().contiguous();out=torch.empty_like(x);m=x.numel()//32
    kernel=_kernel[(triton.cdiv(m,bm),)](x,expansion,contraction,skip_scale,out,m,bm,
        num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return out,kernel


class FusedC32:
    """Instance-level method overrides survive the graph body's class scheduling patch."""
    def __init__(self,model,provider,*,bm=16,warps=4,stages=1):
        self.provider=provider
        self.modules=[module for module in model.modules() if type(module) in (C32MLP,PreMLP)]
        self.options=dict(bm=bm,warps=warps,stages=stages)
        self.calls={}

    def apply(self,module,original,features):
        if current_arithmetic_backend()!='triton' or features.device.type!='xpu' or self.provider.mode!='fp16_xmx':
            return original(features)
        result,_=forward(features,module.expansion,module.contraction,module.skip_scale,**self.options)
        chunks=triton.cdiv(features.numel()//32,32768)
        for _ in range(chunks):
            record_arithmetic_dispatch('fp8');record_arithmetic_dispatch('cubic_fp8')
            record_arithmetic_dispatch('dense');record_arithmetic_dispatch('dense')
        # Counts above describe original logical work, not physical GPU calls.
        key=str(features.numel()//32)
        self.calls[key]=self.calls.get(key,0)+1
        return result

    @contextmanager
    def installed(self):
        previous=[]
        for module in self.modules:
            if 'forward_unquantized' in module.__dict__:
                raise ValueError('An instance already owns a C32 override')
        try:
            for module in self.modules:
                original=module.forward_unquantized
                replacement=lambda features,module=module,original=original:self.apply(module,original,features)
                module.forward_unquantized=replacement
                previous.append((module,replacement))
            yield self
        finally:
            for module,replacement in reversed(previous):
                assert module.forward_unquantized is replacement
                del module.forward_unquantized
