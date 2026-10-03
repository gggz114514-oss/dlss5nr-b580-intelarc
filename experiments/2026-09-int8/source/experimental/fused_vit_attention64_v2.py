"""Fuse ViT64 attention with native-half exponent FMA, retaining every boundary.

This is the model's half-bit exponential, not softmax. Keep half score rounding,
FP8 exponential before value dot, two ordered K32 FP32 accumulations, half
numerator, original half denominator reduction, reciprocal and final half/FP8.
No padded-key or multi-chunk contract is implemented by this bounded kernel.
"""
import torch
import triton
import triton.language as tl
from native_half_attention_fma_v1 import half_fma_attention as _half_fma_value
from nr_backend.triton_attention_weights import _add_half,reciprocal_half_clamped
from nr_backend.triton_attention_normalize import _nan_left
from nr_backend.triton_fp8 import _round_fp8_half
from fused_swin_core_v2 import _halves,_sum4

@triton.jit
def _exponential(x):
    affine=_half_fma_value(x,tl.full((),.08953857421875,tl.float32),tl.full((),1.708984375,tl.float32))
    affine=tl.minimum(tl.maximum(affine.to(tl.float32),1.439453125),1.9775390625).to(tl.float16)
    bits=affine.to(tl.uint16,bitcast=True).to(tl.int32)
    inp=x.to(tl.uint16,bitcast=True).to(tl.int32)
    bits=tl.where((inp&0x7fff)>0x7c00,inp|0x200,bits)
    return (((bits<<4)+0x4000)&0xffff).to(tl.uint16).to(tl.float16,bitcast=True)

@triton.jit
def _denominator(e0,e1,BM:tl.constexpr):
    a0,a1=_halves(e0,BM,32);a2,a3=_halves(e1,BM,32)
    v0,v8=_halves(a0,BM,16);v16,v24=_halves(a1,BM,16)
    v32,v40=_halves(a2,BM,16);v48,v56=_halves(a3,BM,16)
    partial=_add_half(_add_half(_add_half(_add_half(v0,v8),_add_half(v16,v24)),_add_half(v32,v40)),_add_half(v48,v56))
    even,odd=tl.split(tl.reshape(partial,(BM,4,2)))
    return _add_half(_sum4(even,BM),_sum4(odd,BM))

@triton.jit
def _kernel(Q,K,V,OUT,SCORE,EXP,NUM,DEN,RCP,
            QH:tl.constexpr,QR:tl.constexpr,QD:tl.constexpr,
            KH:tl.constexpr,KR:tl.constexpr,KD:tl.constexpr,
            VH:tl.constexpr,VR:tl.constexpr,VD:tl.constexpr,
            BM:tl.constexpr,DEBUG:tl.constexpr):
    head=tl.program_id(1);row=tl.program_id(0)*BM+tl.arange(0,BM)
    lane=tl.arange(0,32)
    q=tl.load(Q+head*QH+row[:,None]*QR+lane[None,:]*QD)
    k0=tl.load(K+head*KH+lane[None,:]*KR+lane[:,None]*KD)
    k1=tl.load(K+head*KH+(lane[None,:]+32)*KR+lane[:,None]*KD)
    s0=tl.dot(q,k0,out_dtype=tl.float32).to(tl.float16)
    s1=tl.dot(q,k1,out_dtype=tl.float32).to(tl.float16)
    e0=_exponential(s0);e1=_exponential(s1)
    v0=tl.load(V+head*VH+lane[:,None]*VR+lane[None,:]*VD)
    v1=tl.load(V+head*VH+(lane[:,None]+32)*VR+lane[None,:]*VD)
    numerator=tl.dot(_round_fp8_half(e0),v0,out_dtype=tl.float32)
    numerator=tl.dot(_round_fp8_half(e1),v1,numerator,out_dtype=tl.float32).to(tl.float16)
    denominator=_denominator(e0,e1,BM)
    reciprocal=reciprocal_half_clamped(denominator)
    product=_nan_left((numerator.to(tl.float32)*reciprocal[:,None].to(tl.float32)).to(tl.float16),numerator,reciprocal[:,None])
    dest=head*2048+row[:,None]*32+lane[None,:]
    tl.store(OUT+dest,_round_fp8_half(product))
    if DEBUG:
        offset=head*4096+row[:,None]*64+lane[None,:]
        tl.store(SCORE+offset,s0);tl.store(SCORE+offset+32,s1)
        tl.store(EXP+offset,e0);tl.store(EXP+offset+32,e1)
        tl.store(NUM+dest,numerator)
        tl.store(DEN+head*64+row,denominator)
        tl.store(RCP+head*64+row,reciprocal)

def forward(query,key,value,*,bm=16,warps=4,stages=1,debug=False):
    if query.ndim!=3 or query.shape!=key.shape or query.shape!=value.shape or query.shape[-2:]!=(64,32):
        raise ValueError('Expected matching heads by 64 by 32 tensors')
    for t in (query,key,value):
        if t.device.type!='xpu' or t.device!=query.device or t.dtype!=torch.float16 or t.numel()==0 or min(t.stride())<=0:
            raise ValueError('Expected nonempty same-device half XPU tensors with positive strides')
    if bm not in (16,32,64) or warps not in (4,8) or stages not in (1,2):
        raise ValueError('Unsupported ViT attention launch')
    h=query.shape[0];out=torch.empty((h,64,32),device=query.device,dtype=query.dtype)
    extras={} if not debug else {name:torch.empty(shape,device=query.device,dtype=query.dtype) for name,shape in [
        ('scores',(h,64,64)),('exponential',(h,64,64)),('numerator',(h,64,32)),('denominator',(h,64,1)),('reciprocal',(h,64,1))]}
    ptrs=[extras.get(name,out) for name in ('scores','exponential','numerator','denominator','reciprocal')]
    kernel=_kernel[(64//bm,h)](query,key,value,out,*ptrs,*query.stride(),*key.stride(),*value.stride(),bm,debug,
        num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return out,kernel,extras
