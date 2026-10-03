"""Experimental whole C32 block in one 64-pixel-window program.

Keep FP16 dot accumulation and every current half/FP8 boundary. Padding is zero
extension, not cyclic shift. Pre, decoder upsample and post have other contracts
and are deliberately outside this primitive. No global intermediate tensors.
"""
import torch
import triton
import triton.language as tl
from nr_backend.triton_fp8 import _round_fp8_half
from nr_backend.triton_attention_normalize import _nan_left,rsqrt_half_clamped
from native_half_attention_fma_v1 import half_fma_attention
from fused_swin_core_native_half_v1 import _halves,_exp,_weights_pair
from fused_c32_mlp_lut_v1 import lookup


@triton.jit
def _normalize(z):
    first,last=_halves(z,64,32)
    x0,x8=_halves(first,64,16)
    x16,x24=_halves(last,64,16)
    a=(x16.to(tl.float32)*x16.to(tl.float32)).to(tl.float16)
    a=_nan_left(half_fma_attention(x0,x0,a),x0,a)
    b=(x24.to(tl.float32)*x24.to(tl.float32)).to(tl.float16)
    b=_nan_left(half_fma_attention(x8,x8,b),x8,b)
    total=_nan_left((a.to(tl.float32)+b.to(tl.float32)).to(tl.float16),a,b)
    lane=tl.arange(0,8)
    for i in tl.static_range(3):
        other=tl.gather(total,tl.broadcast_to((lane^(4>>i))[None,:],(64,8)),1)
        total=_nan_left((total.to(tl.float32)+other.to(tl.float32)).to(tl.float16),total,other)
    denominator=tl.gather(total,tl.full((64,1),0,tl.int32),1)
    scale=rsqrt_half_clamped(denominator)
    return _nan_left((z.to(tl.float32)*scale.to(tl.float32)).to(tl.float16),z,scale)


@triton.jit
def _kernel(X,EXPAND,CONTRACT,MLPSCALE,QKV,QSCALE,ORDER,BIAS,PROJECT,SKIPSCALE,LUT,OUT,
            H:tl.constexpr,W:tl.constexpr,SY:tl.constexpr,SX:tl.constexpr,
            STRIDEY:tl.constexpr,STRIDEX:tl.constexpr,STRIDEC:tl.constexpr):
    window=tl.program_id(0)
    row=tl.arange(0,64);lane=tl.arange(0,32)
    pixel=tl.load(ORDER+row).to(tl.int32)
    cols=(W+2*SX)//8
    y=window//cols*8+pixel//8-SY
    x=window%cols*8+pixel%8-SX
    valid=(y>=0)&(y<H)&(x>=0)&(x<W)
    raw=tl.load(X+y[:,None]*STRIDEY+x[:,None]*STRIDEX+lane[None,:]*STRIDEC,valid[:,None],other=0)
    raw=_round_fp8_half(raw)
    contracted=tl.full((64,32),0.,tl.float32)
    for part in range(4):
        we=tl.load(EXPAND+lane[:,None]*128+part*32+lane[None,:])
        expanded=tl.dot(raw,we,out_dtype=tl.float32).to(tl.float16)
        hidden=lookup(expanded,LUT)
        wc=tl.load(CONTRACT+(part*32+lane[:,None])*32+lane[None,:])
        contracted=tl.dot(hidden,wc,contracted,out_dtype=tl.float32)
    scale=tl.load(MLPSCALE+lane)
    initial=(raw.to(tl.float32)*scale[None,:].to(tl.float32)).to(tl.float16)
    mlp=(contracted+initial.to(tl.float32)).to(tl.float16)
    f=_round_fp8_half(mlp)
    wq=tl.load(QKV+lane[:,None]*96+lane[None,:])
    wk=tl.load(QKV+lane[:,None]*96+32+lane[None,:])
    wv=tl.load(QKV+lane[:,None]*96+64+lane[None,:])
    q=_normalize(tl.dot(f,wq,out_dtype=tl.float32).to(tl.float16))
    qscale=tl.load(QSCALE)
    q=_round_fp8_half(_nan_left((q.to(tl.float32)*qscale.to(tl.float32)).to(tl.float16),q,qscale))
    k=_round_fp8_half(_normalize(tl.dot(f,wk,out_dtype=tl.float32).to(tl.float16)))
    v=_round_fp8_half(tl.dot(f,wv,out_dtype=tl.float32).to(tl.float16))
    k0,k1=_halves(tl.trans(k),32,64)
    score0=tl.dot(q,k0,out_dtype=tl.float32)
    score1=tl.dot(q,k1,out_dtype=tl.float32)
    b0=tl.load(BIAS+row[:,None]*64+lane[None,:])
    b1=tl.load(BIAS+row[:,None]*64+32+lane[None,:])
    e0=_exp((score0+b0.to(tl.float32)).to(tl.float16))
    e1=_exp((score1+b1.to(tl.float32)).to(tl.float16))
    p0,p1=_weights_pair(e0,e1,64)
    vt0,vt1=_halves(tl.trans(v),32,64)
    attended=tl.dot(p0,tl.trans(vt0),out_dtype=tl.float32)
    attended=tl.dot(p1,tl.trans(vt1),attended,out_dtype=tl.float32)
    attended=_round_fp8_half(attended.to(tl.float16))
    wp=tl.load(PROJECT+lane[:,None]*32+lane[None,:])
    result=tl.dot(attended,wp,out_dtype=tl.float32)
    skip=tl.load(SKIPSCALE+lane)
    initial=(mlp.to(tl.float32)*skip[None,:].to(tl.float32)).to(tl.float16)
    result=(result+initial.to(tl.float32)).to(tl.float16)
    tl.store(OUT+(y*W+x)[:,None]*32+lane[None,:],result,valid[:,None])


def forward(features,expansion,contraction,mlp_scale,qkv,q_scale,order,bias,projection,skip_scale,lut,
            *,shift=(0,0),warps=4,stages=1):
    if features.device.type!='xpu' or features.dtype!=torch.float16 or features.ndim!=3 or features.shape[-1]!=32:
        raise ValueError('Expected half XPU HWC32 input')
    h,w=features.shape[:2]
    if min(h,w)<=0 or h%8 or w%8 or len(shift)!=2 or any(s not in (0,4) for s in shift):
        raise ValueError('Expected whole positive windows and observed zero/four shifts')
    for t,shape,dtype in ((expansion,(32,128),torch.float16),(contraction,(128,32),torch.float16),
        (mlp_scale,(32,),torch.float16),(qkv,(32,96),torch.float16),(q_scale,(1,),torch.float16),
        (order,(64,),torch.int64),(bias,(64,64),torch.float16),(projection,(32,32),torch.float16),
        (skip_scale,(32,),torch.float16),(lut,(65536,),torch.int16)):
        if t.device!=features.device or t.dtype!=dtype or t.shape!=shape or not t.is_contiguous():
            raise ValueError('Invalid owned C32 parameters')
    if warps not in (4,8) or stages not in (1,2):raise ValueError('Unsupported launch configuration')
    out=torch.empty((h,w,32),device=features.device,dtype=features.dtype)
    sy,sx=shift
    kernel=_kernel[((h+2*sy)*(w+2*sx)//64,)](features,expansion,contraction,mlp_scale,qkv,q_scale,order,
        bias,projection,skip_scale,lut,out,h,w,sy,sx,*features.stride(),
        num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return out,kernel
