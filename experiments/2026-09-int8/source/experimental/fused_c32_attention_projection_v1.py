"""Fuse C32 attention, FP8 boundary, projection, residual and cropped scatter.

Keep the tested MLP and QKV kernels on the stock compiler pipeline. Only the
attention/projection tail is new; avoid attended HWC materialization and unpack.
Complete padded windows still contribute to valid boundary queries.
"""
import torch
import triton
import triton.language as tl
from nr_backend.triton_fp8 import quantize_fp8,_round_fp8_half
from fused_swin_core_native_half_v1 import _exp,_weights_pair
from fused_c32_mlp_lut_v1 import forward as mlp_forward
from fused_c32_projection_native_half_v1 import forward as projection_forward


@triton.jit
def _tail(MLP,Q,K,V,BIAS,PROJECT,SKIPSCALE,ORDER,OUT,H:tl.constexpr,W:tl.constexpr,SY:tl.constexpr,SX:tl.constexpr):
    window=tl.program_id(1)
    row=tl.program_id(0)*32+tl.arange(0,32);lane=tl.arange(0,32)
    q=tl.load(Q+window*2048+row[:,None]*32+lane[None,:])
    k0=tl.load(K+window*2048+lane[None,:]*32+lane[:,None])
    k1=tl.load(K+window*2048+(lane[None,:]+32)*32+lane[:,None])
    score0=tl.dot(q,k0,out_dtype=tl.float32)
    score1=tl.dot(q,k1,out_dtype=tl.float32)
    b0=tl.load(BIAS+row[:,None]*64+lane[None,:])
    b1=tl.load(BIAS+row[:,None]*64+32+lane[None,:])
    e0=_exp((score0+b0.to(tl.float32)).to(tl.float16))
    e1=_exp((score1+b1.to(tl.float32)).to(tl.float16))
    p0,p1=_weights_pair(e0,e1,32)
    v0=tl.load(V+window*2048+lane[:,None]*32+lane[None,:])
    v1=tl.load(V+window*2048+(lane[:,None]+32)*32+lane[None,:])
    attended=tl.dot(p0,v0,out_dtype=tl.float32)
    attended=tl.dot(p1,v1,attended,out_dtype=tl.float32)
    attended=_round_fp8_half(attended.to(tl.float16))
    wp=tl.load(PROJECT+lane[:,None]*32+lane[None,:])
    result=tl.dot(attended,wp,out_dtype=tl.float32)
    pixel=tl.load(ORDER+row).to(tl.int32)
    cols=(W+2*SX)//8
    y=window//cols*8+pixel//8-SY
    x=window%cols*8+pixel%8-SX
    mlp=tl.load(MLP+((y+SY)*(W+2*SX)+(x+SX))[:,None]*32+lane[None,:])
    skip=tl.load(SKIPSCALE+lane)
    initial=(mlp.to(tl.float32)*skip[None,:].to(tl.float32)).to(tl.float16)
    result=(result+initial.to(tl.float32)).to(tl.float16)
    valid=(y>=0)&(y<H)&(x>=0)&(x<W)
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
    x=quantize_fp8(features)
    if sy or sx:x=torch.nn.functional.pad(x,(0,0,sx,sx,sy,sy))
    mlp,km=mlp_forward(x,expansion,contraction,mlp_scale,lut,bm=32,warps=4,stages=1)
    (q,k,v),kp=projection_forward(quantize_fp8(mlp),qkv,q_scale,order,bm=32,warps=4,stages=1)
    windows=(h+2*sy)*(w+2*sx)//64
    kernel=_tail[(2,windows)](mlp,q,k,v,bias,projection,skip_scale,order,out,h,w,sy,sx,
        num_warps=warps,num_stages=stages,enable_fp_fusion=False)
    return out,(km,kp,kernel)
