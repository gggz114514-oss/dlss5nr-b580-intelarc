# Fast-only C512 probability rounding ablation; not used by the exact branch.
"""Execute complete C512 Swin windows with optional E4M3 probability rounding.

Same BM32 per-window arithmetic as fused_swin_core_v2, head-specific bias.
Input and output retain head,row,col,64,32 order; no per-head result stack.
"""
import torch
import triton
import triton.language as tl
from fused_swin_core_native_half_v1 import _exp, _weights_pair


@triton.jit
def _kernel(Q,K,V,BIAS,OUT,WPH:tl.constexpr,BM:tl.constexpr,
            ROUND_WEIGHTS:tl.constexpr=True):
    window=tl.program_id(1)
    head=window//WPH
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    lane=tl.arange(0,32)
    q=tl.load(Q+window*2048+row[:,None]*32+lane[None,:])
    k0=tl.load(K+window*2048+lane[None,:]*32+lane[:,None])
    k1=tl.load(K+window*2048+(lane[None,:]+32)*32+lane[:,None])
    score0=tl.dot(q,k0,out_dtype=tl.float32)
    score1=tl.dot(q,k1,out_dtype=tl.float32)
    b0=tl.load(BIAS+head*4096+row[:,None]*64+lane[None,:])
    b1=tl.load(BIAS+head*4096+row[:,None]*64+lane[None,:]+32)
    e0=_exp((score0+b0.to(tl.float32)).to(tl.float16))
    e1=_exp((score1+b1.to(tl.float32)).to(tl.float16))
    w0,w1=_weights_pair(e0,e1,BM,ROUND_WEIGHTS=ROUND_WEIGHTS)
    v0=tl.load(V+window*2048+lane[:,None]*32+lane[None,:])
    v1=tl.load(V+window*2048+(lane[:,None]+32)*32+lane[None,:])
    result=tl.dot(w0,v0,out_dtype=tl.float32)
    result=tl.dot(w1,v1,result,out_dtype=tl.float32)
    tl.store(OUT+window*2048+row[:,None]*32+lane[None,:],result.to(tl.float16))


def forward(query,key,value,bias,*,round_weights=True):
    if query.ndim!=5 or query.shape!=key.shape or query.shape!=value.shape or query.shape[-2:]!=(64,32):
        raise ValueError('Expected matching head,row,col,64,32 tensors')
    if query.shape[0] != 16:
        raise ValueError('This candidate is restricted to C512 with 16 heads')
    if tuple(bias.shape)!=(query.shape[0],64,64):
        raise ValueError('Expected one independent 64x64 bias per head')
    for t in (query,key,value,bias):
        if t.device.type!='xpu' or t.device!=query.device or t.dtype!=torch.float16 or not t.is_contiguous() or t.numel()==0:
            raise ValueError('Expected contiguous nonempty half XPU tensors')
    out=torch.empty_like(query)
    kernel=_kernel[(2,query.numel()//2048)](query,key,value,bias,out,query.shape[1]*query.shape[2],32,
        ROUND_WEIGHTS=round_weights,
        num_warps=4,num_stages=1,enable_fp_fusion=False)
    return out,kernel
