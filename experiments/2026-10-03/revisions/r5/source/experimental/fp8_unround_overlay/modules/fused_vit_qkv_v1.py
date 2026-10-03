"""Two independent K512 QKV products followed by fused half merge/preparation.

Preserves the original FP16 fast BK32 products, half sum, exact normalization
helper, separate half query-scale boundaries, and original FP8 conversion.
The complete input tensor is read by stride across each K partition, avoiding
materialized half-width feature copies. No INT8 or exact-backend override.
"""
import torch
import triton
import triton.language as tl
from nr_backend.triton_attention_normalize import _nan_left, rsqrt_half_clamped
from nr_backend.triton_cubic_fp8 import _half_fma_value
from nr_backend.triton_fp8 import _round_fp8_half

@triton.jit
def _parts(X,W,OUT,M:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*BN+tl.arange(0,BN)
    part=tl.program_id(2)
    lane=tl.arange(0,32)
    total=tl.full((BM,BN),0.,tl.float32)
    for block in range(16):
        k=part*512+block*32+lane
        x=tl.load(X+row[:,None]*1024+k[None,:],row[:,None]<M,other=0)
        w=tl.load(W+k[:,None]*3072+col[None,:],col[None,:]<3072,other=0)
        total=tl.dot(x,w,total,out_dtype=tl.float32)
    offset=row[:,None]*3072+col[None,:]
    tl.store(OUT+part*M*3072+offset,total.to(tl.float16),(row[:,None]<M)&(col[None,:]<3072))

@triton.jit
def _merged(PARTS,offset,valid,SIZE:tl.constexpr):
    a=tl.load(PARTS+offset,valid,other=0)
    b=tl.load(PARTS+SIZE+offset,valid,other=0)
    return _nan_left((a.to(tl.float32)+b.to(tl.float32)).to(tl.float16),a,b)

@triton.jit
def _prepare(PARTS,SCALE,Q,K,V,M:tl.constexpr,COUNT:tl.constexpr,BR:tl.constexpr,
             ROUND_QKV:tl.constexpr=True):
    r=tl.program_id(0)*BR+tl.arange(0,BR)
    family=tl.program_id(1)
    lane=tl.arange(0,8)
    head=r%32
    off=(r*96+family*32)[:,None]+lane[None,:]
    valid=r[:,None]<COUNT
    x0=_merged(PARTS,off,valid,M*3072)
    x8=_merged(PARTS,off+8,valid,M*3072)
    x16=_merged(PARTS,off+16,valid,M*3072)
    x24=_merged(PARTS,off+24,valid,M*3072)
    if family < 2:
        a = (x16.to(tl.float32) * x16.to(tl.float32)).to(tl.float16)
        a = _nan_left(_half_fma_value(x0, x0, a), x0, a)
        b = (x24.to(tl.float32) * x24.to(tl.float32)).to(tl.float16)
        b = _nan_left(_half_fma_value(x8, x8, b), x8, b)
        total = _nan_left((a.to(tl.float32) + b.to(tl.float32)).to(tl.float16), a, b)
        for mask in tl.static_range(3):
            other = tl.gather(total, tl.broadcast_to((lane ^ (4 >> mask))[None, :], (BR, 8)), 1)
            total = _nan_left((total.to(tl.float32) + other.to(tl.float32)).to(tl.float16), total, other)
        denominator = tl.gather(total, tl.full((BR, 1), 0, tl.int32), 1)
        scale = rsqrt_half_clamped(denominator)
        x0 = _nan_left((x0.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x0, scale)
        x8 = _nan_left((x8.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x8, scale)
        x16 = _nan_left((x16.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x16, scale)
        x24 = _nan_left((x24.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x24, scale)
        if family == 0:
            scale = tl.load(SCALE + head, r < COUNT, other=0)[:, None]
            x0 = (x0.to(tl.float32) * 5.65625).to(tl.float16)
            x0 = _nan_left((x0.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x0, scale)
            x8 = (x8.to(tl.float32) * 5.65625).to(tl.float16)
            x8 = _nan_left((x8.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x8, scale)
            x16 = (x16.to(tl.float32) * 5.65625).to(tl.float16)
            x16 = _nan_left((x16.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x16, scale)
            x24 = (x24.to(tl.float32) * 5.65625).to(tl.float16)
            x24 = _nan_left((x24.to(tl.float32) * scale.to(tl.float32)).to(tl.float16), x24, scale)
    output = tl.where(family == 0, Q, tl.where(family == 1, K, V))
    dest = r[:, None] * 32 + lane[None, :]
    if ROUND_QKV:
        x0=_round_fp8_half(x0);x8=_round_fp8_half(x8)
        x16=_round_fp8_half(x16);x24=_round_fp8_half(x24)
    tl.store(output + dest, x0, valid)
    tl.store(output + dest + 8, x8, valid)
    tl.store(output + dest + 16, x16, valid)
    tl.store(output + dest + 24, x24, valid)


def forward(features,weight,scale,*,bm=16,bn=32,rows=16,round_qkv=True):
    if features.ndim!=2 or features.shape[0]<=0 or features.shape[1]!=1024:
        raise ValueError('Expected nonempty M by1024 features')
    if weight.shape!=(1024,3072) or scale.shape!=(32,):
        raise ValueError('Expected pinned QKV and query scales')
    for t in (features,weight,scale):
        if t.device.type!='xpu' or t.device!=features.device or t.dtype!=torch.float16:
            raise ValueError('Expected same-device half XPU tensors')
    if not weight.is_contiguous() or not scale.is_contiguous():
        raise ValueError('Expected contiguous model constants')
    if bm not in (16,32) or bn not in (32,64) or rows not in (8,16,32):
        raise ValueError('Unsupported launch')
    x=features.contiguous();m=x.shape[0]
    partial=torch.empty((2,m,3072),dtype=x.dtype,device=x.device)
    outputs=tuple(torch.empty((m,32,32),dtype=x.dtype,device=x.device) for _ in range(3))
    dot=_parts[(triton.cdiv(m,bm),triton.cdiv(3072,bn),2)](x,weight,partial,m,bm,bn,
        num_warps=4,num_stages=1,enable_fp_fusion=False)
    prepare=_prepare[(triton.cdiv(m*32,rows),3)](partial,scale,*outputs,m,m*32,rows,
        ROUND_QKV=round_qkv,
        num_warps=4,num_stages=1,enable_fp_fusion=False)
    return outputs,(dot,prepare)
