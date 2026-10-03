"""Fuse the existing row quantizer and cached W8A8 dot for bounded small K.

Finite FP16 operands: identical max, correctly-rounded FP32 division, symmetric
rounding and scales. K<=256 bounds INT32 dot magnitude below 2**23, so integer
reassociation cannot overflow or lose precision at the FP32 conversion. Original
two ordered scale multiplies, initial addition and final half conversion remain.
No numerical equivalence claim for NaN/Inf operands.
"""
import torch
import triton
import triton.language as tl

@triton.jit
def _fused(A,W,SW,INITIAL,OUT,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
           HAS_INITIAL:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    columns=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    a=tl.load(A+rows[:,None]*K+kk[None,:],(rows[:,None]<M)&(kk[None,:]<K),other=0).to(tl.float32)
    maximum=tl.max(tl.abs(a),1)
    scale=tl.where(maximum>0,tl.div_rn(maximum,127.),1.)
    normalized=tl.div_rn(a,scale[:,None])
    magnitude=tl.minimum(tl.floor(tl.abs(normalized)+.5),127.)
    qa=(magnitude*tl.where(normalized<0,-1.,1.)).to(tl.int8)
    qw=tl.load(W+columns[None,:]*K+kk[:,None],(columns[None,:]<N)&(kk[:,None]<K),other=0)
    total=tl.dot(qa,qw,out_dtype=tl.int32)
    sw=tl.load(SW+columns,columns<N,other=1.)
    value=(total.to(tl.float32)*scale[:,None])*sw[None,:]
    offset=rows[:,None]*N+columns[None,:]
    valid=(rows[:,None]<M)&(columns[None,:]<N)
    if HAS_INITIAL:value+=tl.load(INITIAL+offset,valid,other=0).to(tl.float32)
    tl.store(OUT+offset,value.to(tl.float16),valid)

def dot(a,w,*,packed,initial=None,bm=16,bn=64,warps=4):
    if a.device.type!='xpu' or a.device!=w.device or w.ndim!=2 or a.shape[-1]!=w.shape[0]:
        raise ValueError('Expected compatible XPU dense matrices')
    k,n=w.shape
    if k<16 or k>256 or k%16 or n<1:
        raise ValueError('Fused activation supports K16..256 in groups of16')
    shape=(*a.shape[:-1],n)
    if initial is not None and (initial.device!=a.device or initial.shape!=shape):
        raise ValueError('Initial accumulator mismatch')
    q,s=packed
    if q.device!=a.device or s.device!=a.device or q.dtype!=torch.int8 or s.dtype!=torch.float32 or q.shape!=(n,k) or s.shape!=(n,) or not q.is_contiguous() or not s.is_contiguous():
        raise ValueError('Invalid packed weights')
    aa=a.half().contiguous().reshape(-1,k)
    m=aa.shape[0]
    ini=aa if initial is None else initial.half().contiguous()
    out=torch.empty(shape,device=a.device,dtype=torch.float16)
    kernel=_fused[(triton.cdiv(m,bm),triton.cdiv(n,bn))](aa,q,s,ini,out,m,n,k,initial is not None,bm,bn,max(32,triton.next_power_of_2(k)),num_warps=warps,enable_fp_fusion=False)
    return out,kernel
