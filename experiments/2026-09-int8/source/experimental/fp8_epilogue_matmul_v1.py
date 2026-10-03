"""Original FP16 BK32 matrix reduction with FP8 rounding after the half store boundary.

Only the authenticated nonbatched, non-INT8 call sites selected by the complete
value-use plan are supported by the graph adapter. No reduction reassociation.
"""
import triton
import triton.language as tl
from nr_backend.triton_fp8 import _round_fp8_half

@triton.jit
def _matmul(A,W,SA,SW,INITIAL,OUT,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
            INITIALIZED:tl.constexpr,BATCHED:tl.constexpr,INT8:tl.constexpr,
            BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr):
    batch=tl.program_id(2)
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    wb=batch if BATCHED else 0
    if INT8:
        total=tl.full((BM,BN),0,tl.int32)
    else:
        total=tl.full((BM,BN),0,tl.float32)
    for start in range(0,tl.cdiv(K,BK)):
        k=start*BK+kk
        av=tl.load(A+(batch*M+rows[:,None])*K+k[None,:],(rows[:,None]<M)&(k[None,:]<K),other=0)
        if INT8:
            # Packed weights are contiguous [N,K], stored once per model buffer.
            wv=tl.load(W+cols[None,:]*K+k[:,None],(cols[None,:]<N)&(k[:,None]<K),other=0)
        else:
            wv=tl.load(W+(wb*K+k[:,None])*N+cols[None,:],(cols[None,:]<N)&(k[:,None]<K),other=0)
        total=tl.dot(av,wv,total,out_dtype=tl.int32 if INT8 else tl.float32)
    value=total.to(tl.float32)
    if INT8:
        ar=tl.load(SA+rows,rows<M,other=1)
        wc=tl.load(SW+cols,cols<N,other=1)
        value=value*ar[:,None]*wc[None,:]
    offset=(batch*M+rows[:,None])*N+cols[None,:]
    valid=(rows[:,None]<M)&(cols[None,:]<N)
    if INITIALIZED:
        value+=tl.load(INITIAL+offset,valid,other=0).to(tl.float32)
    tl.store(OUT+offset,_round_fp8_half(value.to(tl.float16)),valid)
