"""Original exact tiled math with optional FP8 at the existing store boundary."""
import torch
import triton
import triton.language as tl
from nr_backend.triton_math import _exponent, _scaled_integer_to_half
from nr_backend.triton_fp8 import _round_fp8_half

@triton.jit
def _tiled(A,W,INITIAL,OUT,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
           CHUNK:tl.constexpr,INITIALIZED:tl.constexpr,WEIGHT_BATCHED:tl.constexpr,
           BM:tl.constexpr,BN:tl.constexpr,STORE_Q:tl.constexpr):
    batch=tl.program_id(2)
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    group=tl.arange(0,CHUNK)
    valid=(rows[:,None]<M)&(cols[None,:]<N)
    offset=(batch*M+rows[:,None])*N+cols[None,:]
    wb=batch if WEIGHT_BATCHED else 0
    if INITIALIZED:acc=tl.load(INITIAL+offset,valid,other=0).to(tl.float32)
    else:acc=tl.full((BM,BN),0,tl.float32)
    fraction:tl.constexpr=24 if CHUNK==8 else 13
    for start in range(0,K,CHUNK):
        av=tl.load(A+(batch*M+rows[:,None])*K+start+group[None,:],rows[:,None]<M,other=0).to(tl.float32)
        wv=tl.load(W+(wb*K+start+group[:,None])*N+cols[None,:],cols[None,:]<N,other=0).to(tl.float32)
        ea,ew=_exponent(av),_exponent(wv)
        if CHUNK==16:
            ea=tl.where(av==0,-1000,tl.maximum(ea,-6))
            ew=tl.where(wv==0,-1000,tl.maximum(ew,-6))
        exponent=tl.minimum(tl.maximum(tl.maximum(tl.max(ea[:,:,None]+ew[None,:,:],1),_exponent(acc)),-50),50)
        scale=((127+fraction-exponent)<<23).to(tl.float32,bitcast=True)
        product=av[:,:,None]*wv[None,:,:]
        summed=tl.sum((product*scale[:,None,:]).to(tl.int32),1)+(acc*scale).to(tl.int32)
        acc=_scaled_integer_to_half(summed,exponent-fraction).to(tl.float32)
    value=acc.to(tl.float16)
    if STORE_Q:value=_round_fp8_half(value)
    tl.store(OUT+offset,value,valid)


def dot_q(a, w, *, initial=None, bm=4, bn=32):
    if a.device.type != 'xpu' or a.device != w.device or w.ndim != 2 or a.shape[-1] != w.shape[0] or a.shape[-1] % 16:
        raise ValueError('Expected exact dense K16 matrices')
    shape = (*a.shape[:-1], w.shape[-1])
    if initial is not None and (initial.shape != shape or initial.device != a.device):
        raise ValueError('Invalid initial accumulator')
    aa, ww = a.half().contiguous(), w.half().contiguous()
    m, k, n = aa.numel()//aa.shape[-1], aa.shape[-1], ww.shape[-1]
    output = torch.empty(shape, device=a.device, dtype=torch.float16)
    _tiled[(triton.cdiv(m,bm),triton.cdiv(n,bn),1)](
        aa, ww, aa if initial is None else initial.half().contiguous(), output,
        m,n,k,16,initial is not None,False,bm,bn,True,
        num_warps=1,enable_fp_fusion=False)
    return output
