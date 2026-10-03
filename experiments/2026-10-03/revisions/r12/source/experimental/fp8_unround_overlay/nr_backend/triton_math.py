"""SM89-observed arithmetic with exact K16 output tiling; not XMX.

This translates the existing finite-half shared-exponent integer oracle into
one kernel. Do not replace backend functions until primitive and full-frame
native-reference validation both pass. FP fusion is disabled intentionally.
"""
import triton
import triton.language as tl
import torch


@triton.jit
def _exponent(x):
    bits = x.to(tl.int32, bitcast=True)
    return tl.where(x == 0, -1000, ((bits >> 23) & 255) - 127)


@triton.jit
def _scaled_integer_to_half(value, scale_exponent):
    magnitude = tl.abs(value)
    lead = tl.minimum(tl.maximum(_exponent(magnitude.to(tl.float32)), 0), 30)
    one = tl.full((), 1, tl.int32)
    lead = lead - (magnitude < (one << lead)).to(tl.int32)
    lead = tl.where(magnitude == 0, 0, lead)
    exponent = lead + scale_exponent
    shift = tl.maximum(exponent - 10, -24) - scale_exponent
    rs = tl.minimum(tl.maximum(shift, 0), 31)
    quotient = magnitude >> rs
    remainder = magnitude - (quotient << rs)
    midpoint = one << tl.minimum(tl.maximum(rs - 1, 0), 30)
    increment = (rs > 0) & ((remainder > midpoint) | ((remainder == midpoint) & ((quotient & 1) != 0)))
    rounded = quotient + increment.to(tl.int32)
    rounded = tl.where(shift < 0, magnitude << tl.minimum(tl.maximum(-shift, 0), 31), rounded)
    bits = tl.minimum(tl.maximum(exponent + 14, 0).to(tl.int32) * 1024 + rounded, 0x7c00)
    bits = tl.where(magnitude == 0, 0, bits) | ((value < 0).to(tl.int32) << 15)
    return bits.to(tl.uint16).to(tl.float16, bitcast=True)


@triton.jit
def _dot(A, W, INITIAL, OUTPUT, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
         CHUNK: tl.constexpr, INITIALIZED: tl.constexpr, BLOCK: tl.constexpr,
         BATCHES: tl.constexpr, WEIGHT_BATCHED: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = offset < BATCHES * M * N
    row, col = offset // N, offset % N
    weight_batch = row // M if WEIGHT_BATCHED else tl.full((BLOCK,), 0, tl.int32)
    if INITIALIZED:
        accumulator = tl.load(INITIAL + offset, valid, other=0).to(tl.float32)
    else:
        accumulator = tl.full((BLOCK,), 0, tl.float32)
    group = tl.arange(0, CHUNK)
    fraction_bits: tl.constexpr = 24 if CHUNK == 8 else 13
    for start in range(0, K, CHUNK):
        av = tl.load(A + row[None, :] * K + start + group[:, None], valid[None, :], other=0).to(tl.float32)
        wv = tl.load(W + (weight_batch[None, :] * K + start + group[:, None]) * N + col[None, :], valid[None, :], other=0).to(tl.float32)
        ea, ew = _exponent(av), _exponent(wv)
        if CHUNK == 16:
            ea = tl.where(av == 0, -1000, tl.maximum(ea, -6))
            ew = tl.where(wv == 0, -1000, tl.maximum(ew, -6))
        exponent = tl.minimum(tl.maximum(tl.maximum(tl.max(ea + ew, 0), _exponent(accumulator)), -50), 50)
        # Exact normal power of two, equivalent to the oracle's bounded ldexp.
        scale = ((127 + fraction_bits - exponent) << 23).to(tl.float32, bitcast=True)
        products = av * wv
        # Finite-half product sums fit INT32; see sm89-i32-product-bound-v1.json.
        product_sum = tl.sum((products * scale[None, :]).to(tl.int32), 0)
        aligned = product_sum + (accumulator * scale).to(tl.int32)
        accumulator = _scaled_integer_to_half(aligned, exponent - fraction_bits).to(tl.float32)
    tl.store(OUTPUT + offset, accumulator.to(tl.float16), valid)


def fused_dot(a, weight, *, chunk_k, initial=None, block=128):
    if a.device != weight.device or weight.ndim != 2 or a.shape[-1] != weight.shape[0]:
        raise ValueError('Expected compatible matrices on one device')
    if chunk_k not in (8, 16) or a.shape[-1] % chunk_k:
        raise ValueError('Expected complete K8/K16 groups')
    shape = (*a.shape[:-1], weight.shape[1])
    if initial is not None and (initial.device != a.device or initial.shape != shape):
        raise ValueError('Initial accumulator must match output')
    if chunk_k == 16:
        return _tiled_dot(a, weight, chunk_k=chunk_k, initial=initial)
    aa = a.half().contiguous().reshape(-1, a.shape[-1])
    ww = weight.half().contiguous()
    init = aa if initial is None else initial.half().contiguous()
    output = torch.empty(shape, dtype=torch.float16, device=a.device)
    _dot[(triton.cdiv(output.numel(), block),)](
        aa, ww, init, output, aa.shape[0], ww.shape[1], ww.shape[0],
        chunk_k, initial is not None, block, 1, False, enable_fp_fusion=False,
    )
    return output


def fused_batched_dot(a, weight, *, initial=None, batches_per_step=32, block=128):
    if a.ndim < 3 or weight.ndim != a.ndim or a.shape[:-2] != weight.shape[:-2] or a.shape[-1] != weight.shape[-2]:
        raise ValueError('Expected matching matrix batches')
    if a.device != weight.device or a.shape[-1] % 32 or batches_per_step < 1:
        raise ValueError('Expected whole K32 instructions on one device')
    shape = (*a.shape[:-1], weight.shape[-1])
    if initial is not None and (initial.shape != shape or initial.device != a.device):
        raise ValueError('Initial accumulator must match output')
    return _tiled_dot(a, weight, initial=initial, batched=True)


@triton.jit
def _tiled(A,W,INITIAL,OUT,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
           CHUNK:tl.constexpr,INITIALIZED:tl.constexpr,WEIGHT_BATCHED:tl.constexpr,
           BM:tl.constexpr,BN:tl.constexpr):
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
    tl.store(OUT+offset,acc.to(tl.float16),valid)

def _tiled_dot(a,w,*,chunk_k=16,initial=None,batched=False,bm=4,bn=32,warps=1):
    if a.device.type!='xpu' or a.device!=w.device or chunk_k not in(8,16) or a.shape[-1]%chunk_k:
        raise ValueError('Compatible XPU matrices with completeK8/K16 required')
    if batched:
        if a.ndim<3 or a.ndim!=w.ndim or a.shape[:-2]!=w.shape[:-2] or a.shape[-1]!=w.shape[-2]:
            raise ValueError('Invalid batches')
        shape=(*a.shape[:-1],w.shape[-1]);aa=a.half().contiguous().reshape(-1,a.shape[-2],a.shape[-1])
        ww=w.half().contiguous();batches,m,k=aa.shape
    else:
        if w.ndim!=2 or a.shape[-1]!=w.shape[0]:raise ValueError('Invalid dense dimensions')
        shape=(*a.shape[:-1],w.shape[1]);aa=a.half().contiguous().reshape(-1,a.shape[-1])
        ww=w.half().contiguous();m,k=aa.shape;batches=1
    if initial is not None and(initial.shape!=shape or initial.device!=a.device):raise ValueError('Invalid initial accumulator')
    ini=aa if initial is None else initial.half().contiguous()
    output=torch.empty(shape,device=a.device,dtype=torch.float16)
    _tiled[(triton.cdiv(m,bm),triton.cdiv(w.shape[-1],bn),batches)](
        aa,ww,ini,output,m,w.shape[-1],k,chunk_k,initial is not None,batched,bm,bn,
        num_warps=warps,enable_fp_fusion=False)
    return output
