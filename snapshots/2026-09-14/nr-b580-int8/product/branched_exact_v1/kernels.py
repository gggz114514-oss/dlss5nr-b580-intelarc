"""Independent branches batched, dependent projections accumulated in order.

K16 exponent selection, integer product truncation and half rounding are copied
from nr_backend.triton_math._tiled. No XMX dot or reassociated branch reduction.
"""
import torch
import triton
import triton.language as tl
from nr_backend.triton_math import _exponent, _scaled_integer_to_half
from exact_all_cubic_v1.cubic import cubic
from exact_shortfp8_v1.fp8 import _round_fp8_half


@triton.jit
def step(av,wv,acc):
    ea,ew=_exponent(av),_exponent(wv)
    ea=tl.where(av==0,-1000,tl.maximum(ea,-6))
    ew=tl.where(wv==0,-1000,tl.maximum(ew,-6))
    exponent=tl.minimum(tl.maximum(tl.maximum(tl.max(ea[:,:,None]+ew[None,:,:],1),_exponent(acc)),-50),50)
    scale=((127+13-exponent)<<23).to(tl.float32,bitcast=True)
    product=av[:,:,None]*wv[None,:,:]
    summed=tl.sum((product*scale[:,None,:]).to(tl.int32),1)+(acc*scale).to(tl.int32)
    return _scaled_integer_to_half(summed,exponent-13).to(tl.float32)


@triton.jit
def branches(A,W,OUT,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
             SHARED_INPUT:tl.constexpr,CUBIC:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    branch=tl.program_id(2)
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    group=tl.arange(0,16)
    input_branch=0 if SHARED_INPUT else branch
    acc=tl.full((BM,BN),0,tl.float32)
    for start in range(0,K,16):
        av=tl.load(A+(input_branch*M+rows[:,None])*K+start+group[None,:],rows[:,None]<M,other=0).to(tl.float32)
        wv=tl.load(W+(branch*K+start+group[:,None])*N+cols[None,:],cols[None,:]<N,other=0).to(tl.float32)
        acc=step(av,wv,acc)
    value=cubic(acc.to(tl.float16)) if CUBIC else _round_fp8_half(acc.to(tl.float16))
    tl.store(OUT+(branch*M+rows[:,None])*N+cols[None,:],value,(rows[:,None]<M)&(cols[None,:]<N))


@triton.jit
def projection(A,W,INIT,OUT,M:tl.constexpr,C:tl.constexpr,B:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN)
    group=tl.arange(0,16)
    valid=(rows[:,None]<M)&(cols[None,:]<C)
    acc=tl.load(INIT+rows[:,None]*C+cols[None,:],valid,other=0).to(tl.float32)
    for branch in range(B):
        for start in range(0,32,16):
            av=tl.load(A+(branch*M+rows[:,None])*32+start+group[None,:],rows[:,None]<M,other=0).to(tl.float32)
            wv=tl.load(W+(branch*32+start+group[:,None])*C+cols[None,:],cols[None,:]<C,other=0).to(tl.float32)
            acc=step(av,wv,acc)
    tl.store(OUT+rows[:,None]*C+cols[None,:],acc.to(tl.float16),valid)


def forward(module,x,*,diagnostics=False):
    if x.device.type!='xpu' or x.dtype!=torch.float16 or x.shape[-1]!=module.channels:
        raise ValueError('Expected quantized half XPU input matching the owned module')
    channels=module.channels
    count=channels//32
    shape=x.shape
    x=x.contiguous()
    m=x.numel()//channels
    expanded=torch.empty((count,m,128),device=x.device,dtype=x.dtype)
    hidden=torch.empty((count,m,32),device=x.device,dtype=x.dtype)
    initial=(x*module.skip_scale).half()
    output=torch.empty(shape,device=x.device,dtype=x.dtype)
    first=branches[(triton.cdiv(m,4),4,count)](x,module.expand,expanded,m,128,channels,True,True,4,32,num_warps=1,enable_fp_fusion=False)
    second=branches[(triton.cdiv(m,4),1,count)](expanded,module.reduce,hidden,m,32,128,False,False,4,32,num_warps=1,enable_fp_fusion=False)
    third=projection[(triton.cdiv(m,4),triton.cdiv(channels,32))](hidden,module.project,initial,output,m,channels,count,4,32,num_warps=1,enable_fp_fusion=False)
    resources=[]
    if diagnostics:
        for kernel in (first,second,third):
            spills=getattr(kernel,'n_spills',None)
            resources.append(dict(name=kernel.name,hash=kernel.hash,spills=spills,registers=kernel.n_regs))
            if spills is None or spills!=0:
                raise RuntimeError(f'Unaccepted branched kernel spill count: {spills}')
    return output,resources
