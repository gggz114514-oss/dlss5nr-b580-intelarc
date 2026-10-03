"""Local mixed dot prototype; lossless operands do not imply FP32 sum parity.

Prepacked weights have exact per K32/column binary scales. Dynamic activation
rows are classified and converted inside the measured kernel. A tile needing
more than one signed INT8 plane keeps the selected FP16 K32 dot instead.
No zero skipping, new FP8 quantization, or temporary activation tensor.
"""
import torch
import triton
import triton.language as tl
from spill_preflight_v1 import select


@triton.jit
def _or(a,b):return a|b


@triton.jit
def _kernel(A,W,W8,WS,INITIAL,OUT,FLAGS,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
            BM:tl.constexpr,BN:tl.constexpr,DEBUG:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*BN+tl.arange(0,BN)
    lane=tl.arange(0,32)
    total=tl.full((BM,BN),0,tl.float32)
    for part in range(K//32):
        k=part*32+lane
        av=tl.load(A+row[:,None]*K+k[None,:],row[:,None]<M,other=0)
        scaled=av.to(tl.float32)*512.0
        ai=scaled.to(tl.int32)
        ored=tl.reduce(tl.abs(ai),1,_or)
        factor=tl.maximum(ored&(-ored),1)
        shift=((factor.to(tl.float32).to(tl.int32,bitcast=True)>>23)&255)-127
        aq=ai>>shift[:,None]
        row_ok=(tl.min(aq,1)>=-128)&(tl.max(aq,1)<=127)
        row_ok=row_ok&(tl.min((scaled==ai.to(tl.float32)).to(tl.int32),1)!=0)
        fits=tl.min(row_ok.to(tl.int32),0)!=0
        if DEBUG:
            if tl.program_id(1)==0:
                tl.store(FLAGS+tl.program_id(0)*(K//32)+part,fits.to(tl.int8))
        if fits:
            bv=tl.load(W8+k[:,None]*N+col[None,:],col[None,:]<N,other=0)
            integer=tl.dot(aq.to(tl.int8),bv,out_dtype=tl.int32)
            ws=tl.load(WS+part*N+col,col<N,other=0)
            subtotal=integer.to(tl.float32)*(factor.to(tl.float32)[:,None]*(1.0/512.0))
            total=total+subtotal*ws[None,:]
        else:
            bv=tl.load(W+k[:,None]*N+col[None,:],col[None,:]<N,other=0)
            total=tl.dot(av,bv,total,out_dtype=tl.float32)
    valid=(row[:,None]<M)&(col[None,:]<N)
    offset=row[:,None]*N+col[None,:]
    initial=tl.load(INITIAL+offset,valid,other=0)
    tl.store(OUT+offset,(total+initial.to(tl.float32)).to(tl.float16),valid)


def arguments(a,w,w8,scales,initial,out,flags,debug):
    assert a.ndim==w.ndim==initial.ndim==out.ndim==2
    m,k=a.shape;wk,n=w.shape
    assert k==wk and k%32==n%32==0 and initial.shape==out.shape==(m,n)
    assert w8.shape==w.shape and scales.shape==(k//32,n)
    assert flags.shape==(triton.cdiv(m,16),k//32)
    for t,dtype in ((a,torch.float16),(w,torch.float16),(w8,torch.int8),
                    (scales,torch.float32),(initial,torch.float16),(out,torch.float16),(flags,torch.int8)):
        assert t.device.type=='xpu' and t.device==a.device and t.dtype==dtype and t.is_contiguous()
    return (a,w,w8,scales,initial,out,flags,m,n,k,16,32,debug),(triton.cdiv(m,16),triton.cdiv(n,32))


def prepare(a,w,w8,scales,initial,out,flags,*,debug=False):
    args,grid=arguments(a,w,w8,scales,initial,out,flags,debug)
    _,compiled,selection=select(_kernel,[(16,32)],lambda _:args,lambda _:grid,
        num_warps=4,num_stages=1,enable_fp_fusion=False)
    return args,grid,compiled,selection


def launch(args,grid):
    return _kernel[grid](*args,num_warps=4,num_stages=1,enable_fp_fusion=False)
