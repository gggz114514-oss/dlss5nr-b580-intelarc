"""Experimental XMX arithmetic after user authorized visual-quality tradeoffs.

Original graph/FP8 boundaries remain. K8 is exact. K16 uses ordinary FP16
matrix products, or symmetric W8A8 dense matrices with FP16 attention.
These are deliberately NOT bit-equivalent to the NVIDIA accumulator rules.
"""
from contextlib import contextmanager
import torch
import triton
import triton.language as tl
import nr_backend.triton_math as reference

@triton.jit
def _quantize_rows(X, Q, SCALE, ROWS:tl.constexpr, K:tl.constexpr,
                   STRIDE_ROW:tl.constexpr, STRIDE_K:tl.constexpr, BK:tl.constexpr):
    row=tl.program_id(0)
    k=tl.arange(0,BK)
    x=tl.load(X+row*STRIDE_ROW+k*STRIDE_K,k<K,other=0).to(tl.float32)
    maximum=tl.max(tl.abs(x),0)
    scale=tl.where(maximum>0,maximum/127.0,1.0)
    normalized=x/scale
    # Symmetric nearest rounding, ties away from zero; no zero point.
    magnitude=tl.floor(tl.abs(normalized)+0.5)
    q=tl.minimum(magnitude,127.0)*tl.where(normalized<0,-1.0,1.0)
    tl.store(Q+row*K+k,q.to(tl.int8),k<K)
    tl.store(SCALE+row,scale)

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
        total=tl.dot(av,wv,total)
    value=total.to(tl.float32)
    if INT8:
        ar=tl.load(SA+rows,rows<M,other=1)
        wc=tl.load(SW+cols,cols<N,other=1)
        value=value*ar[:,None]*wc[None,:]
    offset=(batch*M+rows[:,None])*N+cols[None,:]
    valid=(rows[:,None]<M)&(cols[None,:]<N)
    if INITIALIZED:
        value+=tl.load(INITIAL+offset,valid,other=0).to(tl.float32)
    tl.store(OUT+offset,value.to(tl.float16),valid)

def quantize(x, *, columns=False):
    if x.device.type!='xpu' or x.ndim!=2 or x.dtype!=torch.float16:
        raise ValueError('Expected half XPU matrix')
    rows,k=(x.shape[1],x.shape[0]) if columns else x.shape
    sr,sk=(x.stride(1),x.stride(0)) if columns else x.stride()
    q=torch.empty((rows,k),device=x.device,dtype=torch.int8)
    scale=torch.empty(rows,device=x.device,dtype=torch.float32)
    compiled=_quantize_rows[(rows,)](x,q,scale,rows,k,sr,sk,triton.next_power_of_2(k),num_warps=4,enable_fp_fusion=False)
    return q,scale,compiled

def dot(a,w,*,initial=None,batched=False,int8=False,packed=None):
    if a.device.type!='xpu' or a.device!=w.device or a.shape[-1]!=w.shape[-2]:
        raise ValueError('Incompatible XPU matrices')
    if batched and (a.ndim<3 or a.ndim!=w.ndim or a.shape[:-2]!=w.shape[:-2]):
        raise ValueError('Incompatible batches')
    if not batched and w.ndim!=2:
        raise ValueError('Dense weight must be two-dimensional')
    if batched and int8:
        raise ValueError('V1 INT8 is dense-only')
    shape=(*a.shape[:-1],w.shape[-1])
    if initial is not None and (initial.shape!=shape or initial.device!=a.device):
        raise ValueError('Invalid initial accumulator')
    aa=a.half().contiguous().reshape(-1,a.shape[-2],a.shape[-1]) if batched else a.half().contiguous().reshape(-1,a.shape[-1])
    ww=w.half().contiguous()
    batches,m,k=aa.shape if batched else (1,*aa.shape)
    n=w.shape[-1]
    if int8:
        aa,sa,_=quantize(aa)
        if packed is None:ww,sw,_=quantize(ww,columns=True)
        else:ww,sw=packed
    else:sa=sw=aa
    ini=aa if initial is None else initial.half().contiguous()
    out=torch.empty(shape,device=a.device,dtype=torch.float16)
    compiled=_matmul[(triton.cdiv(m,16),triton.cdiv(n,32),batches)](
        aa,ww,sa,sw,ini,out,m,n,k,initial is not None,batched,int8,16,32,32,
        num_warps=4,enable_fp_fusion=False)
    return out,compiled

class FastMatrices:
    """Scoped experiment only; original source files and dispatch semantics stay intact."""
    def __init__(self):
        self.mode='baseline'
        self.original_dense=reference.fused_dot
        self.original_batched=reference.fused_batched_dot
        self.packed={}
        self.calls={}
        self.compiled={}

    def prepack(self,model):
        # Only immutable registered model buffers are cached. Dynamic attention
        # operands and temporary tensor views cannot be mistaken for weights.
        for name,w in model.named_buffers():
            if w.ndim==2 and w.dtype==torch.float16 and 16<=w.shape[0]<=4096 and w.shape[0]%16==0 and w.shape[1]%16==0 and w.numel()<=2**21:
                q,s,_=quantize(w,columns=True)
                self.packed[id(w)]=(w,q,s,name)
        torch.xpu.synchronize()

    def select(self,mode):
        if mode not in ('baseline','fp16_xmx','int8_dense'):
            raise ValueError('Unknown precision mode')
        self.mode=mode
        self.calls={}

    def record(self,kind):
        self.calls[kind]=self.calls.get(kind,0)+1

    def dense(self,a,w,*,chunk_k,initial=None,**kwargs):
        if chunk_k not in (8,16) or a.shape[-1]%chunk_k:
            raise ValueError('Unsupported original reduction size')
        if self.mode=='baseline' or chunk_k==8:
            self.record('exact_dense_k'+str(chunk_k))
            return self.original_dense(a,w,chunk_k=chunk_k,initial=initial,**kwargs)
        packed=None
        use_int8=self.mode=='int8_dense'
        if use_int8:
            item=self.packed.get(id(w))
            if item is not None and item[0] is w:
                packed=(item[1],item[2]);self.record('packed_weight_hit')
        out,compiled=dot(a,w,initial=initial,int8=use_int8,packed=packed)
        kind='int8_dense' if use_int8 else 'fp16_dense'
        self.record(kind)
        if kind not in self.compiled:self.compiled[kind]=compiled
        return out

    def batched(self,a,w,*,initial=None,**kwargs):
        if self.mode=='baseline':
            self.record('exact_batched')
            return self.original_batched(a,w,initial=initial,**kwargs)
        out,compiled=dot(a,w,initial=initial,batched=True)
        self.record('fp16_batched')
        if 'fp16_batched' not in self.compiled:self.compiled['fp16_batched']=compiled
        return out

    @contextmanager
    def installed(self):
        assert reference.fused_dot is self.original_dense and reference.fused_batched_dot is self.original_batched
        dense,batched=self.dense,self.batched
        reference.fused_dot=dense;reference.fused_batched_dot=batched
        try:yield self
        finally:
            assert reference.fused_dot is dense and reference.fused_batched_dot is batched
            reference.fused_dot=self.original_dense;reference.fused_batched_dot=self.original_batched
