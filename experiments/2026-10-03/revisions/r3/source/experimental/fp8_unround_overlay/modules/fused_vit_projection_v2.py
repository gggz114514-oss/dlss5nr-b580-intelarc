"""Parallel independent ViT K partitions, then ordered half merge.

V1's serial long kernel regressed at64tokens. Keep each partition as independent
GPU work in a shared grid, then combine its complete rounded half results.
"""
import torch
import triton
import triton.language as tl
from nr_backend.triton_attention_normalize import _nan_left
from nr_backend.execution import current_arithmetic_backend,record_arithmetic_dispatch
from fused_vit_projection_v1 import FusedVitProjection as Base


@triton.jit
def _parts(X,W,INITIAL,OUT,M:tl.constexpr,K:tl.constexpr,N:tl.constexpr,
           BM:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*BN+tl.arange(0,BN)
    part=tl.program_id(2)
    lane=tl.arange(0,32)
    total=tl.full((BM,BN),0.,tl.float32)
    for block in range(K//128):
        k=part*(K//4)+block*32+lane
        x=tl.load(X+row[:,None]*K+k[None,:],row[:,None]<M,other=0)
        w=tl.load(W+k[:,None]*N+col[None,:],col[None,:]<N,other=0)
        total=tl.dot(x,w,total,out_dtype=tl.float32)
    offset=row[:,None]*N+col[None,:]
    valid=(row[:,None]<M)&(col[None,:]<N)
    if part==0:
        total=total+tl.load(INITIAL+offset,valid,other=0).to(tl.float32)
    tl.store(OUT+part*M*N+offset,total.to(tl.float16),valid)


@triton.jit
def _merge(X,OUT,N:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    value=tl.load(X+i,i<N,other=0)
    for part in tl.static_range(1,4):
        other=tl.load(X+part*N+i,i<N,other=0)
        value=_nan_left((value.to(tl.float32)+other.to(tl.float32)).to(tl.float16),value,other)
    tl.store(OUT+i,value,i<N)


def forward(features,weight,initial,*,bm=16,bn=32,stages=1):
    if features.ndim!=2 or features.shape[0]<=0 or features.shape[1] not in (1024,4096):
        raise ValueError('Expected original ViT projection K1024 or K4096')
    m,k=features.shape
    if weight.shape!=(k,1024) or initial.shape!=(m,1024):
        raise ValueError('Expected original N1024 weight and initial half skip')
    for t in (features,weight,initial):
        if t.device!=features.device or t.device.type!='xpu' or t.dtype!=torch.float16:
            raise ValueError('Expected same-device half XPU operands')
    if not weight.is_contiguous():raise ValueError('Expected owned contiguous weight')
    if bm not in (16,32) or bn not in (32,64) or stages not in (1,2):
        raise ValueError('Unsupported projection tile')
    x=features.contiguous();initial=initial.contiguous()
    partial=torch.empty((4,m,1024),dtype=features.dtype,device=features.device)
    out=torch.empty((m,1024),dtype=features.dtype,device=features.device)
    kernel=_parts[(triton.cdiv(m,bm),triton.cdiv(1024,bn),4)](
        x,weight,initial,partial,m,k,1024,bm,bn,num_warps=4,num_stages=stages,enable_fp_fusion=False)
    merge=_merge[(triton.cdiv(m*1024,512),)](partial,out,m*1024,512,num_warps=4,enable_fp_fusion=False)
    return out,(kernel,merge)


class FusedVitProjection(Base):
    def apply(self,features,weight,initial,parts=4):
        if self.weights.get(id(weight)) is not weight or self.provider.mode!='fp16_xmx' or current_arithmetic_backend()!='triton' or parts!=4:
            return self.original(features,weight,initial,parts)
        result,_=forward(features,weight,initial,**self.options)
        for _ in range(4):record_arithmetic_dispatch('dense')
        self.calls+=1
        return result
