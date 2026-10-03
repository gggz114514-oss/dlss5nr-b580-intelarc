"""Four original K partitions and sequential half merges in one FP16 kernel.

Every partition starts an independent FP32 K32 dot accumulation. Partition zero
includes the original half skip before its first half store; later partitions
round independently to half before merging into the running half result.
"""
from contextlib import contextmanager
import torch
import triton
import triton.language as tl
import nr_backend.vit_block as vit
from nr_backend.triton_attention_normalize import _nan_left
from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch


@triton.jit
def _kernel(X,W,INITIAL,OUT,M:tl.constexpr,K:tl.constexpr,N:tl.constexpr,
            BM:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*BN+tl.arange(0,BN)
    lane=tl.arange(0,32)
    offset=row[:,None]*N+col[None,:]
    valid=(row[:,None]<M)&(col[None,:]<N)
    initial=tl.load(INITIAL+offset,valid,other=0)
    result=tl.full((BM,BN),0.,tl.float16)
    for part in range(4):
        total=tl.full((BM,BN),0.,tl.float32)
        for block in range(K//128):
            k=part*(K//4)+block*32+lane
            x=tl.load(X+row[:,None]*K+k[None,:],row[:,None]<M,other=0)
            w=tl.load(W+k[:,None]*N+col[None,:],col[None,:]<N,other=0)
            total=tl.dot(x,w,total,out_dtype=tl.float32)
        if part==0:
            result=(total+initial.to(tl.float32)).to(tl.float16)
        else:
            value=total.to(tl.float16)
            result=_nan_left((result.to(tl.float32)+value.to(tl.float32)).to(tl.float16),result,value)
    tl.store(OUT+offset,result,valid)


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
    out=torch.empty((m,1024),dtype=features.dtype,device=features.device)
    kernel=_kernel[(triton.cdiv(m,bm),triton.cdiv(1024,bn))](
        x,weight,initial,out,m,k,1024,bm,bn,num_warps=4,num_stages=stages,enable_fp_fusion=False)
    return out,kernel


class FusedVitProjection:
    """Scoped owned-weight override of the FP16 four-way projection only."""
    def __init__(self,model,provider,*,bm=16,bn=32,stages=1):
        self.weights={id(w):w for m in model.modules() if type(m) is vit.VitBlock for w in (m.contract,m.projection)}
        self.provider=provider
        self.original=vit.split_k_projection
        self.options=dict(bm=bm,bn=bn,stages=stages)
        self.calls=0

    def apply(self,features,weight,initial,parts=4):
        if self.weights.get(id(weight)) is not weight or self.provider.mode!='fp16_xmx' or current_arithmetic_backend()!='triton' or parts!=4:
            return self.original(features,weight,initial,parts)
        result,_=forward(features,weight,initial,**self.options)
        for _ in range(4):record_arithmetic_dispatch('dense')
        self.calls+=1
        return result

    @contextmanager
    def installed(self):
        assert vit.split_k_projection is self.original
        replacement=self.apply
        vit.split_k_projection=replacement
        try:yield self
        finally:
            assert vit.split_k_projection is replacement
            vit.split_k_projection=self.original
