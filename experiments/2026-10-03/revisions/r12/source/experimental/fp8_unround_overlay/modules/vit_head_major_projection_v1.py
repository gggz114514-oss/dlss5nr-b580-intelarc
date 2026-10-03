"""Untested ViT consumer prototype: read attention heads without a layout copy.

Consumes contiguous [32 heads,64 tokens,32 channels] directly. Preserves the
four K partitions, inner K32 order and ordered half merge. Not selected until
complete boundary, whole-body and complete-call comparisons pass.
"""
import torch
import triton
import triton.language as tl
from fused_vit_projection_v2 import _merge


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
        # Logical [token,head*32+channel] in the producer's physical layout.
        offset=(k[None,:]//32)*M*32+row[:,None]*32+k[None,:]%32
        x=tl.load(X+offset,row[:,None]<M,other=0)
        w=tl.load(W+k[:,None]*N+col[None,:],col[None,:]<N,other=0)
        total=tl.dot(x,w,total,out_dtype=tl.float32)
    offset=row[:,None]*N+col[None,:]
    valid=(row[:,None]<M)&(col[None,:]<N)
    if part==0:
        total=total+tl.load(INITIAL+offset,valid,other=0).to(tl.float32)
    tl.store(OUT+part*M*N+offset,total.to(tl.float16),valid)


def forward(heads,weight,initial):
    for t,shape in ((heads,(32,64,32)),(weight,(1024,1024)),(initial,(64,1024))):
        if (tuple(t.shape)!=shape or t.device.type!='xpu' or t.device!=heads.device
                or t.dtype!=torch.float16 or not t.is_contiguous()):
            raise ValueError('Expected owned contiguous half ViT operands')
    partial=torch.empty((4,64,1024),dtype=heads.dtype,device=heads.device)
    out=torch.empty((64,1024),dtype=heads.dtype,device=heads.device)
    args=(heads,weight,initial,partial,64,1024,1024,32,32)
    grid=(2,32,4)
    options=dict(num_warps=4,num_stages=1,enable_fp_fusion=False)
    kernel=_parts.warmup(*args,grid=grid,**options)
    kernel._init_handles()
    if not isinstance(kernel.n_spills,int) or kernel.n_spills!=0:
        raise RuntimeError('Head-major projection spill rejected before launch: '+str(kernel.n_spills))
    _parts[grid](*args,**options)
    merge=_merge[(128,)](partial,out,64*1024,512,num_warps=4,enable_fp_fusion=False)
    return out,(kernel,merge)
