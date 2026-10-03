"""Complete current720 continuous INT8 FFN candidate, frozen quantization.

C512 entry/linear/register-fused eight groups/project removes QH scratch.
ViT entry/expand/full-I32-K4096 contract removes P4 scratch and merge.
Optional fused-entry ViT expand keeps K32 QX/SX in registers, never stores half
hidden or adds an INT8->FP16->INT8 tensor boundary. Its duplicated row-scale
work and register pressure are measured obligations, not a speed claim.
"""
from __future__ import annotations
import torch
import triton
import triton.language as tl
import c512_int8_ffn_rows_v1 as c512
import int8_ffn_segment_rows_v1 as vit
from int8_ffn_segment_gpu_v1 import _q
from c512_int8_ffn_rows_v1 import _cubic


@triton.jit
def _c512_project(QG,W,SW,X,SKIP,OUT,M:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
    # Same accepted complete epilogue, with an honest unrounded output contract.
    c512._project(QG,W,SW,X,SKIP,OUT,X,M,BM,BN,False,False)


@triton.jit
def _vit_merge_p4(PARTIAL,SW,X,SKIP,OUT):
    vit._merge(PARTIAL,SW,X,SKIP,OUT,X,240,1024,False,False)


@triton.jit
def _vit_expand_entry(X,W,SW,SH,QH,BM:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*BN+tl.arange(0,BN)
    kfull=tl.arange(0,1024);lane=tl.arange(0,32)
    x=tl.load(X+row[:,None]*1024+kfull[None,:],row[:,None]<240,other=0).to(tl.float32)
    maximum=tl.max(tl.abs(x),1)
    scale=tl.where(maximum>0,tl.div_rn(maximum,127.),1.)
    # The full row is used only for its unchanged global maximum. Retaining
    # BMx1024 QX across all 32 dots caused actual positive spills in r7.
    # Quantize the same half input with the same row scale at each K32 step.
    total=tl.full((BM,BN),0,tl.int32)
    for block in range(32):
        kk=block*32+lane
        chunk=tl.load(X+row[:,None]*1024+kk[None,:],row[:,None]<240,other=0).to(tl.float32)
        a=_q(chunk,scale[:,None])
        w=tl.load(W+col[None,:]*1024+kk[:,None])
        total=tl.dot(a,w,total,out_dtype=tl.int32)
    sw=tl.load(SW+col);sh=tl.load(SH+col)
    value=((total.to(tl.float32)*scale[:,None])*sw[None,:]).to(tl.float16)
    hidden=_cubic(value)
    tl.store(QH+row[:,None]*4096+col[None,:],
             _q(hidden.to(tl.float32),sh[None,:]),row[:,None]<240)


@triton.jit
def _vit_contract_full(QH,W,SW,X,SKIP,OUT,BM:tl.constexpr,BN:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*BN+tl.arange(0,BN);lane=tl.arange(0,32)
    total=tl.full((BM,BN),0,tl.int32)
    for block in range(128):
        kk=block*32+lane
        a=tl.load(QH+row[:,None]*4096+kk[None,:],row[:,None]<240,other=0)
        w=tl.load(W+col[None,:]*4096+kk[:,None])
        total=tl.dot(a,w,total,out_dtype=tl.int32)
    offset=row[:,None]*1024+col[None,:]
    x=tl.load(X+offset,row[:,None]<240,other=0)
    sw=tl.load(SW+col);skip=tl.load(SKIP+col)
    initial=(x.to(tl.float32)*skip[None,:].to(tl.float32)).to(tl.float16)
    result=(total.to(tl.float32)*sw[None,:]+initial.to(tl.float32)).to(tl.float16)
    tl.store(OUT+offset,result,row[:,None]<240)


def _require(x,shape):
    if (tuple(x.shape)!=shape or x.dtype!=torch.float16 or
            x.device.type!="xpu" or not x.is_contiguous()):
        raise ValueError("Current720 complete FFN requires contiguous actual rows")


def c512_forward(stack,name,features,counter,*,bm=32,bn=64):
    from nr_backend.unround_policy import ENABLED
    from nr_backend.execution import record_arithmetic_dispatch
    _require(features,(24,40,512))
    if "c512" not in ENABLED:raise RuntimeError("Exact C512 path is not a candidate")
    scope=stack.c512_int8
    w0,s0,sz,we,se,sh,wr,sr,sg,wp,sp,skip=scope.packed[name]
    x=features.view(960,512)
    qx=torch.empty((960,512),device=x.device,dtype=torch.int8)
    sx=torch.empty((960,),device=x.device,dtype=torch.float32)
    qz=torch.empty_like(qx);qg=torch.empty_like(qx);out=torch.empty_like(x)
    counter.launch("c512_ffn_entry",c512._entry,(960,),(x,qx,sx,960,False))
    counter.launch("c512_ffn_linear",c512._linear,(triton.cdiv(960,bm),512//bn),
                   (qx,sx,w0,s0,sz,qz,x,960,bm,bn,False))
    # Existing dormant fused groups JIT now routed with actual960 rows. QH
    # quantization stays in INT8 registers and is consumed immediately by reduce.
    counter.launch("c512_ffn_groups",c512._groups,(triton.cdiv(960,bm),8,64//bn),
                   (qz,we,se,sh,wr,sr,sg,qg,x,960,bm,bn,False))
    counter.launch("c512_ffn_project",_c512_project,(triton.cdiv(960,bm),512//bn),
                   (qg,wp,sp,x,skip,out,960,bm,bn))
    scope.ffn_calls+=1
    scope.ffn_blocks[name]=scope.ffn_blocks.get(name,0)+1
    record_arithmetic_dispatch("c512_int8_ffn_segment")
    if scope.ffn_probe is not None:
        # Explicit diagnostic only. Reconstruct QH by the same frozen expand
        # without changing the timed/default fused producer-consumer path.
        qh=torch.empty((960,2048),device=x.device,dtype=torch.int8)
        counter.launch("c512_ffn_debug_qh",c512._expand,(triton.cdiv(960,bm),8,256//bn),
                       (qz,we,se,sh,qh,x,960,bm,bn,False))
        scope.ffn_probe(name,x,dict(qx=qx,sx=sx,qz=qz,qh=qh,qg=qg,out=out))
    return out.view(24,40,512)


def vit_forward(stack,index,x,counter,*,bm=32,bn=64,fuse_entry=False,parts=1):
    from nr_backend.unround_policy import ENABLED
    _require(x,(240,1024))
    if "vit" not in ENABLED:raise RuntimeError("Exact ViT path is not a candidate")
    if parts not in (1,4):raise ValueError("Only full-I32 or original P4 contract")
    scope=stack.int8_vit
    we,se,sh,wc,sc,skip=scope.packed[index]
    qh=torch.empty((240,4096),device=x.device,dtype=torch.int8)
    out=torch.empty_like(x)
    if fuse_entry:
        # Smaller BM caps the wide1024 register row-quantization tile.
        counter.launch("vit_ffn_entry_expand",_vit_expand_entry,
                       (triton.cdiv(240,16),4096//bn),(x,we,se,sh,qh,16,bn))
    else:
        qx=torch.empty((240,1024),device=x.device,dtype=torch.int8)
        sx=torch.empty((240,),device=x.device,dtype=torch.float32)
        counter.launch("vit_ffn_entry",vit._entry,(240,),(x,qx,sx,240,1024))
        counter.launch("vit_ffn_expand",vit._expand,
                       (triton.cdiv(240,bm),4096//bn),
                       (qx,sx,we,se,sh,qh,x,240,1024,4096,bm,bn,False))
    if parts==1:
        counter.launch("vit_ffn_contract_full",_vit_contract_full,
                       (triton.cdiv(240,bm),1024//bn),(qh,wc,sc,x,skip,out,bm,bn))
    else:
        partial=torch.empty((4,240,1024),device=x.device,dtype=torch.int32)
        counter.launch("vit_ffn_contract_p4",vit._contract,
                       (triton.cdiv(240,bm),1024//bn,4),
                       (qh,wc,sc,x,skip,partial,out,x,240,4096,1024,4,bm,bn,False,False))
        counter.launch("vit_ffn_merge",_vit_merge_p4,(480,),
                       (partial,sc,x,skip,out))
    # QH remains the real contract input; tuple API is preserved.
    return out,qh


EXTRA={
    __name__+"._c512_project":(("OUT",),()),
    __name__+"._vit_merge_p4":(("OUT",),()),
    __name__+"._vit_expand_entry":(("QH",),()),
    __name__+"._vit_contract_full":(("OUT",),()),
    "c512_int8_ffn_rows_v1._groups":(("QG",),()),
}
