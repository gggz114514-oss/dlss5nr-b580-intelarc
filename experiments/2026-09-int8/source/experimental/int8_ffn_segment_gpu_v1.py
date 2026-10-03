"""Three/four-kernel approximate INT8 FFN; no intermediate FP8/half hidden store.

Only the observed M64/K1024/H4096/N1024 contract. Static folded weights and hidden
scales belong to the caller; dynamic entry quantization is included in every run.
Debug buffers are written only by separate, untimed debug specializations.
"""
import torch
import triton
import triton.language as tl
from short_fp8_v2 import round_half
from spill_preflight_v1 import select


@triton.jit
def _q(value,scale):
    unit=tl.div_rn(value,scale)
    rounded=tl.floor(tl.abs(unit)+.5)*tl.where(unit<0,-1.,1.)
    return tl.minimum(tl.maximum(rounded,-127.),127.).to(tl.int8)


@triton.jit
def _entry(X,Q,S,M:tl.constexpr,K:tl.constexpr):
    row=tl.program_id(0);k=tl.arange(0,K)
    value=tl.load(X+row*K+k).to(tl.float32)
    maximum=tl.max(tl.abs(value),0)
    scale=tl.where(maximum>0,tl.div_rn(maximum,127.),1.)
    tl.store(Q+row*K+k,_q(value,scale));tl.store(S+row,scale)


@triton.jit
def _expand(QX,SX,W,SW,SH,QH,HRAW,M:tl.constexpr,K:tl.constexpr,H:tl.constexpr,
            BM:tl.constexpr,BN:tl.constexpr,DEBUG:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*BN+tl.arange(0,BN);lane=tl.arange(0,32)
    total=tl.full((BM,BN),0,tl.int32)
    for start in range(K//32):
        k=start*32+lane
        a=tl.load(QX+row[:,None]*K+k[None,:],row[:,None]<M,other=0)
        w=tl.load(W+col[None,:]*K+k[:,None],col[None,:]<H,other=0)
        total=tl.dot(a,w,total,out_dtype=tl.int32)
    sa=tl.load(SX+row,row<M,other=1.)
    sw=tl.load(SW+col,col<H,other=1.)
    x=((total.to(tl.float32)*sa[:,None])*sw[None,:]).to(tl.float16)
    t=tl.minimum(tl.maximum(x.to(tl.float32),-4.),4.).to(tl.float16)
    p=tl.fma(-tl.abs(t),tl.full((),.055908203125,tl.float16),tl.full((),.447265625,tl.float16)).to(tl.float16)
    v=tl.fma(t,p,tl.full((),.89453125,tl.float16)).to(tl.float16)
    hidden=(x.to(tl.float32)*v.to(tl.float32)).to(tl.float16)
    sh=tl.load(SH+col,col<H,other=1.)
    offset=row[:,None]*H+col[None,:];valid=(row[:,None]<M)&(col[None,:]<H)
    tl.store(QH+offset,_q(hidden.to(tl.float32),sh[None,:]),valid)
    if DEBUG:tl.store(HRAW+offset,hidden,valid)


@triton.jit
def _finish(total,sw,x,skip):
    initial=(x.to(tl.float32)*skip.to(tl.float32)).to(tl.float16)
    raw=(total.to(tl.float32)*sw+initial.to(tl.float32)).to(tl.float16)
    return raw,round_half(raw)


@triton.jit
def _contract(QH,W,SW,X,SKIP,PARTIAL,OUT,RAW,M:tl.constexpr,H:tl.constexpr,N:tl.constexpr,
              PARTS:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,DEBUG:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    col=tl.program_id(1)*BN+tl.arange(0,BN);lane=tl.arange(0,32)
    part=tl.program_id(2)
    total=tl.full((BM,BN),0,tl.int32)
    for start in range(H//(PARTS*32)):
        k=part*(H//PARTS)+start*32+lane
        a=tl.load(QH+row[:,None]*H+k[None,:],row[:,None]<M,other=0)
        w=tl.load(W+col[None,:]*H+k[:,None],col[None,:]<N,other=0)
        total=tl.dot(a,w,total,out_dtype=tl.int32)
    offset=row[:,None]*N+col[None,:];valid=(row[:,None]<M)&(col[None,:]<N)
    if PARTS==1:
        sw=tl.load(SW+col,col<N,other=1.)
        skip=tl.load(SKIP+col,col<N,other=0)
        x=tl.load(X+offset,valid,other=0)
        raw,out=_finish(total,sw[None,:],x,skip[None,:])
        tl.store(OUT+offset,out,valid)
        if DEBUG:tl.store(RAW+offset,raw,valid)
    else:
        tl.store(PARTIAL+part*M*N+offset,total,valid)


@triton.jit
def _merge(PARTIAL,SW,X,SKIP,OUT,RAW,M:tl.constexpr,N:tl.constexpr,DEBUG:tl.constexpr):
    i=tl.program_id(0)*512+tl.arange(0,512);valid=i<M*N
    total=tl.load(PARTIAL+i,valid,other=0)
    for part in tl.static_range(1,4):total+=tl.load(PARTIAL+part*M*N+i,valid,other=0)
    sw=tl.load(SW+i%N,valid,other=1.);skip=tl.load(SKIP+i%N,valid,other=0)
    raw,out=_finish(total,sw,tl.load(X+i,valid,other=0),skip)
    tl.store(OUT+i,out,valid)
    if DEBUG:tl.store(RAW+i,raw,valid)


class Segment:
    def __init__(self,x,wexpand,sexpand,sh,wcontract,scontract,skip):
        for t,shape,dtype in (
            (x,(64,1024),torch.float16),(wexpand,(4096,1024),torch.int8),
            (sexpand,(4096,),torch.float32),(sh,(4096,),torch.float32),
            (wcontract,(1024,4096),torch.int8),(scontract,(1024,),torch.float32),
            (skip,(1024,),torch.float16)):
            assert tuple(t.shape)==shape and t.dtype==dtype and t.device==x.device and t.device.type=='xpu' and t.is_contiguous()
        self.qx=torch.empty((64,1024),device=x.device,dtype=torch.int8)
        self.sx=torch.empty(64,device=x.device,dtype=torch.float32)
        self.qh=torch.empty((64,4096),device=x.device,dtype=torch.int8)
        self.hidden_raw=torch.empty((64,4096),device=x.device,dtype=torch.float16)
        self.partial=torch.empty((4,64,1024),device=x.device,dtype=torch.int32)
        self.outputs={p:torch.empty_like(x) for p in (1,4)}
        self.raw_outputs={p:torch.empty_like(x) for p in (1,4)}
        self.plans={};self.resources={};self.selections={}

        def bind(label,jit,configs,args_for,grid_for):
            config,kernel,selection=select(jit,configs,args_for,grid_for,
                num_warps=4,num_stages=1,enable_fp_fusion=False)
            args,grid=args_for(config),grid_for(config)
            self.resources[label]=dict(hash=kernel.hash,spills=kernel.n_spills,
                registers=kernel.n_regs,shared_bytes=kernel.metadata.shared)
            self.selections[label]=selection
            def run():
                actual=jit[grid](*args,num_warps=4,num_stages=1,enable_fp_fusion=False)
                assert actual is kernel
            return run
        self.entry=bind('entry',_entry,[(1,)],lambda c:(x,self.qx,self.sx,64,1024),lambda c:(64,))
        for debug in (False,True):
            expand=bind('expand_debug' if debug else 'expand',_expand,[(32,32),(16,32)],
                lambda c:(self.qx,self.sx,wexpand,sexpand,sh,self.qh,self.hidden_raw,64,1024,4096,*c,debug),
                lambda c:(triton.cdiv(64,c[0]),triton.cdiv(4096,c[1])))
            for parts in (1,4):
                contract=bind(f'contract_p{parts}_debug{debug}',_contract,[(32,32),(16,32)],
                    lambda c:(self.qh,wcontract,scontract,x,skip,self.partial,self.outputs[parts],self.raw_outputs[parts],64,4096,1024,parts,*c,debug),
                    lambda c:(triton.cdiv(64,c[0]),triton.cdiv(1024,c[1]),parts))
                steps=[self.entry,expand,contract]
                if parts==4:
                    steps.append(bind(f'merge_debug{debug}',_merge,[(512,)],
                        lambda c:(self.partial,scontract,x,skip,self.outputs[4],self.raw_outputs[4],64,1024,debug),
                        lambda c:(128,)))
                self.plans[parts,debug]=steps

    def run(self,parts,debug=False):
        for fn in self.plans[parts,debug]:fn()
        return self.outputs[parts]
