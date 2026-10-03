"""Fuse an existing following FP8 boundary into complete projection stores.

Keep all dot order, half residual/branch boundaries and model weights. The
branched replacement is entered only through BranchedMLP.forward; its public
unquantized method retains its original behavior. C512 pooling and boundary
probes keep the original unquantized projection.
"""
from contextlib import contextmanager
import torch
import triton
import triton.language as tl
import nr_backend.multihead_block as multi
import batched_branched_mlp_v1 as batched
import native_cubic_batched_v1 as native
import c512_window_layout_scope_v1 as c512
from nr_backend.execution import current_arithmetic_backend
from quantization_dataflow_v1 import CONTRACTS
from short_fp8_v2 import round_half
from spill_preflight_v1 import select

EXTRA = {
    'quantized_projection_store_v1._branched': (('OUT',), ('OUT',)),
    'quantized_projection_store_v1._c512': (('OUT',), ('OUT',)),
}


@triton.jit
def _branched(X, LATENT, PROJECT, SCALE, OUT, M:tl.constexpr, C:tl.constexpr,
              BM:tl.constexpr, BN:tl.constexpr):
    row=tl.program_id(0)*BM+tl.arange(0,BM)
    column=tl.program_id(1)*BN+tl.arange(0,BN)
    lane=tl.arange(0,32)
    valid=(row[:,None]<M)&(column[None,:]<C)
    x=tl.load(X+row[:,None]*C+column[None,:],valid,other=0)
    scale=tl.load(SCALE+column,column<C,other=0)
    result=(x.to(tl.float32)*scale[None,:].to(tl.float32)).to(tl.float16)
    for branch in range(C//32):
        hidden=tl.load(LATENT+(branch*M+row[:,None])*32+lane[None,:],row[:,None]<M,other=0)
        weight=tl.load(PROJECT+(branch*32+lane[:,None])*C+column[None,:],column[None,:]<C,other=0)
        projected=tl.dot(hidden,weight,out_dtype=tl.float32)
        result=(projected+result.to(tl.float32)).to(tl.float16)
    tl.store(OUT+row[:,None]*C+column[None,:],round_half(result),valid)


@triton.jit
def _c512(X,W,RESIDUAL,SCALE,INVERSE,OUT,HP:tl.constexpr,WP:tl.constexpr,
          H:tl.constexpr,WIDTH:tl.constexpr,SY:tl.constexpr,SX:tl.constexpr,
          BM:tl.constexpr,BN:tl.constexpr):
    rows=tl.program_id(0)*BM+tl.arange(0,BM)
    cols=tl.program_id(1)*BN+tl.arange(0,BN);kk=tl.arange(0,32)
    y=rows//WIDTH+SY;x=rows%WIDTH+SX
    window=(y//8)*(WP//8)+x//8
    local=tl.load(INVERSE+(y%8)*8+x%8).to(tl.int32)
    total=tl.full((BM,BN),0,tl.float32)
    for head in range(16):
        av=tl.load(X+(head*(HP//8)*(WP//8)+window[:,None])*2048+local[:,None]*32+kk[None,:],rows[:,None]<H*WIDTH,other=0)
        wv=tl.load(W+(head*32+kk[:,None])*512+cols[None,:],cols[None,:]<512,other=0)
        total=tl.dot(av,wv,total,out_dtype=tl.float32)
    offset=rows[:,None]*512+cols[None,:]
    valid=(rows[:,None]<H*WIDTH)&(cols[None,:]<512)
    r=tl.load(RESIDUAL+offset,valid,other=0);s=tl.load(SCALE+cols,cols<512,other=0)
    initial=(r.to(tl.float32)*s[None,:].to(tl.float32)).to(tl.float16)
    result=(total+initial.to(tl.float32)).to(tl.float16)
    tl.store(OUT+offset,round_half(result),valid)


class ProjectionStores:
    def __init__(self,stack):
        self.stack=stack;self.model=stack.model;self.provider=stack.provider
        self.modules={id(m) for m in self.model.modules() if type(m) is multi.BranchedMLP}
        self.original=multi.BranchedMLP.forward
        self.resources={};self.selections={};self.calls={};self.depth=0

    def launch(self,label,jit,configs,args_for,grid_for):
        options=dict(num_warps=4,num_stages=1,enable_fp_fusion=False)
        config,kernel,decision=select(jit,configs,args_for,grid_for,**options)
        self.resources[label+':'+kernel.hash]=dict(spills=kernel.n_spills,registers=kernel.n_regs,shared_bytes=kernel.metadata.shared)
        self.selections[label]=decision
        assert jit[grid_for(config)](*args_for(config),**options) is kernel
        self.calls[label]=self.calls.get(label,0)+1
        return kernel,decision

    def __getitem__(self,grid):
        # Compatibility shim for the two frozen batched _project call sites.
        def run(*args,**options):
            assert self.depth==1 and len(args)==9
            assert options==dict(num_warps=4,num_stages=1,enable_fp_fusion=False)
            x,latent,weight,scale,out,m,c,bm,bn=args
            assert c in (64,128,256) and bm==16 and bn==64
            assert m>0 and all(t.device.type=='xpu' and t.dtype==torch.float16 and t.is_contiguous() for t in (x,latent,weight,scale,out))
            assert tuple(grid)==(triton.cdiv(m,bm),triton.cdiv(c,bn))
            label=f'branched_{m}x{c}'
            # Try the existing tile first, then bounded smaller output tiles.
            return self.launch(label,_branched,[(bm,bn),(16,32),(16,16)],
                lambda cfg:(x,latent,weight,scale,out,m,c,*cfg),
                lambda cfg:(triton.cdiv(m,cfg[0]),triton.cdiv(c,cfg[1])))[0]
        return run

    def forward(self,module,features):
        if (id(module) not in self.modules or current_arithmetic_backend()!='triton'
                or self.provider.mode!='fp16_xmx' or features.device.type!='xpu'):
            return self.original(module,features)
        assert self.depth==0
        previous=(batched._project,native._project)
        self.depth=1;batched._project=self;native._project=self
        try:
            return self.original(module,features)
        finally:
            valid=batched._project is self and native._project is self
            batched._project,native._project=previous;self.depth=0
            assert valid,'Projection binding interference'

    def c512_project(self,windows,weight,residual,scale,inverse,*,shift):
        assert tuple(residual.shape)==(12,12,512) and tuple(weight.shape)==(512,512)
        assert tuple(scale.shape)==(512,) and tuple(inverse.shape)==(64,)
        assert windows.ndim==5 and windows.shape[0]==16 and tuple(windows.shape[-2:])==(64,32)
        assert all(t.dtype==torch.float16 and t.device==windows.device and t.device.type=='xpu' and t.is_contiguous() for t in (windows,weight,residual,scale))
        assert inverse.dtype==torch.int64 and inverse.device==windows.device and inverse.is_contiguous()
        sy,sx=shift;assert sy in (0,4) and sx in (0,4)
        hp,wp=windows.shape[1]*8,windows.shape[2]*8
        assert hp>=12+sy and wp>=12+sx
        out=torch.empty_like(residual)
        kernel,decision=self.launch('c512_'+str(shift),_c512,[(16,32),(16,16)],
            lambda cfg:(windows,weight,residual,scale,inverse,out,hp,wp,12,12,sy,sx,*cfg),
            lambda cfg:(triton.cdiv(144,cfg[0]),triton.cdiv(512,cfg[1])))
        return out,kernel,decision

    @contextmanager
    def installed(self):
        assert multi.BranchedMLP.forward is self.original and not any(k in CONTRACTS for k in EXTRA)
        replacement=lambda module,features:self.forward(module,features)
        CONTRACTS.update(EXTRA);multi.BranchedMLP.forward=replacement
        try:
            yield self
        finally:
            valid=multi.BranchedMLP.forward is replacement and self.depth==0
            multi.BranchedMLP.forward=self.original
            for k,v in EXTRA.items():assert CONTRACTS.pop(k)==v
            assert valid,'Quantized projection scope interference'

    def verify_restored(self):
        assert multi.BranchedMLP.forward is self.original and self.depth==0
        assert not any(k in CONTRACTS for k in EXTRA)

    def metadata(self):
        return dict(resources=self.resources,selections=self.selections,calls=self.calls,
            arithmetic='original dot and intermediate half order; same final FP8 boundary moved into store',
            unquantized_branched_api_unchanged=True,c512_pooling_unquantized=True)


class C512QuantizedLayout(c512.C512WindowLayout):
    def __init__(self,stack,stores):
        super().__init__(stack);self.stores=stores;self.quantized_calls=0;self.pooling_calls=0

    def split(self,module,features):
        # The last encoder block pools the raw half projection. Never round it early.
        if module.final_weight is not None or self.probe is not None:
            self.pooling_calls+=1
            return super().split(module,features)
        previous=c512.project;replacement=self.stores.c512_project;c512.project=replacement
        try:
            result=super().split(module,features);self.quantized_calls+=1
            return result
        finally:
            valid=c512.project is replacement;c512.project=previous
            assert valid,'C512 projection binding interference'
