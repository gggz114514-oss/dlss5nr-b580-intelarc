"""Keep producer packing but specialize the native expansion and load merge rows.

V1 completed the full output gate but saved only 0.18% continuous host time.
This candidate removes runtime matrix-shape arithmetic from the native kernel
and loads contiguous projection partials, then scatters the private companion.
"""
import torch
import triton
import triton.language as tl
from nr_backend.triton_fp8 import _round_fp8_half
from nr_backend.triton_attention_normalize import _nan_left
from nr_backend.execution import record_arithmetic_dispatch
from fused_vit_projection_v2 import _parts
from esimd_vit_chain_v1 import VitChain as Base,D
from esimd_dense_adapter_v2 import Dense
from spill_preflight_v1 import select


@triton.jit
def _merge_rows(X,OUT,PACK,M:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<M*K
    value=tl.load(X+i,valid,other=0)
    for part in tl.static_range(1,4):
        other=tl.load(X+part*M*K+i,valid,other=0)
        value=_nan_left((value.to(tl.float32)+other.to(tl.float32)).to(tl.float16),value,other)
    value=_round_fp8_half(value)
    row=i//K;col=i%K
    packed=(row//8)*K*8+(col//16)*128+(row%8)*16+col%16
    tl.store(OUT+i,value,valid);tl.store(PACK+packed,value,valid)


CONTRACTS={'esimd_vit_chain_v2._merge_rows':(('OUT','PACK'),('OUT','PACK'))}


class VitChain(Base):
    def __init__(self,model,provider):
        super().__init__(model,provider)
        self.engine=Dense(D/'experimental/esimd-vit-expand-build-v1/esimd_vit_expand_v1.dll')
        self.engine.require(True)

    def final_projection(self,features,weight,initial):
        m,k=features.shape
        if (m,k)!=(64,1024) or weight.shape!=(1024,1024) or initial.shape!=(64,1024):
            raise ValueError('Unvalidated final ViT projection')
        x=features.contiguous();ini=initial.contiguous()
        partial=torch.empty((4,m,1024),device=x.device,dtype=x.dtype)
        out=torch.empty((m,1024),device=x.device,dtype=x.dtype)
        packed=torch.empty((m//8,64,8,16),device=x.device,dtype=x.dtype)
        _parts[(triton.cdiv(m,32),32,4)](x,weight,ini,partial,m,k,1024,32,32,num_warps=4,num_stages=1,enable_fp_fusion=False)
        args=(partial,out,packed,m,1024,512);grid=(triton.cdiv(m*1024,512),)
        if 'merge' not in self.selection:
            _,_,self.selection['merge']=select(_merge_rows,[(512,)],lambda _:args,lambda _:grid,num_warps=4,enable_fp_fusion=False)
        _merge_rows[grid](*args,num_warps=4,enable_fp_fusion=False)
        for _ in range(4):record_arithmetic_dispatch('dense')
        record_arithmetic_dispatch('fp8')
        return out,packed
