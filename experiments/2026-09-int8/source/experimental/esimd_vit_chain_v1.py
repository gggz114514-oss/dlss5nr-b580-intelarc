"""Eight owned ViT blocks with producer-written ESIMD activation layout.

Only the no-initial 1024->4096 expansion uses native DPAS. The existing four
partition projection kernel and ordered half merge remain. Final FP8 rounding
writes both the public row-major value and a private blocked companion used by
the next block. Companions are local to one complete body invocation.
"""
from pathlib import Path
import hashlib
import torch
import triton
import triton.language as tl
from nr_backend.triton_fp8 import _round_fp8_half
from nr_backend.triton_attention_normalize import _nan_left
from nr_backend.execution import record_arithmetic_dispatch
import nr_backend.vit_block as vit
from fused_vit_projection_v2 import _parts
from fused_vit_qkv_v1 import forward as prepare
from fused_vit_attention64_v2 import forward as attend
from esimd_dense_adapter_v2 import Dense
from spill_preflight_v1 import select

D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
DLL=D/'experimental/esimd-dense-build-v4/esimd_dense_v4.dll'
RECEIPT=D/'experimental/esimd-dense-v3/validation.json'


@triton.jit
def _entry_pack(X,P,M:tl.constexpr,K:tl.constexpr):
    tile=tl.program_id(0);i=tl.arange(0,128)
    rb=tile//(K//16);kb=tile%(K//16)
    row=rb*8+i//16;col=kb*16+i%16
    x=tl.load(X+row*K+col,row<M,other=0)
    tl.store(P+tile*128+i,_round_fp8_half(x))


@triton.jit
def _merge_packed(X,OUT,PACK,M:tl.constexpr,K:tl.constexpr,B:tl.constexpr):
    # Iterate in packed order: contiguous PACK store and grouped row-major OUT.
    p=tl.program_id(0)*B+tl.arange(0,B)
    rb=p//(K*8);rem=p%(K*8)
    kb=rem//128;inner=rem%128
    row=rb*8+inner//16;col=kb*16+inner%16
    index=row*K+col;valid=row<M
    value=tl.load(X+index,valid,other=0)
    for part in tl.static_range(1,4):
        other=tl.load(X+part*M*K+index,valid,other=0)
        value=_nan_left((value.to(tl.float32)+other.to(tl.float32)).to(tl.float16),value,other)
    rounded=_round_fp8_half(value)
    tl.store(OUT+index,rounded,valid)
    tl.store(PACK+p,rounded,p<((M+7)//8)*8*K)


CONTRACTS={
    'esimd_vit_chain_v1._entry_pack':(('P',),('P',)),
    'esimd_vit_chain_v1._merge_packed':(('OUT','PACK'),('OUT','PACK')),
}


class VitChain:
    def __init__(self,model,provider):
        import json
        assert hashlib.sha256(RECEIPT.read_bytes()).hexdigest()=='8b8214b33aea961cab0655913748b9a42b655d4c38da6cbc4b42ba38474cd6eb'
        receipt=json.loads(RECEIPT.read_text())
        assert hashlib.sha256(DLL.read_bytes()).hexdigest()==receipt['sources'][str(DLL)]
        self.model=model;self.provider=provider;self.blocks=tuple(model.vit)
        assert len(self.blocks)==8 and all(type(b) is vit.VitBlock for b in self.blocks)
        self.engine=Dense(DLL);self.engine.require(True)
        self.weights={id(b):self.engine.pack_weight(b.expand.detach().cpu().numpy()) for b in self.blocks}
        self.calls=0;self.selection={};self.probe=None

    def final_projection(self,features,weight,initial):
        m,k=features.shape
        if (m,k)!=(64,1024) or weight.shape!=(1024,1024) or initial.shape!=(64,1024):
            raise ValueError('Unvalidated final ViT projection')
        x=features.contiguous();ini=initial.contiguous()
        partial=torch.empty((4,m,1024),device=x.device,dtype=x.dtype)
        out=torch.empty((m,1024),device=x.device,dtype=x.dtype)
        packed=torch.empty((m//8,1024//16,8,16),device=x.device,dtype=x.dtype)
        _parts[(triton.cdiv(m,32),32,4)](x,weight,ini,partial,m,k,1024,32,32,
            num_warps=4,num_stages=1,enable_fp_fusion=False)
        args=(partial,out,packed,m,1024,512);grid=(triton.cdiv(m*1024,512),)
        if 'merge' not in self.selection:
            _,_,self.selection['merge']=select(_merge_packed,[(512,)],lambda _:args,lambda _:grid,num_warps=4,enable_fp_fusion=False)
        _merge_packed[grid](*args,num_warps=4,enable_fp_fusion=False)
        for _ in range(4):record_arithmetic_dispatch('dense')
        record_arithmetic_dispatch('fp8')
        return out,packed

    def __call__(self,features,analysis):
        if features.shape!=(64,1024) or features.dtype!=torch.float16 or not features.is_contiguous():
            raise ValueError('Expected complete NR256 ViT entry')
        x=vit.q(features)
        packed=torch.empty((8,64,8,16),device=x.device,dtype=x.dtype)
        args=(x,packed,64,1024);grid=(8*64,)
        if 'entry' not in self.selection:
            _,_,self.selection['entry']=select(_entry_pack,[(128,)],lambda _:args,lambda _:grid,num_warps=4,enable_fp_fusion=False)
        _entry_pack[grid](*args,num_warps=4,enable_fp_fusion=False)
        for index,block in enumerate(self.blocks):
            if index: # Retain the original logical FP8 call; it is idempotent.
                x=vit.q(x)
            expanded=torch.empty((64,4096),device=x.device,dtype=x.dtype)
            weight=self.weights[id(block)]
            self.engine.into(packed,weight,None,expanded,m=64,k=1024,n=4096,packed=True)
            analysis.native_dense(packed,weight,expanded)
            record_arithmetic_dispatch('dense');self.provider.record('fp16_dense')
            self.provider.record('fp16_esimd_vit_expand')
            hidden=vit.cubic_quantize(expanded)
            mlp=vit.q(vit.split_k_projection(hidden,block.contract,(x*block.ffn_skip).half()))
            query,key,value=prepare(mlp,block.qkv_weight,block.query_scale,bm=32,bn=64,rows=16)[0]
            for _ in range(2):record_arithmetic_dispatch('dense')
            for _ in range(2):record_arithmetic_dispatch('attention_normalize_c32')
            for _ in range(3):record_arithmetic_dispatch('fp8')
            attended=attend(query.transpose(0,1),key.transpose(0,1),value.transpose(0,1),bm=32,warps=4,stages=1)[0].transpose(0,1).reshape(64,1024)
            for kind in ('batched','attention_exp_vit','fp8','batched','attention_row_sum64','fp8'):record_arithmetic_dispatch(kind)
            out,next_packed=self.final_projection(attended,block.projection,(mlp*block.attn_skip).half())
            if self.probe is not None:self.probe(index,x,(hidden,mlp,query,key,value,attended,out),next_packed)
            x,packed=out,next_packed
        self.calls+=1
        return x

    def close(self):
        self.weights.clear()
