"""Owned ViT forward path consuming the attention producer's head layout.

Only the output-only forward is replaced in this capture scope. The existing
seven-boundary API stays intact; an optional untimed probe exposes all seven
candidate boundaries for full comparison before graph capture.
"""
from contextlib import contextmanager
import nr_backend.vit_block as vit
from nr_backend.execution import current_arithmetic_backend,record_arithmetic_dispatch
from fused_vit_qkv_v1 import forward as prepare
from fused_vit_attention64_v2 import forward as attend
from vit_head_major_projection_v1 import forward as project
from quantization_dataflow_v1 import CONTRACTS

EXTRA={'vit_head_major_projection_v1._parts':(('OUT',),())}


class VitHeadLayout:
    def __init__(self,model,provider,enabled=True):
        self.model=model;self.provider=provider;self.enabled=enabled
        self.modules={id(b):i for i,b in enumerate(model.vit)}
        assert len(self.modules)==8
        self.original=vit.VitBlock.forward
        self.calls=0;self.blocks={};self.resources={};self.probe=None

    def apply(self,module,features):
        if (not self.enabled or id(module) not in self.modules or self.provider.mode!='fp16_xmx'
                or current_arithmetic_backend()!='triton' or features.device.type!='xpu'
                or tuple(features.shape)!=(64,1024)):
            return self.original(module,features)
        x=vit.q(features)
        hidden=vit.cubic_quantize(vit.dot(x,module.expand,chunk_k=16))
        mlp=vit.q(vit.split_k_projection(hidden,module.contract,(x*module.ffn_skip).half()))
        query,key,value=prepare(mlp,module.qkv_weight,module.query_scale,bm=32,bn=64,rows=16)[0]
        for _ in range(2):record_arithmetic_dispatch('dense')
        for _ in range(2):record_arithmetic_dispatch('attention_normalize_c32')
        for _ in range(3):record_arithmetic_dispatch('fp8')
        heads=attend(query.transpose(0,1),key.transpose(0,1),value.transpose(0,1),bm=32,warps=4,stages=1)[0]
        for kind in ('batched','attention_exp_vit','fp8','batched','attention_row_sum64','fp8'):
            record_arithmetic_dispatch(kind)
        projected,kernels=project(heads,module.projection,(mlp*module.attn_skip).half())
        kernel=kernels[0]
        self.resources[kernel.hash]=dict(spills=kernel.n_spills,registers=kernel.n_regs,shared_bytes=kernel.metadata.shared)
        for _ in range(4):record_arithmetic_dispatch('dense')
        output=vit.q(projected)
        index=self.modules[id(module)]
        self.calls+=1;self.blocks[index]=self.blocks.get(index,0)+1
        if self.probe is not None:
            attended=heads.transpose(0,1).reshape(64,1024)
            self.probe(index,features,(hidden,mlp,query,key,value,attended,output),heads)
        return output

    @contextmanager
    def installed(self):
        assert vit.VitBlock.forward is self.original and not any(k in CONTRACTS for k in EXTRA)
        boundary=vit.VitBlock.forward_boundaries
        CONTRACTS.update(EXTRA)
        def replacement(module,features):return self.apply(module,features)
        vit.VitBlock.forward=replacement
        try:
            yield self
        finally:
            valid=vit.VitBlock.forward is replacement and vit.VitBlock.forward_boundaries is boundary
            vit.VitBlock.forward=self.original
            for k,v in EXTRA.items():assert CONTRACTS.pop(k)==v
            assert valid,'ViT method scope interference'

    def verify_restored(self):
        assert vit.VitBlock.forward is self.original and not any(k in CONTRACTS for k in EXTRA)
