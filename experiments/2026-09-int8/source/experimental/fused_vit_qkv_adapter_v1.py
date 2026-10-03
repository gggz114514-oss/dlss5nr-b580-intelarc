"""Owned 64-token FP16 ViT QKV fusion, preserving seven public boundaries.

Other models, arithmetic modes and token sizes retain the original method.
Use inside the existing serialized graph/constant ownership adapter.
"""
from contextlib import contextmanager
import nr_backend.vit_block as vit
from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch
from fused_vit_qkv_v1 import forward


class FusedVitQKV:
    def __init__(self,model,provider):
        self.modules={id(m):m for m in model.modules() if type(m) is vit.VitBlock}
        self.provider=provider
        self.original=vit.VitBlock.forward_boundaries
        self.calls=0

    def apply(self,module,features):
        if (self.modules.get(id(module)) is not module or self.provider.mode!='fp16_xmx'
                or current_arithmetic_backend()!='triton' or features.device.type!='xpu'
                or tuple(features.shape)!=(64,1024)):
            return self.original(module,features)
        x=vit.q(features)
        hidden=vit.cubic_quantize(vit.dot(x,module.expand,chunk_k=16))
        mlp=vit.q(vit.split_k_projection(hidden,module.contract,(x*module.ffn_skip).half()))
        query,key,value=forward(mlp,module.qkv_weight,module.query_scale,bm=32,bn=64,rows=16)[0]
        for _ in range(2):record_arithmetic_dispatch('dense')
        for _ in range(2):record_arithmetic_dispatch('attention_normalize_c32')
        for _ in range(3):record_arithmetic_dispatch('fp8')
        attended=vit.vit_attention(query.transpose(0,1),key.transpose(0,1),value.transpose(0,1)).transpose(0,1).reshape(64,1024)
        output=vit.q(vit.split_k_projection(attended,module.projection,(mlp*module.attn_skip).half()))
        self.calls+=1
        return hidden,mlp,query,key,value,attended,output

    @contextmanager
    def installed(self):
        assert vit.VitBlock.forward_boundaries is self.original
        def replacement(module,features):return self.apply(module,features)
        vit.VitBlock.forward_boundaries=replacement
        try:yield self
        finally:
            assert vit.VitBlock.forward_boundaries is replacement
            vit.VitBlock.forward_boundaries=self.original
