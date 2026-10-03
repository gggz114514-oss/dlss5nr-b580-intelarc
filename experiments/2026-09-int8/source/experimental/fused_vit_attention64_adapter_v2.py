"""Owned 64-token FP16 ViT QKV plus full attention fusion, preserving seven boundaries.

Other models, arithmetic modes and token sizes retain the original method.
Use inside the existing serialized graph/constant ownership adapter.
"""
from contextlib import contextmanager
import nr_backend.vit_block as vit
from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch
from fused_vit_qkv_v1 import forward
from fused_vit_attention64_v2 import forward as attention64


class FusedVitQKV:
    def __init__(self,model,provider):
        self.modules={id(m):m for m in model.modules() if type(m) is vit.VitBlock}
        self.provider=provider
        self.original=vit.VitBlock.forward_boundaries
        self.calls=0
        self.attention_calls=0

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
        attended=attention64(query.transpose(0,1),key.transpose(0,1),value.transpose(0,1),bm=32,warps=4,stages=1)[0].transpose(0,1).reshape(64,1024)
        for kind in ('batched','attention_exp_vit','fp8','batched','attention_row_sum64','fp8'):record_arithmetic_dispatch(kind)
        self.attention_calls+=1
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
