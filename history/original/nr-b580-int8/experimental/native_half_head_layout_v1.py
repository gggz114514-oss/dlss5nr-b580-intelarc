"""Owned C32/multihead attention adapter for the native-half FMA experiment.

Optional ablations select normalization and multihead exp independently. C32 exp
is controlled by the separately installed FusedSwin adapter. Other model paths,
including ViT math and K8 output projection, retain their previous arithmetic.
"""
import nr_backend.multihead_block as multihead
from nr_backend.execution import current_arithmetic_backend,record_arithmetic_dispatch
from fused_c32_projection_native_half_v1 import ProjectedHeadLayout as Base
from projected_head_layout_v1 import ProjectedHeadLayout as Previous
from fused_qkv_pack_v1 import forward as old_prepare
from fused_qkv_pack_native_half_v1 import forward as native_prepare
from fused_swin_heads_v1 import forward as old_heads
from fused_swin_heads_native_half_v1 import forward as native_heads
from window_layout_v1 import unpack

class HeadLayout(Base):
    def __init__(self,model,provider,*,normalize=True,swin=True):
        super().__init__(model,provider,bm=32,warps=4,stages=1)
        self.native_normalize=normalize;self.native_swin=swin

    def c32(self,module,features):
        if not self.native_normalize:return Previous.c32(self,module,features)
        return super().c32(module,features)

    def multi(self,model,features):
        if id(model) not in self.modules or self.provider.mode!='fp16_xmx' or current_arithmetic_backend()!='triton' or features.device.type!='xpu':
            return super().multi(model,features)
        height,width,channels=features.shape
        if channels!=model.channels or height%8 or width%8:raise ValueError('Expected whole multihead windows')
        z=multihead.sm89_f16_dot(multihead.quantize_fp8_grid(features),model.qkv,chunk_k=16).reshape(height,width,model.heads,3,32)
        prepare=native_prepare if self.native_normalize else old_prepare
        (q,k,v),_=prepare(z,model.scale,model.pixel_order,rows=self.rows)
        for _ in range(2):record_arithmetic_dispatch('attention_normalize_c32')
        for _ in range(3):record_arithmetic_dispatch('fp8')
        forward=native_heads if self.native_swin else old_heads
        result,_=forward(q,k,v,model.bias)
        for _ in range(model.heads*((height//8*(width//8)+1023)//1024)):
            record_arithmetic_dispatch('batched');record_arithmetic_dispatch('attention_exp_swin')
            record_arithmetic_dispatch('attention_weights');record_arithmetic_dispatch('batched')
        self.calls['multi']=self.calls.get('multi',0)+1
        self.calls['batched_heads']=self.calls.get('batched_heads',0)+model.heads
        self.calls['qkv_pack']=self.calls.get('qkv_pack',0)+1
        return unpack(result,model.pixel_inverse).reshape(height,width,channels)
