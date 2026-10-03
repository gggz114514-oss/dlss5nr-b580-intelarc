"""Batch this model's FP16 multihead Swin, keeping original projections/layout."""
import nr_backend.multihead_block as multihead
from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch
from window_layout_v1 import WindowLayout as Base,pack,unpack
from fused_swin_heads_v1 import forward


class HeadLayout(Base):
    def __init__(self,model,provider):
        super().__init__()
        self.modules={id(m) for m in model.modules() if type(m) is multihead.MultiHeadAttention}
        self.provider=provider

    def multi(self,model,features):
        if id(model) not in self.modules or self.provider.mode!='fp16_xmx' or current_arithmetic_backend()!='triton' or features.device.type!='xpu':
            return super().multi(model,features)
        height,width,channels=features.shape
        if channels!=model.channels or height%8 or width%8:
            raise ValueError('Expected whole multihead windows')
        z=multihead.sm89_f16_dot(multihead.quantize_fp8(features),model.qkv,chunk_k=16).reshape(height,width,model.heads,3,32)
        q=multihead.quantize_fp8((multihead.normalize_c32(z[:,:,:,0])*model.scale[None,None,:,None]).half())
        k=multihead.quantize_fp8(multihead.normalize_c32(z[:,:,:,1]))
        v=multihead.quantize_fp8(z[:,:,:,2])
        q,k,v=[pack(t,model.pixel_order) for t in (q,k,v)]
        result,_=forward(q,k,v,model.bias)
        # Preserve the old logical groups, not a physical launch count.
        for _ in range(model.heads*((height//8*(width//8)+1023)//1024)):
            record_arithmetic_dispatch('batched');record_arithmetic_dispatch('attention_exp_swin')
            record_arithmetic_dispatch('attention_weights');record_arithmetic_dispatch('batched')
        self.calls['multi']=self.calls.get('multi',0)+1
        self.calls['batched_heads']=self.calls.get('batched_heads',0)+model.heads
        return unpack(result,model.pixel_inverse).reshape(height,width,channels)
