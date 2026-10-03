"""Keep C32 projection scratch bounded; write normalized QKV straight to windows."""
import torch
import nr_backend.attention as attention
from nr_backend.execution import current_arithmetic_backend,record_arithmetic_dispatch
from qkv_head_layout_v1 import QKVHeadLayout
from window_layout_v1 import unpack
from fused_c32_qkv_scatter_v1 import scatter
import capture_body_v1 as body


def prepare(module,features,*,rows=16):
    if features.ndim!=3 or features.shape[-1]!=32 or min(features.shape)<=0 or features.shape[0]%8 or features.shape[1]%8:
        raise ValueError('Expected complete HWC32 windows')
    if features.device.type!='xpu' or features.dtype!=torch.float16:
        raise ValueError('Expected half XPU features already at the caller FP8 boundary')
    h,w=features.shape[:2]
    outputs=tuple(torch.empty((h//8,w//8,64,32),dtype=features.dtype,device=features.device) for _ in range(3))
    flat=features.reshape(-1,32)
    # capture_body.c32_front removes only explicit periodic synchronization.
    # Keep that exact scheduling distinction for ordinary progress/eager calls.
    synchronize=attention.C32AttentionFront.forward is not body.c32_front
    for start in range(0,len(flat),32768):
        projected=attention.sm89_f16_dot(flat[start:start+32768],module.front.qkv,chunk_k=16)
        scatter(projected,module.front.scale,module.pixel_inverse,outputs,start,rows=rows)
        del projected
        for _ in range(2):record_arithmetic_dispatch('attention_normalize_c32')
        for _ in range(2):record_arithmetic_dispatch('fp8')
        if synchronize and (start//32768)%8==7:torch.xpu.synchronize(features.device)
    # V was quantized once after the old chunked front finished. The physical
    # per-chunk conversion above retains that same single logical boundary.
    record_arithmetic_dispatch('fp8')
    return outputs


class ChunkedHeadLayout(QKVHeadLayout):
    def __init__(self,model,provider,*,c32_rows=16):
        super().__init__(model,provider)
        self.c32_modules={id(m) for m in model.modules() if isinstance(m,attention.C32AttentionCore)}
        self.c32_rows=c32_rows

    def c32(self,module,features):
        if id(module) not in self.c32_modules or self.provider.mode!='fp16_xmx' or current_arithmetic_backend()!='triton' or features.device.type!='xpu':
            return super().c32(module,features)
        q,k,v=prepare(module,features,rows=self.c32_rows)
        result=attention.swin_attention_windows(q,k,v,module.bias)
        self.calls['c32']=self.calls.get('c32',0)+1
        self.calls['c32_chunk_pack']=self.calls.get('c32_chunk_pack',0)+(features.numel()//32+32767)//32768
        return unpack(result.unsqueeze(0),module.pixel_inverse).reshape(features.shape)
