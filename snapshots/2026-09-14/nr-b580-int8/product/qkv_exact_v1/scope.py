from contextlib import contextmanager
import torch
from .pack import pack


class QKV:
    def __init__(self,model):
        from nr_backend.multihead_block import MultiHeadAttention
        self.modules=[m for m in model.modules() if isinstance(m,MultiHeadAttention) and m.channels in (128,256)]
        self.resources={}
        self.diagnostics=False

    def forward(self,module,features):
        import nr_backend.multihead_block as multi
        h,w,c=features.shape
        z=multi.sm89_f16_dot(multi.quantize_fp8(features),module.qkv,chunk_k=16).reshape(h,w,module.heads,3,32)
        (q,k,v),resource=pack(z,module.scale,module.pixel_inverse,diagnostics=self.diagnostics)
        if resource is not None:self.resources[resource['hash']]=resource
        result=torch.stack([multi.swin_attention_windows(q[i],k[i],v[i],module.bias[i]) for i in range(module.heads)])
        return result[...,module.pixel_inverse,:].reshape(module.heads,h//8,w//8,8,8,32).permute(1,3,2,4,0,5).reshape(h,w,c)

    @contextmanager
    def installed(self):
        restored=[]
        for module in self.modules:
            had='forward' in module.__dict__
            old=module.__dict__.get('forward')
            module.forward=lambda features,m=module:self.forward(m,features)
            restored.append((module,had,old))
        try:yield
        finally:
            for module,had,old in reversed(restored):
                if had:module.forward=old
                else:del module.forward
