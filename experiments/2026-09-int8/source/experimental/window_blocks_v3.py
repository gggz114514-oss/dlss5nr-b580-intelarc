"""Joint FP8 packed-attention/output-projection layout with compile-time spill gates.

The MLP and native-half attention are unchanged. Only C64/C128/C256 blocks use
the new layout; standalone attention and C512 boundary-returning APIs retain
their full tensors. Class overrides and dataflow contracts are scoped/serialized.
"""
from contextlib import contextmanager
import torch
import nr_backend.multihead_block as blocks
from nr_backend.execution import current_arithmetic_backend,record_arithmetic_dispatch
from fused_qkv_pack_native_half_v1 import forward as prepare
from window_block_attention_v3 import forward as attend
from current_dense_tiled_provider_v1 import POLICY
from quantization_dataflow_v1 import CONTRACTS
from window_block_projection_v3 import forward as project

KERNEL='window_block_projection_v3._project'
ATTENTION='window_block_attention_v3._kernel'

class WindowBlocks:
    def __init__(self,model,provider,head_layout):
        self.provider=provider;self.layout=head_layout
        self.modules={id(m):name for name,m in model.named_modules() if isinstance(m,blocks.MultiHeadSwinBlock)}
        self.original=blocks.MultiHeadSwinBlock.forward_unquantized
        self.calls=[];self.probe=None

    def windows(self,module,features):
        height,width,channels=features.shape
        z=blocks.sm89_f16_dot(blocks.quantize_fp8_grid(features),module.qkv,chunk_k=16).reshape(height,width,module.heads,3,32)
        (q,k,v),_=prepare(z,module.scale,module.pixel_order,rows=self.layout.rows)
        for _ in range(2):record_arithmetic_dispatch('attention_normalize_c32')
        for _ in range(3):record_arithmetic_dispatch('fp8')
        result,self.last_attention_kernel,self.last_attention_selection=attend(q,k,v,module.bias)
        for _ in range(module.heads*((height//8*(width//8)+1023)//1024)):
            for key in ('batched','attention_exp_swin','attention_weights','batched'):record_arithmetic_dispatch(key)
        for key,n in (('multi',1),('batched_heads',module.heads),('qkv_pack',1)):
            self.layout.calls[key]=self.layout.calls.get(key,0)+n
        return result

    def apply(self,module,features):
        if (id(module) not in self.modules or self.provider.mode!='fp16_xmx' or current_arithmetic_backend()!='triton'
                or features.device.type!='xpu' or not self.layout.native_normalize or not self.layout.native_swin):
            return self.original(module,features)
        if features.ndim!=3 or features.shape[-1]!=module.channels:raise ValueError('Wrong HWC features')
        h,w=features.shape[:2]
        if min(h,w)==0 or h%2 or w%2:raise ValueError('Expected positive even dimensions')
        sy,sx=module.window_shift
        padded=blocks.pad_or_identity(blocks.quantize_fp8_grid(features),(0,0,sx,(-w-sx)%8,sy,(-h-sy)%8))
        mlp=module.mlp(padded);packed=self.windows(module.attention,mlp)
        hp,wp,c=mlp.shape;key=(hp*wp,c,c,True,(True,True,True))
        tile=POLICY.get(key,(16,32,32,4))
        result,kernel,selection=project(packed,module.output_weight,mlp,module.skip_scale,module.attention.pixel_order,
            height=h,width=w,shift=(sy,sx),tile=tile)
        # Preserve logical arithmetic receipts while replacing its physical work.
        record_arithmetic_dispatch('fp8');record_arithmetic_dispatch('dense')
        self.provider.record('fp16_dense')
        if key in POLICY:
            self.provider.record('fp16_tiled')
            label=f'{hp*wp}x{c}x{c}:initial=True:contiguous={(True,True,True)}'
            self.provider.tiled_calls[label]=self.provider.tiled_calls.get(label,0)+1
        info=dict(module=self.modules[id(module)],shape=list(features.shape),padded_shape=list(mlp.shape),shift=[sy,sx],tile=[*selection['selected'],32,4],projection_selection=selection,attention_selection=self.last_attention_selection)
        self.calls.append(info)
        if self.probe is not None:self.probe(module,features,packed,mlp,result,kernel,info)
        return result

    @contextmanager
    def installed(self):
        assert blocks.MultiHeadSwinBlock.forward_unquantized is self.original and KERNEL not in CONTRACTS and ATTENTION not in CONTRACTS
        replacement=lambda module,features:self.apply(module,features)
        blocks.MultiHeadSwinBlock.forward_unquantized=replacement;CONTRACTS[KERNEL]=(('OUT',),());CONTRACTS[ATTENTION]=(('OUT',),('OUT',))
        try:yield self
        finally:
            assert blocks.MultiHeadSwinBlock.forward_unquantized is replacement and CONTRACTS[KERNEL]==(('OUT',),())
            blocks.MultiHeadSwinBlock.forward_unquantized=self.original;del CONTRACTS[KERNEL];del CONTRACTS[ATTENTION]
