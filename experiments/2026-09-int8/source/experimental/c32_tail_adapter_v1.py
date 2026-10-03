"""Owned C32 block adapter for the validated FP16 attention/projection tail.

Only the four encoder C32 blocks and three ordinary decoder C32 blocks use this
contract. Pre, decoder upsampling and post retain their own arithmetic. This is
experimental and requires the separately authenticated Triton3.8 environment.
"""
from contextlib import contextmanager
import torch
import triton
from nr_backend.c32_block import C32SwinBlock
from nr_backend.execution import current_arithmetic_backend,record_arithmetic_dispatch
from cubic_lut_constant_v1 import Constant
from fused_c32_attention_projection_v1 import forward


class C32Tail:
    def __init__(self,model,provider):
        self.model=model;self.provider=provider;self.constant=Constant(model)
        self.modules=tuple(model.encoder[0])+tuple(model.decoder[-1][1:])
        if len(self.modules)!=7 or any(type(m) is not C32SwinBlock for m in self.modules):
            raise ValueError('Expected the seven ordinary C32 blocks of the pinned NR graph')
        self.names={id(m):n for n,m in model.named_modules() if m in self.modules}
        self.calls={}

    def apply(self,module,original,features):
        if (self.provider.mode!='fp16_xmx' or current_arithmetic_backend()!='triton'
                or features.device.type!='xpu' or features.dtype!=torch.float16):
            return original(features)
        value,_=forward(features,module.mlp.expansion,module.mlp.contraction,module.mlp.skip_scale,
            module.attention.front.qkv,module.attention.front.scale,module.attention.pixel_order,
            module.attention.bias,module.output_weight,module.skip_scale,self.constant.require(),
            shift=module.window_shift,warps=4,stages=1)
        h,w=features.shape[:2];sy,sx=module.window_shift;rows=(h+2*sy)*(w+2*sx)
        chunks=triton.cdiv(rows,32768);groups=triton.cdiv(rows//64,1024)
        counts=dict(dense=1+3*chunks,fp8=4+3*chunks,cubic_fp8=chunks,
            attention_normalize_c32=2*chunks,batched=2*groups,
            attention_exp_swin=groups,attention_weights=groups)
        # Original logical work, for completeness accounting; these are NOT
        # separate hardware launches in this fused implementation.
        for name,count in counts.items():
            for _ in range(count):record_arithmetic_dispatch(name)
        name=self.names[id(module)];self.calls[name]=self.calls.get(name,0)+1
        return value

    @contextmanager
    def installed(self):
        for module in self.modules:
            if 'forward_unquantized' in module.__dict__:
                raise ValueError('An instance already owns a C32 block override')
        previous=[]
        try:
            for module in self.modules:
                original=module.forward_unquantized
                replacement=lambda features,module=module,original=original:self.apply(module,original,features)
                module.forward_unquantized=replacement;previous.append((module,replacement))
            yield self
        finally:
            for module,replacement in reversed(previous):
                assert module.forward_unquantized is replacement
                del module.forward_unquantized
