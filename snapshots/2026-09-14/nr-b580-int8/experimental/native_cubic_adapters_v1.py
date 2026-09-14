"""Select seven measured native-cubic geometries; retain all other paths.

Selection: native-cubic-mlp-v1, af7de9283b9e31ad1d3f2a82c7317aebdca12eea894969854ca5f7cf1796288f.
Only owned FP16 XPU modules with observed contiguous half inputs use new kernels.
Existing table identity/version guards and original logical dispatch remain.
"""
import torch
import triton
import nr_backend.split_block as split
from nr_backend.execution import current_arithmetic_backend,record_arithmetic_dispatch
from fused_c32_mlp_lut_v1 import FusedC32 as OldC32
from batched_branched_mlp_v2 import FusedBatched as OldBatched
from fused_split_ffwd_v2 import FusedSplit as OldSplit
from native_cubic_c32_v1 import forward as c32_forward
from native_cubic_batched_v1 import forward as batched_forward
from native_cubic_split_v1 import forward as split_forward

C32_ROWS=frozenset((102400,25600,28224,26880,107584))

def eligible(features,provider):
    return (current_arithmetic_backend()=='triton' and provider.mode=='fp16_xmx'
            and features.device.type=='xpu' and features.dtype==torch.float16
            and features.is_contiguous() and features.ndim>=2)

class FusedC32(OldC32):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.native_modules={id(m) for m in self.modules};self.native_calls={}
    def apply(self,module,original,features):
        rows=features.numel()//32
        if (id(module) not in self.native_modules or not eligible(features,self.provider)
                or features.shape[-1]!=32 or rows not in C32_ROWS
                or self.options!=dict(bm=32,warps=4,stages=1)):
            return super().apply(module,original,features)
        result,_=c32_forward(features,module.expansion,module.contraction,module.skip_scale,
                             self.constant.require(),**self.options)
        for _ in range(triton.cdiv(rows,32768)):
            for kind in ('fp8','cubic_fp8','dense','dense'):record_arithmetic_dispatch(kind)
        key=str(rows);self.calls[key]=self.calls.get(key,0)+1
        self.native_calls[key]=self.native_calls.get(key,0)+1
        return result

class FusedBatched(OldBatched):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.native_calls={}
    def apply(self,module,features):
        if (id(module) not in self.modules or not eligible(features,self.provider)
                or module.channels!=256 or features.shape[-1]!=256 or features.numel()//256!=576
                or self.configurations[256]!=dict(pair_bm=16,pair_stages=1,project_bm=16,project_bn=64)):
            return super().apply(module,features)
        result,_=batched_forward(features,module.expand,module.reduce,module.project,module.skip_scale,
            self.constant.require(),pair_bm=32,pair_stages=1,project_bm=16,project_bn=64)
        self.calls['256']=self.calls.get('256',0)+1
        self.native_calls['576x256']=self.native_calls.get('576x256',0)+1
        return result

class FusedSplit(OldSplit):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.native_calls={}
    def apply(self,module,features):
        if (id(module) not in self.modules or not eligible(features,self.provider)
                or features.shape[-1]!=512 or features.numel()//512!=144):
            return super().apply(module,features)
        z=split.q(split.dot(split.q(features),module.linear,chunk_k=16))
        result,_=split_forward(z,module.expand,module.reduce,bm=16,bn=32,stages=1)
        for _ in range(8):
            for kind in ('dense','cubic_fp8','dense','fp8'):record_arithmetic_dispatch(kind)
        self.calls+=1;self.native_calls['144x512']=self.native_calls.get('144x512',0)+1
        return result
