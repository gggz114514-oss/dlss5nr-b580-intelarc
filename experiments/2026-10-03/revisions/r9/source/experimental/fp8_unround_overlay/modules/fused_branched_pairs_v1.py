"""Fuse streaming expansion/activation/reduction per branch, keep projection separate."""
from contextlib import contextmanager
import nr_backend.multihead_block as multihead
from nr_backend.execution import current_arithmetic_backend,record_arithmetic_dispatch
from fused_mlp_pair_v1 import forward as pair


def forward(features,expand,reduce,project,skip_scale,*,bm=16,warps=4,stages=1):
    x=multihead.quantize_fp8(features)
    result=(x*skip_scale).half()
    kernel=None
    for branch in range(features.shape[-1]//32):
        hidden,kernel=pair(x,expand[branch],reduce[branch],bm=bm,warps=warps,stages=stages)
        # Logical original operations, not newly issued standalone kernel calls.
        record_arithmetic_dispatch('dense');record_arithmetic_dispatch('dense')
        record_arithmetic_dispatch('cubic_fp8');record_arithmetic_dispatch('fp8')
        result=multihead.sm89_f16_dot(hidden,project[branch],chunk_k=16,initial=result)
    return result,kernel


class FusedPairs:
    """Only this model's FP16 modules participate; scoped, serialized experiment."""
    def __init__(self,model,provider,*,channels=(64,128,256),bm=16,warps=4,stages=1):
        self.provider=provider
        self.modules={id(module) for module in model.modules() if isinstance(module,multihead.BranchedMLP) and module.channels in channels}
        self.options=dict(bm=bm,warps=warps,stages=stages)
        self.original=multihead.BranchedMLP.forward_unquantized
        self.calls={}

    def apply(self,module,features):
        if id(module) not in self.modules or current_arithmetic_backend()!='triton' or features.device.type!='xpu' or self.provider.mode!='fp16_xmx':
            return self.original(module,features)
        value,_=forward(features,module.expand,module.reduce,module.project,module.skip_scale,**self.options)
        c=module.channels
        self.calls[str(c)]=self.calls.get(str(c),0)+1
        return value

    @contextmanager
    def installed(self):
        assert multihead.BranchedMLP.forward_unquantized is self.original
        replacement=lambda module,features:self.apply(module,features)
        multihead.BranchedMLP.forward_unquantized=replacement
        try:yield self
        finally:
            assert multihead.BranchedMLP.forward_unquantized is replacement
            multihead.BranchedMLP.forward_unquantized=self.original
