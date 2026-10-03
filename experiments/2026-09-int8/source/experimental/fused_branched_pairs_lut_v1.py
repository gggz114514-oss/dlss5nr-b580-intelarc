"""Use an explicit cubic LUT in each ordered branch expansion and contraction."""
from contextlib import contextmanager
import nr_backend.multihead_block as multihead
from nr_backend.execution import current_arithmetic_backend,record_arithmetic_dispatch
from fused_mlp_pair_lut_v1 import forward as pair
from fused_branched_pairs_v1 import FusedPairs as Base
from cubic_lut_constant_v1 import Constant


def forward(features,expand,reduce,project,skip_scale,lut,*,bm=16,warps=4,stages=1):
    x=multihead.quantize_fp8(features)
    result=(x*skip_scale).half()
    kernel=None
    for branch in range(features.shape[-1]//32):
        hidden,kernel=pair(x,expand[branch],reduce[branch],lut,bm=bm,warps=warps,stages=stages)
        # Logical original operations, not newly issued standalone kernel calls.
        record_arithmetic_dispatch('dense');record_arithmetic_dispatch('dense')
        record_arithmetic_dispatch('cubic_fp8');record_arithmetic_dispatch('fp8')
        result=multihead.sm89_f16_dot(hidden,project[branch],chunk_k=16,initial=result)
    return result,kernel


class FusedPairs(Base):
    def __init__(self, model, provider, *, launch_configs=None, min_rows=1024):
        super().__init__(model, provider)
        self.constant = Constant(model)
        self.min_rows = int(min_rows)
        if self.min_rows < 0:
            raise ValueError('Minimum rows must be nonnegative')
        self.launch_configs = {c: dict(bm=16, warps=4, stages=2 if c==256 else 1)
                               for c in (64,128,256)}
        if launch_configs is not None:
            if set(launch_configs) != {64,128,256}:
                raise ValueError('Expected all three channel configurations')
            self.launch_configs = {c: dict(v) for c,v in launch_configs.items()}

    def apply(self, module, features):
        if (id(module) not in self.modules or current_arithmetic_backend()!='triton'
                or features.device.type!='xpu' or self.provider.mode!='fp16_xmx'
                or features.numel()//module.channels < self.min_rows):
            return self.original(module, features)
        c = module.channels
        value, _ = forward(features, module.expand, module.reduce, module.project,
                           module.skip_scale, self.constant.require(), **self.launch_configs[c])
        self.calls[str(c)] = self.calls.get(str(c),0) + 1
        return value
