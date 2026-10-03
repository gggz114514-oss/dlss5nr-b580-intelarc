"""Select group output tiles from measured small and large C512 geometries.

BM16/BN32 for up to512 rows; BM16/BN64 beyond that. Stage count1 in both.
The threshold is a conservative experimental policy, not an autotuner.
"""
import nr_backend.split_block as split
from nr_backend.execution import current_arithmetic_backend, record_arithmetic_dispatch
from fused_split_ffwd_v1 import FusedSplit as Base, forward


class FusedSplit(Base):
    def apply(self,module,features):
        if id(module) not in self.modules or current_arithmetic_backend()!='triton' or self.provider.mode!='fp16_xmx':
            return self.original(module,features)
        z=split.q(split.dot(split.q(features),module.linear,chunk_k=16))
        value,_=forward(z,module.expand,module.reduce,bm=16,bn=32 if z.numel()//512<=512 else 64,stages=1)
        for _ in range(8):
            record_arithmetic_dispatch('dense');record_arithmetic_dispatch('cubic_fp8')
            record_arithmetic_dispatch('dense');record_arithmetic_dispatch('fp8')
        self.calls+=1
        return value
