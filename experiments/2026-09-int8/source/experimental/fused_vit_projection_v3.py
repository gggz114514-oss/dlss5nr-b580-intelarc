"""Measured two-kernel tile policy: BM32, BN32 at64tokens, BN64 above64.

All selected tiles were byte-checked at64/96/128/640/960 and19rows in primitivev2.
Timing anchors are real64 and synthetic128/960, not an exhaustive geometry tune.
"""
from fused_vit_projection_v2 import FusedVitProjection as Base,forward
from nr_backend.execution import current_arithmetic_backend,record_arithmetic_dispatch


class FusedVitProjection(Base):
    def __init__(self,model,provider):
        super().__init__(model,provider,bm=32,bn=32,stages=1)

    def apply(self,features,weight,initial,parts=4):
        if self.weights.get(id(weight)) is not weight or self.provider.mode!='fp16_xmx' or current_arithmetic_backend()!='triton' or parts!=4:
            return self.original(features,weight,initial,parts)
        result,_=forward(features,weight,initial,bm=32,bn=32 if features.shape[0]<=64 else 64,stages=1)
        for _ in range(4):record_arithmetic_dispatch('dense')
        self.calls+=1
        return result
