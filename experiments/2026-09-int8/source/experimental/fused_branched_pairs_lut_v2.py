"""Measured channel-specific lookup launches; preserve the small-row fallback.

Chosen from wide-mlp-lut-v1 real C64/C128/C256 operands. The tested 19/577-row
lookup kernels did not beat the existing original path, so keep its 1024 cutoff.
"""
from fused_branched_pairs_lut_v1 import FusedPairs as Base


class FusedPairs(Base):
    def __init__(self, model, provider):
        super().__init__(model, provider, min_rows=1024, launch_configs={
            64: dict(bm=32, warps=4, stages=1),
            128: dict(bm=32, warps=4, stages=2),
            256: dict(bm=16, warps=4, stages=2),
        })
