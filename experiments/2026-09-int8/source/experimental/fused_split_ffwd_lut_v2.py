"""BM32/BN64 selected in both small real and repeated-large group measurements.

Large-frame pipeline timings and byte checks remain the adoption gate.
"""
from fused_split_ffwd_lut_v1 import FusedSplit as Base


class FusedSplit(Base):
    def __init__(self, model, provider):
        super().__init__(model, provider, configuration=dict(bm=32, bn=64, stages=1))
