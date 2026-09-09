"""C32 projection/pack configuration selected on real NR256 module operands."""
from fused_c32_projection_pack_v1 import ProjectedHeadLayout as Base


class ProjectedHeadLayout(Base):
    def __init__(self,model,provider):
        super().__init__(model,provider,bm=32,warps=4,stages=1)
