"""Selected serialized NR256 fast stack after ShortFP8 complete-call validation.

Constructs exactly the v2 Stack with ShortFP8Graph, as used by the paired native,
paired residual and 390-frame validation. The caller owns toolchain selection,
graph session lifetime and serialization. Exact branch is unchanged.
"""
from nr256_selected_stack_v2 import Stack as Base
from short_fp8_graph_v1 import ShortFP8Graph


class Stack(Base):
    def __init__(self, *args, rewrite_type=ShortFP8Graph, **kwargs):
        super().__init__(*args, rewrite_type=rewrite_type, **kwargs)
