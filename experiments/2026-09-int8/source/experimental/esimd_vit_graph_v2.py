"""V2 native ViT chain under the same explicit native-write analysis."""
from contextlib import contextmanager
from fp8_graph_rewrite_v1 import FP8GraphRewrite
from esimd_vit_graph_v1 import EsimdVitGraph as Base
from esimd_vit_chain_v2 import VitChain,CONTRACTS as EXTRA
from quantization_dataflow_v1 import CONTRACTS


class EsimdVitGraph(Base):
    def __init__(self,model,provider):
        FP8GraphRewrite.__init__(self,model,provider);self.chain=VitChain(model,provider)

    @contextmanager
    def installed(self):
        assert not any(k in CONTRACTS for k in EXTRA)
        CONTRACTS.update(EXTRA)
        try:
            with super().installed():yield self
        finally:
            for k,v in EXTRA.items():assert CONTRACTS.pop(k)==v
