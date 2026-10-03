"""Retain an independent quantized-weight snapshot for the graph's lifetime."""
import torch
from graph_front_v2 import GraphFront as Base
from static_weight_cache_v2 import StaticWeightCache

class GraphFront(Base):
    def __init__(self,model,*,arithmetic):
        super().__init__(model,arithmetic=arithmetic)
        self._graph_cache=None

    @property
    def owned_packed_bytes(self):
        return 0 if self._graph_cache is None else self._graph_cache.packed_bytes

    def _build(self,key,inputs,options):
        if not self.arithmetic.use_view_cache:return super()._build(key,inputs,options)
        original=self.arithmetic.static_cache
        if self._graph_cache is None:
            snapshot=StaticWeightCache()
            # Both graphs share this one owned cache. Parent matrices stay alive,
            # while packed tensors no longer depend on the provider cache lifetime.
            with torch.inference_mode(False):
                for k,entry in original.entries.items():
                    snapshot.entries[k]=type(entry)(entry.parent,entry.matrix,entry.packed.clone(),entry.scale.clone(),entry.version,entry.name)
            torch.xpu.synchronize()
            self._graph_cache=snapshot
        self.arithmetic.static_cache=self._graph_cache
        try:return super()._build(key,inputs,options)
        finally:self.arithmetic.static_cache=original

    def close(self):
        super().close()
        self._graph_cache=None
