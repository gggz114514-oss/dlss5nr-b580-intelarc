"""Reuse a session-owned stream and graph pool for serialized NR bodies.

Static inputs and model constants are outside the pool. Each graph retains its
own live output allocation; only freed intermediates may be reused by a later
capture. Public results and private history are external clones. The inherited
completion boundary finishes each replay before another call can reuse storage.
Experimental single-owner, serialized calls only; no concurrent stream sharing.
"""
from types import SimpleNamespace
import time
import torch
from nr_backend.execution import use_arithmetic_backend
import capture_body_v1 as body
from graph_front_v3 import GraphFront as Base
from static_weight_cache_v2 import StaticWeightCache


class GraphFront(Base):
    def __init__(self, model, *, arithmetic):
        super().__init__(model, arithmetic=arithmetic)
        self.capture_stream = torch.xpu.Stream()
        self.capture_pool = torch.xpu.graph_pool_handle()

    def _build(self, key, inputs, options):
        if not self.arithmetic.use_view_cache:
            return self._capture(key, inputs, options)
        original = self.arithmetic.static_cache
        if self._graph_cache is None:
            snapshot = StaticWeightCache()
            with torch.inference_mode(False):
                for k, entry in original.entries.items():
                    snapshot.entries[k] = type(entry)(entry.parent, entry.matrix, entry.packed.clone(),
                                                     entry.scale.clone(), entry.version, entry.name)
            torch.xpu.synchronize()
            self._graph_cache = snapshot
        self.arithmetic.static_cache = self._graph_cache
        try:
            return self._capture(key, inputs, options)
        finally:
            self.arithmetic.static_cache = original

    def _capture(self, key, inputs, options):
        started = time.perf_counter()
        static = {name: None if t is None else t.clone(memory_format=torch.contiguous_format)
                  for name, t in inputs.items()}
        stream = self.capture_stream
        torch.xpu.synchronize()

        def compute():
            return body.forward_front(self.model, **static, **options)

        with body.installed():
            with stream:
                for _ in range(2):
                    with use_arithmetic_backend('triton'):
                        warm = compute()
            torch.xpu.synchronize()
            del warm
            graph = torch.xpu.XPUGraph()
            with use_arithmetic_backend('triton') as captured_dispatch:
                with torch.xpu.graph(graph, stream=stream, pool=self.capture_pool):
                    output = compute()
        assert graph.pool() == self.capture_pool
        entry = SimpleNamespace(graph=graph, stream=stream, inputs=static, output=output,
                                dispatch=dict(captured_dispatch), replays=0,
                                build_seconds=time.perf_counter() - started)
        self.entries[key] = entry
        return entry

    def metadata(self):
        rows = super().metadata()
        for row in rows:
            row['pool'] = str(self.capture_pool)
            row['pool_scope'] = 'Shared only by this serialized session; live outputs retained separately'
        return rows

    def close(self):
        super().close()
        self.capture_pool = None
        self.capture_stream = None
