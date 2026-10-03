"""Share transient graph storage while publishing into pool-external output buffers.

An earlier graph may overwrite a later graph's captured temporary allocation
when replay order changes. v4 demonstrated this on temporal -> reset. Every
capture here copies its complete fresh result into its own external buffer
before completion. No inputs, constants, public results or private history
depend on preserving pooled bytes between calls. Serialized single-owner only.
"""
from types import SimpleNamespace
import time
import torch
from nr_backend.execution import use_arithmetic_backend
import capture_body_v1 as body
from graph_front_v4 import GraphFront as Base


class GraphFront(Base):
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
            # Allocated before capture: this result is outside the shared pool.
            output = torch.empty_like(warm, memory_format=torch.contiguous_format)
            del warm
            graph = torch.xpu.XPUGraph()
            with use_arithmetic_backend('triton') as captured_dispatch:
                with torch.xpu.graph(graph, stream=stream, pool=self.capture_pool):
                    temporary_result = compute()
                    output.copy_(temporary_result)
            del temporary_result
        assert graph.pool() == self.capture_pool
        segments = torch.xpu.memory_snapshot(self.capture_pool)
        for tensor in (output, *[t for t in static.values() if t is not None]):
            assert not any(s['address'] <= tensor.data_ptr() < s['address'] + s['total_size']
                           for s in segments), 'Persistent graph input/output entered transient pool'
        entry = SimpleNamespace(graph=graph, stream=stream, inputs=static, output=output,
                                dispatch=dict(captured_dispatch), replays=0,
                                build_seconds=time.perf_counter() - started)
        self.entries[key] = entry
        return entry

    def metadata(self):
        rows = super().metadata()
        for row in rows:
            row['pool_scope'] = 'Only transient body tensors; every graph copies to its own pool-external output'
            row['persistent_inputs_and_output_verified_outside_pool'] = True
        return rows
