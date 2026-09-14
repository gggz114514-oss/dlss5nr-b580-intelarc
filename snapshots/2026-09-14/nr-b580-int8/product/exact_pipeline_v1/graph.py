"""Exact body graph: owned constants, external I/O, preserved controls boundary.

Adapted from graph_front_v1/v2/v5/v6/v7 without fast provider dependencies.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import torch
from nr_backend.executor import ResetNR
from nr_backend.execution import current_arithmetic_backend
from . import body
from .swin import SwinScheduling


def descriptor(t):
    return None if t is None else (tuple(t.shape), t.dtype, t.device, tuple(t.stride()))


class Graph:
    def __init__(self, model):
        self.model = model
        self.entries = {}
        self.stream = torch.xpu.Stream()
        self.pool = torch.xpu.graph_pool_handle()
        self.constants = self._constants()
        self.closed = False
        self.calls = 0

    def _constants(self):
        return tuple((name, id(t), t.data_ptr(), descriptor(t), t._version)
                     for name, t in self.model.named_buffers() if name != '_previous')

    def compute(self, rgb, front, **options):
        if self.closed or self.constants != self._constants():
            raise RuntimeError('Closed graph or changed model constants')
        inputs = dict(rgb=rgb, front=front, previous=options.pop('previous', None),
                      history_reciprocal=options.pop('history_reciprocal', None))
        key = tuple(descriptor(t) for t in inputs.values()) + (options.get('return_float32', False),)
        if options.get('sigmoid') is not self.model.sigmoid or options.get('blend_scale') is not self.model.blend_scale:
            raise ValueError('Graph requires owned sigmoid and blend scale')
        entry = self.entries.get(key)
        if entry is None:
            static = {name: None if t is None else t.clone(memory_format=torch.contiguous_format)
                      for name, t in inputs.items()}
            def run():
                return body.forward_front(self.model, **static, **options)
            torch.xpu.synchronize()
            # Same 1024-window arithmetic, without device fences inside capture.
            # The external warmup/replay fences still guard history publication.
            with body.installed(), SwinScheduling(enabled=True).installed():
                with self.stream:
                    for _ in range(2):
                        warm = run()
                torch.xpu.synchronize()
                output = torch.empty_like(warm)
                del warm
                graph = torch.xpu.XPUGraph()
                with torch.xpu.graph(graph, stream=self.stream, pool=self.pool):
                    value = run()
                    output.copy_(value)
                del value
            segments = torch.xpu.memory_snapshot(self.pool)
            for t in (output, *[t for t in static.values() if t is not None]):
                if any(s['address'] <= t.data_ptr() < s['address'] + s['total_size'] for s in segments):
                    raise RuntimeError('Persistent graph I/O entered transient pool')
            entry = SimpleNamespace(graph=graph, inputs=static, output=output)
            self.entries[key] = entry
        for name, t in inputs.items():
            if t is not None:
                entry.inputs[name].copy_(t)
        entry.graph.replay()
        result = entry.output.clone()
        torch.xpu.synchronize(rgb.device)
        self.calls += 1
        return result

    @contextmanager
    def installed(self):
        original = ResetNR._forward_front
        def forward(model, rgb, front, *, progress=None, **options):
            if model is not self.model or progress is not None or current_arithmetic_backend() != 'triton':
                return original(model, rgb, front, progress=progress, **options)
            return self.compute(rgb, front, **options)
        ResetNR._forward_front = forward
        try:
            yield
        finally:
            ResetNR._forward_front = original

    def close(self):
        if not self.closed:
            torch.xpu.synchronize()
            for entry in self.entries.values():
                entry.graph.reset()
            self.entries.clear()
            self.pool = self.stream = None
            self.closed = True
