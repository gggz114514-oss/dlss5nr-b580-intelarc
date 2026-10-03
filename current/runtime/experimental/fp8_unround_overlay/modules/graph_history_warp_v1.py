"""Capture the unchanged square five-tap history warp with checked table lookup.

Batch the three reciprocal-domain checks into one completion boundary. Invalid
indices are clamped for memory safety, then rejected before returning a result.
Only the owning session's 256/512 square path participates; one serialized caller.
"""
from contextlib import contextmanager
from types import SimpleNamespace
import torch
import nr_backend.temporal as temporal
from nr_backend.reciprocal import START_BITS
from nr_backend.execution import current_arithmetic_backend


class GraphHistoryWarp:
    def __init__(self, model):
        self.table = model.reciprocal
        self.original = temporal.warp_history_square
        self.entries = {}
        self.replays = 0
        self.closed = False
        self.constant = self._constant()

    def _constant(self):
        t = self.table.values
        return (id(t), t.data_ptr(), t._version, t.dtype, t.device, t.shape, t.stride())

    def _build(self, image, motion):
        static_image, static_motion = image.clone(), motion.clone()
        flags = []

        def reciprocal(value):
            index = (value.float().contiguous().view(torch.int32) - START_BITS).long()
            flags.append(((index >= 0) & (index < len(self.table.values))).all())
            return self.table.values[index.clamp(0, len(self.table.values)-1)]

        def compute():
            flags.clear()
            a, b = self.original(static_image, static_motion, return_components=True, reciprocal_source=reciprocal)
            assert len(flags) == 3
            return a, b, torch.stack(flags).all()

        stream = torch.xpu.Stream()
        torch.xpu.synchronize()
        with stream:
            for _ in range(2):
                warm = compute()
        torch.xpu.synchronize()
        outputs = tuple(torch.empty_like(t) for t in warm)
        del warm
        graph = torch.xpu.XPUGraph()
        with torch.xpu.graph(graph, stream=stream):
            temporary = compute()
            for output, value in zip(outputs, temporary):
                output.copy_(value)
        del temporary
        flags.clear()
        segments = torch.xpu.memory_snapshot(graph.pool())
        for t in (static_image, static_motion, *outputs):
            assert not any(s['address'] <= t.data_ptr() < s['address']+s['total_size'] for s in segments)
        return SimpleNamespace(graph=graph,stream=stream,image=static_image,motion=static_motion,outputs=outputs)

    def apply(self, image, motion, *, return_components=False, reciprocal_source=None):
        if (reciprocal_source is not self.table or current_arithmetic_backend() != 'triton'
                or image.device.type != 'xpu' or not return_components):
            return self.original(image,motion,return_components=return_components,reciprocal_source=reciprocal_source)
        if self.closed or self._constant() != self.constant:
            raise RuntimeError('History warp session closed or reciprocal table changed')
        if (image.ndim!=3 or image.shape[-1]!=3 or tuple(image.shape[:2]) not in ((256,256),(512,512))
                or motion.shape!=(*image.shape[:2],2) or image.device!=motion.device):
            raise ValueError('Expected same-device square RGB history and pixel motion')
        key=(image.shape,image.dtype,image.device,motion.shape,motion.dtype)
        entry=self.entries.get(key)
        if entry is None:
            entry=self._build(image,motion)
            self.entries[key]=entry
        entry.image.copy_(image);entry.motion.copy_(motion)
        entry.graph.replay();self.replays+=1
        # Readback completes all prior graph work before MotionNR may commit.
        if not bool(entry.outputs[2]):
            raise ValueError('Input outside the validated native reciprocal interval')
        return entry.outputs[0].clone(),entry.outputs[1].clone()

    @contextmanager
    def installed(self):
        if self.closed: raise RuntimeError('History warp session closed')
        assert temporal.warp_history_square is self.original
        replacement=self.apply
        temporal.warp_history_square=replacement
        try: yield self
        finally:
            assert temporal.warp_history_square is replacement
            temporal.warp_history_square=self.original

    def close(self):
        if self.closed: return
        torch.xpu.synchronize()
        for entry in self.entries.values(): entry.graph.reset()
        self.entries.clear();self.closed=True
