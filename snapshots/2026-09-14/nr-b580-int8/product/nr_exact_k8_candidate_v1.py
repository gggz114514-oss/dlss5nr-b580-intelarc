"""Exact K8 execution tiling only; controls and all other math stay unchanged.

Port of experimental/k8_tiled_provider_v1.py without its fast provider hierarchy.
Initial rollout is limited to the tested 256-square source path.
"""
from contextlib import contextmanager
from nr_exact_controls_candidate_v1 import Session as ControlsSession
from nr_exact_runtime_v1 import _SERIAL


class Session(ControlsSession):
    calls = {'pre': 0, 'post': 0}

    @contextmanager
    def _k8_scope(self):
        import torch
        import nr_backend.triton_math as arithmetic
        original = arithmetic.fused_dot
        owned = ((self._model.pre.front_weight, 'pre', (16, 32), 2, 32),
                 (self._model.post.head_weight, 'post', (32, 8), 4, 8))

        def dense(a, weight, *, chunk_k, initial=None, block=128):
            for target, name, shape, bm, bn in owned:
                if (weight is target and tuple(weight.shape) == shape
                        and chunk_k == 8 and initial is None
                        and a.device.type == 'xpu' and a.device == weight.device
                        and a.dtype == torch.float16 and weight.dtype == torch.float16
                        and a.shape[-1] == shape[0]):
                    result = arithmetic._tiled_dot(a, weight, chunk_k=8,
                                                  bm=bm, bn=bn, warps=1)
                    self.calls[name] += 1
                    return result
            return original(a, weight, chunk_k=chunk_k, initial=initial, block=block)

        arithmetic.fused_dot = dense
        try:
            yield
        finally:
            arithmetic.fused_dot = original

    def process(self, rgb, motion, *, reset=False):
        with _SERIAL:
            self._ready()
            if tuple(getattr(rgb, 'shape', ())[:2]) != (256, 256):
                return super().process(rgb, motion, reset=reset)
            with self._k8_scope():
                return super().process(rgb, motion, reset=reset)
