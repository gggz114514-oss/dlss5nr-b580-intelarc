"""ShortFP8-only exact candidate, independent of the K8 candidate."""
from contextlib import contextmanager
from nr_exact_controls_candidate_v1 import Session as ControlsSession
from nr_exact_runtime_v1 import _SERIAL


class Session(ControlsSession):
    calls = {}
    resources = {}
    diagnostics = True

    @contextmanager
    def _scope(self):
        import nr_backend.triton_fp8 as base_fp8
        import nr_backend.triton_cubic_fp8 as base_cubic
        import nr_backend.triton_attention_weights as base_weights
        from exact_shortfp8_v1 import fp8, cubic, weights
        from triton.compiler.compiler import CompiledKernel
        changed = {fp8._kernel, cubic._direct, weights._weights}
        original_metadata = CompiledKernel.launch_metadata
        def metadata(kernel, grid, stream, *args):
            if kernel.src.fn in changed:
                kernel._init_handles()
                row = dict(name=kernel.src.fn.fn.__name__, spills=kernel.n_spills,
                           registers=kernel.n_regs, shared_bytes=kernel.metadata.shared)
                self.resources[kernel.hash] = row
                if not isinstance(kernel.n_spills, int) or kernel.n_spills != 0:
                    raise RuntimeError(f'ShortFP8 changed kernel spill rejected: {row}')
            return original_metadata(kernel, grid, stream, *args)
        bindings = [(base_fp8, 'quantize_fp8', fp8.quantize_fp8),
                    (base_cubic, 'direct_cubic_fp8', cubic.direct_cubic_fp8),
                    (base_weights, 'normalize_weights', weights.normalize_weights)]
        originals = [(module, name, getattr(module, name)) for module, name, _ in bindings]
        try:
            if self.diagnostics:
                CompiledKernel.launch_metadata = metadata
            for module, name, replacement in bindings:
                def counted(*args, _name=name, _fn=replacement, **kwargs):
                    result = _fn(*args, **kwargs)
                    self.calls[_name] = self.calls.get(_name, 0) + 1
                    return result
                setattr(module, name, counted if self.diagnostics else replacement)
            yield
        finally:
            CompiledKernel.launch_metadata = original_metadata
            for module, name, original in originals:
                setattr(module, name, original)

    def process(self, rgb, motion, *, reset=False):
        with _SERIAL:
            self._ready()
            if tuple(getattr(rgb, 'shape', ())[:2]) != (256, 256):
                return super().process(rgb, motion, reset=reset)
            with self._scope():
                return super().process(rgb, motion, reset=reset)
