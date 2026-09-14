"""Independent exact decoder gather candidate; source 256 rollout only."""
from nr_exact_controls_candidate_v1 import Session as ControlsSession
from nr_exact_runtime_v1 import _SERIAL


class Session(ControlsSession):
    diagnostics = True
    calls = {}
    resources = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from exact_decoder_v1.scope import DecoderGather
        self._gather = DecoderGather(self._model)

    def process(self, rgb, motion, *, reset=False):
        with _SERIAL:
            self._ready()
            if tuple(getattr(rgb, 'shape', ())[:2]) != (256, 256):
                return super().process(rgb, motion, reset=reset)
            self._gather.diagnostics = self.diagnostics
            before = dict(self._gather.calls)
            with self._gather.installed():
                value = super().process(rgb, motion, reset=reset)
            if self.diagnostics:
                for name, count in self._gather.calls.items():
                    self.calls[name] = self.calls.get(name, 0) + count - before.get(name, 0)
                self.resources.update(self._gather.resources)
            return value

    def close(self):
        with _SERIAL:
            if self._gather is not None:
                self._gather.verify_restored()
                self._gather = None
            super().close()
