"""Selectable execution-only candidates. Explicit composition, exact math."""
from contextlib import ExitStack
from nr_exact_controls_candidate_v1 import Session as ControlsSession
from nr_exact_runtime_v1 import _SERIAL


class Session(ControlsSession):
    enabled = frozenset({'front', 'history', 'schedule', 'post'})
    diagnostics = False
    calls = {}
    resources = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from exact_pipeline_v1.front import FusedFront
        from exact_pipeline_v1.scopes import History, Schedule, PostRegion
        from exact_decoder_v1.scope import DecoderGather
        self.adapters = dict(front=FusedFront(self._model), history=History(self._model),
                             schedule=Schedule(self._model), post=PostRegion(self._model),
                             decoder=DecoderGather(self._model))
        if not self.enabled <= self.adapters.keys():
            raise ValueError('Unknown exact pipeline option')

    def process(self, rgb, motion, *, reset=False):
        with _SERIAL:
            self._ready()
            with ExitStack() as stack:
                for name in ('front', 'history', 'schedule', 'decoder', 'post'):
                    if name in self.enabled:
                        adapter = self.adapters[name]
                        if name == 'decoder':
                            adapter.diagnostics = self.diagnostics
                        stack.enter_context(adapter.installed())
                result = super().process(rgb, motion, reset=reset)
            return result

    def close(self):
        with _SERIAL:
            self.adapters.clear()
            super().close()
