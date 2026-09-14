"""Owned complete body with exact layout-reading projections."""
from nr_exact_controls_candidate_v1 import Session as ControlsSession
from nr_exact_runtime_v1 import _SERIAL


class Session(ControlsSession):
    compact = False
    calls = {}
    resources = {}
    diagnostics = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from exact_pipeline_v1.layout import Layout
        from exact_pipeline_v1.compact import CompactLayout
        self.layout = (CompactLayout if self.compact else Layout)(self._model)

    def process(self, rgb, motion, *, reset=False):
        with _SERIAL:
            self._ready()
            before = self.layout.calls
            with self.layout.installed():
                value = super().process(rgb, motion, reset=reset)
            if self.diagnostics:
                self.calls['layout_blocks'] = self.calls.get('layout_blocks', 0) + self.layout.calls-before
            return value

    def close(self):
        with _SERIAL:
            self.layout = None
            super().close()
