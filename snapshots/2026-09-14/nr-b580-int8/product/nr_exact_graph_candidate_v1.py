"""Exact graph candidate preserving derived control injection and private state."""
from nr_exact_controls_candidate_v1 import Session as ControlsSession
from nr_exact_runtime_v1 import _SERIAL


class Session(ControlsSession):
    calls = {}
    resources = {}
    diagnostics = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from exact_pipeline_v1.graph import Graph
        self.graph = Graph(self._model)

    def process(self, rgb, motion, *, reset=False):
        with _SERIAL:
            self._ready()
            with self.graph.installed():
                result = super().process(rgb, motion, reset=reset)
            if self.diagnostics:
                self.calls['graph'] = self.calls.get('graph', 0) + 1
            return result

    def close(self):
        with _SERIAL:
            self.graph.close()
            super().close()
