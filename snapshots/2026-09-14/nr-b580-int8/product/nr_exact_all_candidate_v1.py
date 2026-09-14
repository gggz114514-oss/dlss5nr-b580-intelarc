"""Explicit composition of exact candidates; no default product replacement."""
from contextlib import ExitStack, contextmanager
from nr_exact_controls_candidate_v1 import Session as ControlsSession
from nr_exact_runtime_v1 import _SERIAL


class Session(ControlsSession):
    enabled = frozenset({'k8', 'shortfp8', 'cubic', 'decoder', 'front', 'history',
                         'schedule', 'post', 'layout', 'compact', 'graph'})
    diagnostics = True
    calls = {}
    resources = {}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from exact_pipeline_v1.front import FusedFront
        from exact_pipeline_v1.scopes import History, Schedule, PostRegion
        from exact_pipeline_v1.graph import Graph
        from exact_pipeline_v1.layout import Layout
        from exact_pipeline_v1.compact import CompactLayout
        from exact_decoder_v1.scope import DecoderGather
        allowed = {'k8', 'shortfp8', 'cubic', 'decoder', 'front', 'history',
                   'schedule', 'post', 'layout', 'compact', 'graph'}
        if not self.enabled <= allowed:
            raise ValueError('Unknown exact optimization')
        self.adapters = dict(front=FusedFront(self._model), history=History(self._model),
                             schedule=Schedule(self._model), post=PostRegion(self._model),
                             decoder=DecoderGather(self._model),
                             layout=(CompactLayout if 'compact' in self.enabled else Layout)(self._model))
        if 'graph' in self.enabled:
            self.adapters['graph'] = Graph(self._model)

    @contextmanager
    def _graph_layout(self):
        from exact_pipeline_v1 import body
        original = body.forward_front
        body.forward_front = self.adapters['layout'].forward_body
        try:
            yield
        finally:
            body.forward_front = original

    def process(self, rgb, motion, *, reset=False):
        with _SERIAL:
            self._ready()
            from nr_exact_k8_candidate_v1 import Session as K8
            from nr_exact_shortfp8_candidate_v1 import Session as Short
            from nr_exact_cubic_candidate_v1 import Session as Cubic
            from nr_exact_all_cubic_candidate_v1 import Session as Both
            with ExitStack() as stack:
                if 'k8' in self.enabled:
                    # This scope expects named counters even when diagnostics are off.
                    self.calls.setdefault('pre', 0)
                    self.calls.setdefault('post', 0)
                    stack.enter_context(K8._k8_scope(self))
                if 'shortfp8' in self.enabled:
                    stack.enter_context(Short._scope(self))
                if 'cubic' in self.enabled:
                    stack.enter_context((Both if 'shortfp8' in self.enabled else Cubic)._scope(self))
                for name in ('front', 'history', 'decoder', 'post'):
                    if name in self.enabled:
                        if name == 'decoder':
                            self.adapters[name].diagnostics = self.diagnostics
                        stack.enter_context(self.adapters[name].installed())
                has_layout = bool(self.enabled & {'layout', 'compact'})
                if 'graph' in self.enabled:
                    if has_layout:
                        stack.enter_context(self._graph_layout())
                    stack.enter_context(self.adapters['graph'].installed())
                elif has_layout:
                    from exact_pipeline_v1 import body
                    if 'schedule' in self.enabled:
                        stack.enter_context(body.installed())
                    stack.enter_context(self.adapters['layout'].installed())
                elif 'schedule' in self.enabled:
                    stack.enter_context(self.adapters['schedule'].installed())
                result = super().process(rgb, motion, reset=reset)
            return result

    def close(self):
        with _SERIAL:
            if 'graph' in self.adapters:
                self.adapters['graph'].close()
            self.adapters.clear()
            super().close()
