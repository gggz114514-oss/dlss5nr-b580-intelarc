"""Capture the shared base body without bypassing derived control/front methods.

v6 remains the frozen MotionNR reference. This opt-in adapter leaves the derived
ControlledMotionNR._forward_front in the call chain, so per-frame controls, masks,
raw-private handling and outer style/UI processing execute normally. It patches
the base body only for its own model. Still a serialized experimental context.
"""
from contextlib import contextmanager
from nr_backend.executor import ResetNR
from nr_backend.temporal import MotionNR
from nr_backend.controlled_temporal import ControlledMotionNR
from nr_backend.live_temporal import LiveControlledMotionNR
from graph_front_v6 import GraphFront as Base

_NATIVE_BASE_BODY = ResetNR._forward_front


class GraphFront(Base):
    def __init__(self, model, *, arithmetic):
        if type(model) not in (MotionNR, ControlledMotionNR, LiveControlledMotionNR):
            raise TypeError('Unverified model override: expected the known NR control hierarchy')
        if '_forward_front' in model.__dict__ or ResetNR._forward_front is not _NATIVE_BASE_BODY:
            raise RuntimeError('Construct the graph adapter outside other body overrides')
        super().__init__(model, arithmetic=arithmetic)
        # Fallback is the base body: derived controls have already executed before
        # this interception. Calling the derived entry here would repeat them.
        self.original = _NATIVE_BASE_BODY.__get__(model, type(model))

    @contextmanager
    def installed(self):
        if self.closed:
            raise RuntimeError('Graph session is closed')
        if '_forward_front' in self.model.__dict__ or ResetNR._forward_front is not _NATIVE_BASE_BODY:
            raise RuntimeError('Another body override is active')

        def replacement(model, *args, **kwargs):
            if model is self.model:
                return self.forward(*args, **kwargs)
            return _NATIVE_BASE_BODY(model, *args, **kwargs)

        ResetNR._forward_front = replacement
        try:
            yield self
        finally:
            assert ResetNR._forward_front is replacement
            ResetNR._forward_front = _NATIVE_BASE_BODY
