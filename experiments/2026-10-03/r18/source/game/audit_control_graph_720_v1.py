"""Cold raw-private graph warmup admission, keeping replay/provider gates intact."""
from contextlib import contextmanager
import sys
from threading import RLock


def control_lanes(controls):
    skin = controls.local_structure if controls.skin_structure is None else controls.skin_structure
    return (controls.style / 128, controls.local_tone,
            1 if controls.auto_mask else controls.local_structure,
            skin if controls.auto_mask else -1,
            controls.local_structure if controls.auto_mask else -1)


def cold_raw_capture_allowed(model, graph):
    modes = model.__dict__.get('_audit_history_host_modes_720')
    token = None if modes is None else modes.__dict__.get('_audit_raw_graph_warmup_720')
    if token is None:
        return False
    token.require(model, graph)
    return True


class RawGraphWarmup720:
    def __init__(self, modes, adapter, guard):
        self.modes, self.adapter, self.guard = modes, adapter, guard
        self.current = None
        self.completed = []
        self.checks = 0
        self.active = True

    def _require_owner(self, model, graph):
        modes = self.modes
        session = modes.session
        host = getattr(self.adapter, 'host', None)
        if (sys.modules.get('cyberpunk_nr_adapter') is not self.adapter
                or getattr(self.adapter, '_process_serial_lock', None) is not self.guard
                or type(self.guard) is not type(RLock())
                or self.guard is getattr(self.adapter, '_settings_lock', None)
                or host is not sys.modules.get('nr_game_pre_xess_host')
                or getattr(host, '_modes', None) is not modes or getattr(host, '_failed', True)
                or not self.guard._is_owned() or session is None
                or session._stack.model is not model or session._stack.graph is not graph
                or not self.active or session._failed or graph.closed
                or modes.height != 720 or tuple(modes.source) != (720, 1280)
                or modes.variant != 'unrounded'):
            raise RuntimeError('Raw graph warmup lost the actual fixed720 serial adapter owner')
        from replay_lifecycle_audit_base_720_v1 import state_for
        state = state_for(getattr(modes, '_numeric_cleanup_owner', None))
        if state is None:
            raise RuntimeError('Cold raw graph warmup requires actual numeric/lifecycle admission')
        state.require_frame()
        return state

    def require(self, model, graph):
        state = self._require_owner(model, graph)
        controls = model._controls
        if (self.current is None or self.current['frame'] is not state.frame
                or self.checks != 0 or state.pending is not None or state.consumed is not None
                or controls.style != 0 or not 0 <= controls.intensity < 1):
            raise RuntimeError('Raw capture requires the current cold reset/history frame token')
        self.checks += 1

    @contextmanager
    def frame(self, *, reset):
        """Wrap one real modes.process call INSIDE its adapter validation frame."""
        if (type(reset) is not bool or self.current is not None or len(self.completed) >= 2
                or reset is not (len(self.completed) == 0)):
            raise RuntimeError('Raw warmup requires exactly cold reset then history')
        session = self.modes.session
        model, graph = session._stack.model, session._stack.graph
        state = self._require_owner(model, graph)
        if state.pending is not None or state.consumed is not None or any(r['frame'] is state.frame for r in self.completed):
            raise RuntimeError('Raw warmup needs two distinct unconsumed serial validation frames')
        self.current = dict(frame=state.frame, frame_id=state.frame_id, reset=reset)
        self.checks = 0
        try:
            yield self
            self._require_owner(model, graph)
            consumed = state.consumed
            if consumed is None or self.checks != 1:
                raise RuntimeError('Raw warmup did not consume its one admitted graph entry')
            key, seal, before_entry, before_graph = consumed
            entry = seal.entry
            if (key[-1] is not True or (key[2] is None) is not reset
                    or dict.get(state.entries, key) is not entry or not seal.current()
                    or graph.last_entry is not entry or entry.replays != before_entry + 1
                    or graph.replays != before_graph + 1):
                raise RuntimeError('Raw warmup lacks actual FP32 reset/history replay proof')
            self.completed.append(self.current)
        except BaseException:
            session._failed = True
            raise
        finally:
            self.current = None


@contextmanager
def cold_raw_control_frame(modes, *, game_adapter, serial_guard):
    """Main bootstrapping uses TWO real serialized frames at intensity<1:

    reset=True then reset=False, each with its own adapter validation frame.
    Existing GraphFront V5 builds/publishes/validates the raw-FP32 entries.
    Run before user-visible steady frames; never keep this token for gameplay.
    """
    slot = '_audit_raw_graph_warmup_720'
    if slot in modes.__dict__:
        raise RuntimeError('Nested cold raw graph warmup')
    token = RawGraphWarmup720(modes, game_adapter, serial_guard)
    modes.__dict__[slot] = token
    try:
        yield token
        if len(token.completed) != 2:
            raise RuntimeError('Raw warmup token needs two successful reset/history frames')
    except BaseException:
        session = modes.session
        if session is not None:
            session._failed = True
            modes._audit_raw_graph_failed_anchor_720 = (token, session, session._stack.graph)
        raise
    finally:
        if modes.__dict__.get(slot) is not token:
            raise RuntimeError('Cold raw graph warmup token owner changed')
        modes.__dict__.pop(slot)
        token.active = False
