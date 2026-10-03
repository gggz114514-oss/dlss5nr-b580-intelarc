"""Default-off Base/Branch lifecycle audit trial; no GPU imports or math.

Enable TRIAL_ENABLED before selecting a fresh owned game session. Only the
actual adapter.process frame under its serial RLock can arm replay. Static
readiness is a lifecycle seal, never a cached validate() return value.
"""
from __future__ import annotations

import sys
from threading import get_ident
from types import MethodType

TRIAL_ENABLED = False
_SLOT = '_serial_validation_frame_720'
_STATE = '_replay_lifecycle_base_720'
_ADAPTER_FRAME = '_replay_lifecycle_adapter_frame_720'
_MIGRATED = ('decoder_input', 'branch_accum', 'history', 'vit', 'post', 'front')
_GRAPH_PINS = {
    1: 'd2b8a5f3cb913b71c836e8c2dc0a19cb34b0dbacf6a83851010d11a5a172f5de',
    2: '09471129801447263272c596708d4ce0d5b1e059621db35afab7490df6055725',
    3: 'bacf2c8acb00356c79d82b48c73954cf559254fec9d18ab6cca01663f1f05c25',
    4: 'bdd8ed34ee979608f445eb82e2094c8b1b04c3ed288ee5038949c1c52872ca5e',
    5: '8887facb3d98f91f6a068206698283b2865d895cbe7cdae366b0455939f3a6ec',
    6: '3cf46091d2a44457975a95670aa317d6205e0ec9db4c5bc5acf4e70c9af7a052',
}


def trial_requested():
    if type(TRIAL_ENABLED) is not bool:
        raise TypeError('Lifecycle trial flag must be a bool')
    return TRIAL_ENABLED


def state_for(owner):
    return getattr(owner, _STATE, None)


def invalidate_suite(owner, reason):
    state = state_for(owner)
    if state is not None:
        state.invalidate(reason)
        # The owned Suite scope is exiting. Child scopes unwind under their
        # original complete guards, not a revoked hot-frame authorization.
        # Arbitrary entry/resource invalidation does not take this path.
        state.armed = False


def prepare_suite(owner):
    if not trial_requested():
        return
    if any(k in owner.children for k in _MIGRATED):
        state = state_for(owner)
        if state is None:
            state = _Lifecycle(owner)
            setattr(owner, _STATE, state)
        state.seal_static()
    # First select / mode replacement can create the suite inside host.process,
    # after adapter.process established its serial frame. Attach only that
    # exact active frame, never adopt an already captured graph next frame.
    adapter = sys.modules.get('cyberpunk_nr_adapter')
    active = getattr(adapter, _ADAPTER_FRAME, None)
    if active is not None:
        active.bind_owner(owner)


def live_child(child, *, check_thread=True):
    owner = getattr(child.modes, '_numeric_cleanup_owner', None)
    state = state_for(owner)
    if state is None or not state.armed:
        return False
    state.require_frame()
    if state.children.get(child.serial_child_label) is not child:
        state.fail('Lifecycle child owner changed')
    child._validate_live_owner(check_thread=check_thread)
    return True


def require_cold_dispatch(child):
    owner = getattr(child.modes, '_numeric_cleanup_owner', None)
    state = state_for(owner)
    if state is not None and state.armed:
        state.require_frame()
        if state.pending is None:
            state.fail('Candidate dispatch requires the actual audited cold graph build')


class _EntrySeal:
    def __init__(self, entry, owner):
        self.entry, self.graph, self.stream = entry, entry.graph, entry.stream
        self.owner = owner
        self.model, self.constants = owner.model, owner.constants
        self.cache, self.pool, self.capture_stream = owner._graph_cache, owner.capture_pool, owner.capture_stream
        self.inputs, self.output, self.dispatch = entry.inputs, entry.output, entry.dispatch
        self.buffers = tuple(entry.inputs.items())

    def current(self):
        e = self.entry
        return (self.owner.model is self.model and self.owner.constants is self.constants and
                e.graph is self.graph and e.stream is self.stream and
                self.owner._graph_cache is self.cache and self.owner.capture_pool is self.pool and
                self.owner.capture_stream is self.capture_stream and
                e.inputs is self.inputs and e.output is self.output and
                e.dispatch is self.dispatch and len(e.inputs) == len(self.buffers) and
                all(e.inputs.get(k) is v for k, v in self.buffers))


class _Entries(dict):
    """Observe the unchanged v1 get / v5 capture-publish consumption path."""
    def __init__(self, state):
        super().__init__()
        self.state = state

    def get(self, key, default=None):
        value = super().get(key, default)
        caller = sys._getframe(1)
        if caller.f_code is self.state.forward_code:
            return self.state.lookup(caller, key, value)
        return value

    def __setitem__(self, key, entry):
        self.state.publish(sys._getframe(1), key, entry)
        dict.__setitem__(self, key, entry)

    def __delitem__(self, key):
        self.state.invalidate('Graph entry deleted')
        return super().__delitem__(key)

    def clear(self):
        self.state.invalidate('Graph entries cleared / graph retired')
        return super().clear()

    def pop(self, *args):
        self.state.invalidate('Graph entry popped')
        return super().pop(*args)

    def popitem(self):
        self.state.invalidate('Graph entry popped')
        return super().popitem()

    def update(self, *args, **kwargs):
        self.state.invalidate('Graph entries replaced by update')
        return super().update(*args, **kwargs)

    def setdefault(self, key, default=None):
        if key not in self:
            self.state.invalidate('Graph entry inserted outside capture')
        return super().setdefault(key, default)

    def __ior__(self, other):
        self.update(other)
        return self


class _Lifecycle:
    def __init__(self, owner):
        from decoder_input_full_k_720_v1 import _Sources
        self.owner, self.session, self.stack = owner, owner.session, owner.stack
        self.graph, self.provider = owner.graph, owner.stack.provider
        self.all_children = dict(owner.children)
        self.children = {k: v for k, v in owner.children.items() if k in _MIGRATED}
        self.sources = _Sources()
        modules = {i: self.sources.load('graph_front_v' + str(i), sha256=h)
                   for i, h in _GRAPH_PINS.items()}
        if type(self.graph) is not modules[6].GraphFront:
            raise RuntimeError('Lifecycle trial requires the actual pinned v1-v6 GraphFront')
        self.forward_module, self.capture_module = modules[1], modules[5]
        self.forward_code = modules[1].GraphFront.forward.__code__
        self.capture_code = modules[5].GraphFront._capture.__code__
        self.descriptor = modules[1].descriptor
        for value, module, name in (
                (self.graph.forward, modules[2], 'GraphFront.forward'),
                (self.graph._build, modules[4], 'GraphFront._build'),
                (self.graph._capture, modules[5], 'GraphFront._capture'),
                (self.graph.close, modules[6], 'GraphFront.close')):
            self.sources.function(value, module=module, qualname=name)
        self.sources.function(modules[1].GraphFront.forward)
        self.sources.function(self.descriptor)
        self.sources.add(sys.modules[__name__])
        from replay_lifecycle_audit_rest_720_v1 import Binding
        self.sources.add(sys.modules[Binding.__module__])
        self.original_entries = self.graph.entries
        self.original_close = self.graph.close
        self.close_hook = None
        self.entries = None
        self.static_ready = self.armed = False
        self.frame_cold = False
        self.frame = self.pending = self.consumed = None
        self.frame_id = self.frame_thread = None
        self.sealed = {}
        self.reason = None
        self.fixed_audits = 0
        from replay_lifecycle_graph_constants_720_v1 import GraphConstants
        self.graph_constants = GraphConstants(self, modules[1])

    def invalidate(self, reason):
        self.static_ready = False
        self.reason = reason
        self.pending = self.consumed = None
        # Retain entry/binary owners through graph retirement, including failure.

    def fail(self, reason):
        self.invalidate(reason)
        self.session._failed = True
        raise RuntimeError(reason)

    def full_audit(self):
        self.owner._validate_suite_owner()
        self.graph_constants.audit()  # Sources + immutable model: cold scan before any build.
        from replay_lifecycle_audit_rest_720_v1 import LABELS, ready, full_audit
        for label, child in self.children.items():
            complete = ready(child, label) if label in LABELS else child.preflight_complete
            if not complete:
                self.fail('Lifecycle static audit requires completed child preflight')
            if label in LABELS:
                full_audit(child, label, require_ready=True)
            else:
                child.sources.verify()
                child.validate()  # Explicit, uncached Base/Branch audit.
        self.fixed_audits += 1

    def seal_static(self):
        if self.armed or self.graph.entries or self.graph.replays:
            self.fail('Lifecycle preflight seal requires a fresh uncaptured session')
        try:
            self.full_audit()
            from replay_lifecycle_audit_rest_720_v1 import LABELS, _binding
            for label, child in self.children.items():
                if label in LABELS:
                    _binding(child).seal_resources()
            if self.close_hook is None:
                # Preserve the admitted v6 method and its return/error effects.
                # Revoke readiness before synchronize/reset, including failure
                # before the original method can clear entries or set closed.
                def close(graph):
                    self.invalidate('Owned graph.close requested')
                    return self.original_close()
                self.close_hook = MethodType(close, self.graph)
                self.graph.close = self.close_hook
                self.sources.function(self.close_hook)
                self.sources.watch(self.graph, 'close')
            self.graph_constants.install()
            self.static_ready = True
        except BaseException:
            self.invalidate('Static preflight audit failed')
            self.session._failed = True
            raise

    def begin_frame(self, frame):
        if not self.static_ready or self.frame is not None:
            self.fail('Lifecycle static_ready is absent or a frame is already active')
        self.frame = frame
        self.frame_cold = False
        self.frame_id, self.frame_thread = frame.frame_id, frame.thread
        self.require_frame()
        if not self.armed:
            if self.graph.entries is not self.original_entries or self.graph.entries:
                self.fail('Lifecycle trial cannot adopt pre-existing graph entries')
            self.entries = _Entries(self)
            self.graph.entries = self.entries
            self.armed = True

    def require_frame(self):
        f = self.frame
        if (not self.static_ready or f is None or not f._current() or
                f.frame_id != self.frame_id or f.thread != self.frame_thread or
                get_ident() != self.frame_thread or
                state_for(self.owner) is not self or f.owner is not self.owner or
                getattr(self.owner, _SLOT, None) is not f or
                getattr(f.adapter, '_process_serial_lock', None) is not f.guard or
                getattr(getattr(f.adapter, _ADAPTER_FRAME, None), 'cycle', None) is not f or
                getattr(f.adapter.host, '_failed', False) or
                self.owner.session is not self.session or self.owner.stack is not self.stack or
                self.stack.graph is not self.graph or self.stack.provider is not self.provider or
                self.graph.arithmetic is not self.provider or self.graph.closed or
                (self.armed and self.graph.entries is not self.entries) or
                len(self.owner.children) != len(self.all_children) or
                any(self.owner.children.get(k) is not v for k, v in self.all_children.items())):
            self.fail('Lifecycle live frame/lock/session/graph/provider/child owner changed')
        try:
            self.owner._validate_suite_owner()
        except BaseException:
            self.invalidate('Suite owner gate failed in the live frame')
            raise

    def live_all(self):
        self.require_frame()
        for child in self.children.values():
            child._validate_live_owner()

    def lookup(self, caller, key, entry):
        self.live_all()
        for label, child in self.children.items():
            if label in ('history', 'vit', 'post', 'front'):
                child._require_front_after_live_owner()
            else:
                child._require_live_front()
        if (caller.f_globals is not self.forward_module.__dict__ or
                caller.f_locals.get('self') is not self.graph):
            self.fail('Lifecycle entry lookup requires the actual GraphFront.forward owner')
        inputs = caller.f_locals['inputs']
        actual_key = tuple(self.descriptor(t) for t in inputs.values()) + (
            bool(caller.f_locals['return_float32']),)
        if key != actual_key or self.consumed is not None or self.pending is not None:
            self.fail('Lifecycle actual input descriptor key / frame consumption changed')
        if entry is None:
            # Miss is never a hot authorization. Original warmup/capture runs.
            self.frame_cold = True
            self.full_audit()
            self.pending = (key, self.counts())
            return None
        seal = self.sealed.get(key)
        if seal is None or seal.entry is not entry or not seal.current():
            self.fail('Lifecycle replay entry/resource is not the audited actual key owner')
        self.consumed = (key, seal, entry.replays, self.graph.replays)
        return entry

    def counts(self):
        return ({k: (dict(v.calls), dict(v.capture_calls)) for k, v in self.children.items()},
                dict(self.owner.modes.c512_library_calls), dict(self.owner.modes.native_k8_calls))

    @staticmethod
    def capture_expected(label, child, key, entry):
        if label in ('decoder_input', 'branch_accum', 'vit'):
            return dict.fromkeys(child.capture_calls, 1)
        expected = dict.fromkeys(child.capture_calls, 0)
        if label == 'post':
            branch = 'reset' if entry.inputs['previous'] is None else 'history'
            expected['post.' + branch] = 1
            if branch == 'history' and child.options['post_sigmoid'] == 'native':
                expected['sigmoid.native'] = 1
            if child.options['post_store'] != 'reference' and not key[-1]:
                expected['store.' + branch] = 1
        # History/Front execute before actual entry lookup, outside body capture.
        return expected

    def publish(self, caller, key, entry):
        self.live_all()
        if (self.pending is None or key != self.pending[0] or key in self.entries or
                caller.f_code is not self.capture_code or
                caller.f_globals is not self.capture_module.__dict__ or
                caller.f_locals.get('self') is not self.graph or
                caller.f_locals.get('entry') is not entry):
            self.fail('Lifecycle graph entry replacement / publication outside owned v5 capture')
        before, c512, k8 = self.pending[1]
        # Same 2-warmup + 1-capture requirements as Base._frame, before first replay.
        for label, child in self.children.items():
            calls, capture = before[label]
            expected = self.capture_expected(label, child, key, entry)
            if (set(child.capture_calls) != set(capture) or
                    any(child.calls.get(k, 0) - calls.get(k, 0) != 3 * expected.get(k, 0)
                        for k in set(child.calls) | set(calls)) or
                    any(child.capture_calls[k] - v != expected.get(k, 0)
                        for k, v in capture.items())):
                self.fail('Lifecycle original candidate capture gate failed before publication')
        if (any(self.owner.modes.c512_library_calls[k] - v != 3 for k, v in c512.items()) or
                any(self.owner.modes.native_k8_calls[k] - v != 3 for k, v in k8.items())):
            self.fail('Lifecycle original C512/K8 capture gate failed before publication')
        actual_key = tuple(self.descriptor(t) for t in entry.inputs.values()) + (key[-1],)
        if actual_key != key:
            self.fail('Lifecycle captured static inputs differ from the real descriptor key')
        self.full_audit()
        seal = _EntrySeal(entry, self.graph)
        self.sealed[key] = seal
        self.pending = None
        self.consumed = (key, seal, entry.replays, self.graph.replays)

    def finish_frame(self, frame):
        # One final live validation, including original child counters/error
        # effects and any non-migrated interface; no second generic child pass.
        self.require_frame()
        self.owner.validate_frame_context()
        if self.frame is not frame or self.pending is not None or self.consumed is None:
            self.fail('Lifecycle frame did not consume an audited graph entry')
        key, seal, before_entry, before_graph = self.consumed
        if (dict.get(self.entries, key) is not seal.entry or not seal.current() or
                self.graph.last_entry is not seal.entry or
                seal.entry.replays != before_entry + 1 or self.graph.replays != before_graph + 1):
            self.fail('Lifecycle actual replay/entry commit gate failed')
        self.frame = self.consumed = None

    def abort_frame(self, frame):
        if self.frame is frame:
            self.invalidate('Lifecycle frame aborted / capture failed')
            self.session._failed = True
            self.frame = None

    def release_retired(self):
        if self.graph.closed and not self.graph.entries:
            self.sealed.clear()
            self.graph_constants.release_retired()
