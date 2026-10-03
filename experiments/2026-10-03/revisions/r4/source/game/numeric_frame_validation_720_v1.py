"""Frame-local CPU validation reuse for the owned Cyberpunk fast adapter.

Legacy validation reuse stays frame-local. The default-off Base/Branch
lifecycle trial uses separate live interfaces under the actual game lock;
the other four families retain first/phase and uncached exit validation.
"""
from __future__ import annotations

from functools import wraps
import sys
from threading import get_ident, RLock
import time


_SLOT = "_serial_validation_frame_720"
_LOCK_TYPE = type(RLock())


def _function_identity(value, anchors):
    function = getattr(value, "__func__", value)
    owner = getattr(value, "__self__", None)
    code = getattr(function, "__code__", None)
    anchors.extend((function, owner, code))
    return id(function), id(owner), id(code)


def _phase(child, cycle):
    # Frame/dispatch hooks have different guards from idle handoff validation.
    result = tuple(bool(getattr(child, name, False)) for name in
                   ("_local", "_in_scope", "_post_active", "_in_frame", "in_frame",
                    "_live", "_retired", "closed", "retired", "_closed"))
    result += (bool(getattr(child, "active", True)), bool(getattr(child, "_active", True)))
    for target, name in ((getattr(child, "model", None), "forward"),
                         (getattr(child, "model", None), "_forward_front"),
                         (getattr(child, "owner", None), "apply"),
                         (getattr(child, "projection_owner", None), "apply"),
                         (getattr(child, "front_owner", None), "apply"),
                         (getattr(child, "post", None), "forward"),
                         (getattr(child, "sigmoid", None), "forward")):
        result += _function_identity(getattr(target, name, None), cycle._anchors)
    return result


class _NoFrame:
    def close(self, success):
        return None

    def abort(self):
        return None

    def snapshot(self):
        return {"enabled": False, "active": False, "full_checks": 0,
                "reused_checks": 0, "final_validation_ms": 0.0}


class _Frame:
    def __init__(self, owner, adapter, serial_guard, frame_id, *, cache_enabled=True):
        self.owner, self.adapter, self.guard = owner, adapter, serial_guard
        self.session, self.modes, self.graph = owner.session, owner.modes, owner.graph
        self.thread, self.frame_id = get_ident(), int(frame_id)
        self.children = dict(owner.children)
        self._memo, self._anchors = {}, []
        self.active = True
        self.full_checks = self.reused_checks = 0
        self.final_validation_ms = 0.0
        self.retired = False
        self.cache_enabled = cache_enabled
        self.lifecycle = None

    def _current(self):
        owner = self.owner
        return (self.active and get_ident() == self.thread and self.guard._is_owned()
                and getattr(self.adapter.host, "_modes", None) is self.modes
                and self.modes.session is self.session
                and getattr(self.modes, "_numeric_cleanup_owner", None) is owner
                and getattr(owner, _SLOT, None) is self
                and not owner.closed and not self.graph.closed
                and owner.graph is self.graph and owner.session is self.session
                and len(owner.children) == len(self.children)
                and all(owner.children.get(k) is v for k, v in self.children.items()))

    def checked(self, child, label, operation, *, check_thread=True):
        if (not self.cache_enabled or not self.active or
                get_ident() != self.thread or not self.guard._is_owned()):
            return operation()
        if not self._current():
            if self._retired_session():
                return operation()
            self.session._failed = True
            raise RuntimeError("Validation frame child/session binding changed")
        if not any(child is item for item in self.children.values()):
            return operation()
        # This inexpensive check remains on every hit, including after handoff.
        self.owner._validate_suite_owner()
        if check_thread:
            fixed_check = getattr(child, "_require_fixed_session", None)
            if callable(fixed_check):
                try:
                    fixed_check()
                except BaseException:
                    # Preserve the original family's failed-session side effects.
                    return operation()
            if hasattr(child, "thread") and child.thread != get_ident():
                # Let the family's original guard perform its failure rollback.
                return operation()
            thread_check = getattr(child, "_require_thread", None)
            if callable(thread_check):
                thread_check()
        key = (id(child), label, _phase(child, self))
        if key in self._memo:
            self.reused_checks += 1
            value = self._memo[key]
            return dict(value) if isinstance(value, dict) else value
        value = operation()
        self.full_checks += 1
        self._memo[key] = value
        return value

    def abort(self):
        if self.lifecycle is not None:
            self.lifecycle.abort_frame(self)
        self.active = False
        self._memo.clear()
        self._anchors.clear()
        if getattr(self.owner, _SLOT, None) is self:
            delattr(self.owner, _SLOT)

    def _retired_session(self):
        return (getattr(self.adapter.host, "_modes", None) is not self.modes
                or self.modes.session is not self.session
                or (self.owner.closed and self.graph.closed))

    def close(self, success):
        if not self.active:
            return
        current = self._current()
        self.retired = self._retired_session()
        if self.lifecycle is not None:
            # Keep the controlled live authorization through the successful exit.
            # Other families still perform their uncached final checks.
            self._memo.clear()
            self._anchors.clear()
            try:
                if success and current:
                    started = time.perf_counter_ns()
                    try:
                        self.lifecycle.finish_frame(self)
                    finally:
                        self.final_validation_ms = (time.perf_counter_ns() - started) / 1e6
                elif success and not self.retired:
                    self.session._failed = True
                    raise RuntimeError("Validation frame binding/lock changed before commit")
            finally:
                self.abort()
            return
        # Clear before exit checks: the final validation must never hit a cache.
        self.abort()
        if success and not current and not self.retired:
            self.session._failed = True
            raise RuntimeError("Validation frame binding/lock changed before commit")
        if success and current:
            started = time.perf_counter_ns()
            try:
                self.owner.validate_context()
            finally:
                self.final_validation_ms = (time.perf_counter_ns() - started) / 1e6

    def snapshot(self):
        return {"enabled": self.cache_enabled, "active": self.active, "frame_id": self.frame_id,
                "full_checks": self.full_checks, "reused_checks": self.reused_checks,
                "final_validation_ms": self.final_validation_ms, "retired": self.retired,
                "lifecycle_trial": self.lifecycle is not None,
                "semantics": ("Six-family lifecycle audit + live frame/entry authorization; GraphFront constants unchanged"
                              if self.lifecycle is not None else
                              "frame-local reuse; full first/phase check and uncached successful exit; no cross-frame cache")}


class _TrialFrame:
    """Actual adapter call lifetime, including first select inside host.process."""
    def __init__(self, adapter, guard, frame_id, enabled):
        self.adapter, self.guard, self.frame_id = adapter, guard, int(frame_id)
        self.thread, self.enabled, self.cycle = get_ident(), enabled, None
        self.active = True

    def _current(self):
        from replay_lifecycle_audit_base_720_v1 import _ADAPTER_FRAME
        return (self.active and get_ident() == self.thread and self.guard._is_owned() and
                getattr(self.adapter, '_process_serial_lock', None) is self.guard and
                getattr(self.adapter, _ADAPTER_FRAME, None) is self)

    def bind_owner(self, owner):
        from replay_lifecycle_audit_base_720_v1 import state_for
        if (not self._current() or getattr(self.adapter.host, '_modes', None) is not owner.modes or
                getattr(owner.modes, '_numeric_cleanup_owner', None) is not owner):
            owner.session._failed = True
            raise RuntimeError('Lifecycle suite select requires the actual current adapter frame')
        if self.cycle is not None:
            if self.cycle.owner is owner:
                return
            if not self.cycle._retired_session():
                owner.session._failed = True
                raise RuntimeError('Lifecycle suite changed without retiring the previous session')
            self.cycle.abort()
        if getattr(owner, _SLOT, None) is not None:
            raise RuntimeError('Lifecycle suite already has an active frame')
        owner._validate_suite_owner()
        cycle = _Frame(owner, self.adapter, self.guard, self.frame_id, cache_enabled=self.enabled)
        self.cycle = cycle
        setattr(owner, _SLOT, cycle)
        from replay_lifecycle_audit_base_720_v1 import _MIGRATED
        if any(k in owner.children for k in _MIGRATED):
            cycle.lifecycle = state_for(owner)
            if cycle.lifecycle is None:
                owner.session._failed = True
                raise RuntimeError('Lifecycle trial requires completed suite preflight/static_ready')
            cycle.lifecycle.begin_frame(cycle)

    def close(self, success):
        try:
            if self.cycle is not None:
                self.cycle.close(success)
        finally:
            self.abort()

    def abort(self):
        from replay_lifecycle_audit_base_720_v1 import _ADAPTER_FRAME
        if self.cycle is not None:
            self.cycle.abort()
        self.active = False
        if getattr(self.adapter, _ADAPTER_FRAME, None) is self:
            delattr(self.adapter, _ADAPTER_FRAME)

    def snapshot(self):
        return self.cycle.snapshot() if self.cycle is not None else _NoFrame().snapshot()


def serial_validation_frame(owner, *, game_adapter, serial_guard, frame_id, enabled=True):
    """Called directly by actual adapter.process while its RLock is held."""
    from replay_lifecycle_audit_base_720_v1 import trial_requested, state_for
    trial = trial_requested()
    if not trial and (not enabled or owner is None or not getattr(owner.options, "active", False)):
        return _NoFrame()
    caller = sys._getframe(1)
    if (game_adapter is not sys.modules.get("cyberpunk_nr_adapter")
            or getattr(game_adapter, "host", None) is not sys.modules.get("nr_game_pre_xess_host")
            or caller.f_globals is not game_adapter.__dict__
            or caller.f_code is not getattr(game_adapter.process, "__code__", None)):
        raise RuntimeError("Validation reuse requires the actual Cyberpunk adapter.process caller")
    if (type(serial_guard) is not _LOCK_TYPE
            or serial_guard is not getattr(game_adapter, "_process_serial_lock", None)
            or not serial_guard._is_owned()
            or serial_guard is getattr(game_adapter, "_settings_lock", None)):
        raise RuntimeError("Validation reuse requires the owned adapter process lock")
    if trial:
        from replay_lifecycle_audit_base_720_v1 import _ADAPTER_FRAME
        if getattr(game_adapter, _ADAPTER_FRAME, None) is not None:
            raise RuntimeError('Lifecycle adapter frame is already active')
        cycle = _TrialFrame(game_adapter, serial_guard, frame_id, enabled)
        setattr(game_adapter, _ADAPTER_FRAME, cycle)
        try:
            if owner is not None and getattr(owner.options, 'active', False):
                cycle.bind_owner(owner)
        except BaseException:
            state = state_for(owner)
            if state is not None:
                state.invalidate('Suite owner gate failed before frame creation')
            cycle.abort()
            raise
        return cycle
    if (getattr(game_adapter.host, "_modes", None) is not owner.modes
            or getattr(owner.modes, "numeric_cleanup_calls", None) is not owner
            or getattr(owner, _SLOT, None) is not None):
        raise RuntimeError("Validation reuse lost its active frame/session owner")
    owner._validate_suite_owner()
    cycle = _Frame(owner, game_adapter, serial_guard, frame_id, cache_enabled=enabled)
    setattr(owner, _SLOT, cycle)
    return cycle


def checked(child, label, operation, *, check_thread=True):
    owner = getattr(getattr(child, "modes", None), "_numeric_cleanup_owner", None)
    cycle = getattr(owner, _SLOT, None)
    if isinstance(cycle, _Frame):
        return cycle.checked(child, label, operation, check_thread=check_thread)
    return operation()


def bind_child_validation(owner, label, child):
    """Stable instance entries, including direct family-specific heavy guards."""
    bindings = []
    if label in ('decoder_input', 'branch_accum'):
        # Explicit diagnostics are always full. Their frame/handoff scopes call
        # the distinct _validate_replay interface, never a cached validate().
        return child, ()

    def bind(name, wrapper):
        bindings.append((name, wrapper, name in child.__dict__, child.__dict__.get(name)))
        setattr(child, name, wrapper)

    original = child.validate

    @wraps(original)
    def validated(*args, **kwargs):
        if args or kwargs:
            return original(*args, **kwargs)
        return checked(child, "validate", original)

    bind("validate", validated)
    # These families do not inherit _SessionCounter. Their private handoff,
    # finish-frame and captured dispatch can enter heavy guards directly.
    guard_name = {"history": "_validate_fixed_owner", "vit": "_guard",
                  "post": "_guard_fixed", "front": "_guard_fixed"}.get(label)
    if guard_name is not None:
        full_guard = getattr(child, guard_name)

        @wraps(full_guard)
        def guarded(*args, **kwargs):
            from replay_lifecycle_audit_base_720_v1 import trial_requested
            if trial_requested():
                # Trial guards choose live/cold themselves; never memoize a cold
                # guard or an explicit fixed diagnosis through the legacy frame.
                return full_guard(*args, **kwargs)
            if label in ("post", "front"):
                names = ("fresh", "dispatch", "sources")
                if (args or set(kwargs) - set(names)
                        or any(type(value) is not bool for value in kwargs.values())):
                    return full_guard(*args, **kwargs)
                flags = tuple(kwargs.get(name, False) for name in names)
                if flags[0] or flags[2]:
                    # Fresh graph and source-file checks must observe every call.
                    return full_guard(*args, **kwargs)
                if flags[1]:
                    # Authorization is live even when fixed weight metadata was
                    # checked already. Preserve the family's original error path.
                    modules = child.modules
                    if (not child._live or not child._in_frame or not child._preflight_complete
                            or modules["policy"].ENABLED != modules["policy"].FAMILIES
                            or modules["execution"].current_arithmetic_backend() != "triton"):
                        return full_guard(*args, **kwargs)
            elif args or kwargs:
                return full_guard(*args, **kwargs)
            else:
                flags = ()
            return checked(child, (guard_name, flags),
                           lambda: full_guard(*args, **kwargs), check_thread=False)

        bind(guard_name, guarded)
    if label in ("history", "vit", "post", "front"):
        from replay_lifecycle_audit_rest_720_v1 import bind as bind_rest
        bind_rest(child, label, bind)
    return child, tuple(bindings)


def restore_child_validation(binding):
    child, bindings = binding
    for name, wrapper, had, saved in reversed(bindings):
        if child.__dict__.get(name) is wrapper:
            if had:
                setattr(child, name, saved)
            else:
                delattr(child, name)
