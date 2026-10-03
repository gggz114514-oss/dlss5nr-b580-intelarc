"""One owned front route, shared by post/front observers; stdlib only.

Observers run before the single selected implementation. The dispatcher's
__wrapped__ is the real bound FusedFront.apply reference, never an observer
that a numeric implementation could silently bypass. No tensor/kernel work
is performed here. All registrations are scoped to one session/front owner.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
from os.path import abspath, normcase
from types import FunctionType, MethodType


_MARKER = "__nr_front_route_dispatch_720_v1__"


def _same_method(left, right):
    return (type(left) is MethodType and type(right) is MethodType
            and left.__self__ is right.__self__ and left.__func__ is right.__func__)


def _failed(counter):
    counter.session._failed = True


def _reference(counter):
    from replay_lifecycle_audit_rest_720_v1 import front_reference
    reference = front_reference(counter)
    if reference is not None:
        return reference
    owner, session, module = counter.front_owner, counter.session, counter.modules["fused_front"]
    stack = session._stack
    if (counter.stack is not stack or counter.model is not stack.model
            or not any(component is owner for component in stack.components)
            or type(owner) is not module.FusedFront
            or owner.noise is not counter.noise or counter.model.noise is not counter.noise):
        raise RuntimeError("Front dispatcher requires the same owned model/session/front")
    function = module.FusedFront.apply
    frozen = [callee for obj, name, callee in counter.known_callees
              if obj is module.FusedFront and name == "apply"]
    if (len(frozen) != 1 or function is not frozen[0] or type(function) is not FunctionType
            or function.__globals__ is not vars(module)
            or normcase(abspath(function.__code__.co_filename)) != normcase(abspath(module.__file__))):
        raise RuntimeError("Front dispatcher lost its authenticated FusedFront.apply source")
    return MethodType(function, owner)


class _Dispatcher:
    def __init__(self, counter, parent):
        self.owner, self.session, self.model = counter.front_owner, counter.session, counter.model
        self.module = counter.modules["fused_front"]
        self.parent, self.parent_code = parent, parent.__func__.__code__
        self.saved = ("apply" in self.owner.__dict__, self.owner.__dict__.get("apply"))
        self.observers, self.scopes = (), []
        self.impl, self.impl_counter, self.impl_code = None, None, None
        self.noise_mode, self.running = "table", False

        @wraps(parent)
        def apply(instance, rgb, previous=None, **options):
            return self.dispatch(instance, rgb, previous, options)

        self.function, self.code = apply, apply.__code__
        apply.__nr_numeric_front_noise__ = self.noise_mode
        self.apply = MethodType(apply, self.owner)

    def check(self, counter):
        reference = _reference(counter)
        if (counter.front_owner is not self.owner or counter.session is not self.session
                or counter.model is not self.model or counter.modules["fused_front"] is not self.module
                or not _same_method(reference, self.parent)
                or self.parent.__func__.__code__ is not self.parent_code
                or self.owner.__dict__.get(_MARKER) is not self
                or self.owner.__dict__.get("apply") is not self.apply
                or self.apply.__self__ is not self.owner or self.apply.__func__ is not self.function
                or self.function.__code__ is not self.code
                or self.function.__wrapped__ is not self.parent
                or self.function.__globals__ is not globals()
                or self.function.__closure__[0].cell_contents is not self
                or self.function.__nr_numeric_front_noise__ != self.noise_mode):
            raise RuntimeError("Owned front dispatcher/apply/reference chain changed")
        if self.impl_counter is None:
            if self.impl is not None or self.impl_code is not None or self.noise_mode != "table":
                raise RuntimeError("Baseline front dispatcher implementation changed")
        elif (getattr(self.impl, "__self__", None) is not self.impl_counter
              or not _same_method(self.impl, self.impl_counter._run)
              or self.impl.__func__.__code__ is not self.impl_code
              or self.impl_counter.options.get("front_noise", "table") != self.noise_mode):
            raise RuntimeError("Selected front implementation/counter changed")

    def dispatch(self, instance, rgb, previous, options):
        counters = tuple(counter for counter, route, code in self.observers)
        try:
            if instance is not self.owner or self.running or not counters:
                raise RuntimeError("Foreign, recursive or unobserved front dispatch")
            for counter, route, code in self.observers:
                self.check(counter)
                if (getattr(route, "__self__", None) is not counter
                        or not _same_method(route, counter._front_route)
                        or route.__func__.__code__ is not code):
                    raise RuntimeError("Owned front route observer changed")
            if self.impl_counter is not None:
                self.check(self.impl_counter)
                if not any(counter is self.impl_counter for counter in counters):
                    raise RuntimeError("Selected front implementation has no route observer")
            self.running = True
            try:
                for counter, route, code in self.observers:
                    route(rgb, previous, options)
                    self.check(counter)
                if self.impl_counter is None:
                    return self.parent(rgb, previous, **options)
                return self.impl(rgb, previous, options)
            finally:
                self.running = False
        except BaseException:
            self.session._failed = True
            for counter in counters:
                _failed(counter)
            raise


@contextmanager
def _scope(counter):
    state, token = None, object()
    try:
        parent = _reference(counter)
        state = counter.front_owner.__dict__.get(_MARKER)
        if state is None:
            if not _same_method(counter.front_owner.apply, parent):
                raise RuntimeError("Unrecognized FusedFront.apply; metadata wrappers are not dispatchers")
            state = _Dispatcher(counter, counter.front_owner.apply)
            counter.front_owner.__dict__[_MARKER] = state
            counter.front_owner.apply = state.apply
        elif type(state) is not _Dispatcher:
            raise RuntimeError("Foreign front dispatcher owner/source")
        state.check(counter)
        if state.running:
            raise RuntimeError("Cannot change front registrations during dispatch")
        state.scopes.append(token)
        try:
            yield state
        finally:
            try:
                state.check(counter)
                if not state.scopes or state.scopes[-1] is not token:
                    raise RuntimeError("Front dispatcher scopes must restore in reverse order")
            finally:
                if state.scopes and state.scopes[-1] is token:
                    state.scopes.pop()
                    if not state.scopes:
                        if state.owner.__dict__.get("apply") is state.apply:
                            if state.saved[0]:
                                state.owner.__dict__["apply"] = state.saved[1]
                            else:
                                state.owner.__dict__.pop("apply", None)
                        if state.owner.__dict__.get(_MARKER) is state:
                            state.owner.__dict__.pop(_MARKER)
    except BaseException:
        _failed(counter)
        raise


@contextmanager
def observe(counter):
    """Register this counter's real route observer on the shared front entry."""
    with _scope(counter) as state:
        if any(owned is counter for owned, route, code in state.observers):
            raise RuntimeError("Front counter already has a route observer")
        route = counter._front_route
        if getattr(route, "__self__", None) is not counter:
            raise RuntimeError("Front route observer is not owned by its counter")
        previous = state.observers
        selected = previous + ((counter, route, route.__func__.__code__),)
        state.observers = selected
        try:
            yield
        finally:
            changed = state.observers is not selected
            state.observers = previous
            if changed:
                raise RuntimeError("Front observer registrations changed before restoration")


@contextmanager
def implementation(counter, callback):
    """Select one owned (rgb, previous, options-dict) callback beneath observers."""
    with _scope(counter) as state:
        if (getattr(callback, "__self__", None) is not counter
                or not _same_method(callback, counter._run)):
            raise RuntimeError("Front implementation must be this counter's actual _run")
        previous = (state.impl_counter, state.impl, state.impl_code, state.noise_mode)
        code, noise_mode = callback.__func__.__code__, counter.options.get("front_noise", "table")
        state.impl_counter, state.impl, state.impl_code, state.noise_mode = counter, callback, code, noise_mode
        state.function.__nr_numeric_front_noise__ = noise_mode
        try:
            yield
        finally:
            changed = (state.impl_counter is not counter or state.impl is not callback
                       or state.impl_code is not code or state.noise_mode != noise_mode
                       or state.function.__nr_numeric_front_noise__ != noise_mode)
            state.impl_counter, state.impl, state.impl_code, state.noise_mode = previous
            state.function.__nr_numeric_front_noise__ = state.noise_mode
            if changed:
                raise RuntimeError("Front implementation changed before restoration")


def require_selected(counter):
    """Require the actual selected callback in a frame, restored reference otherwise."""
    try:
        reference = _reference(counter)
        state = counter.front_owner.__dict__.get(_MARKER)
        if not counter._in_frame:
            if state is not None or not _same_method(counter.front_owner.apply, reference):
                raise RuntimeError("Front apply was not restored outside its frame scope")
            return
        if type(state) is not _Dispatcher:
            raise RuntimeError("Selected front implementation has no owned dispatcher")
        state.check(counter)
        if (state.impl_counter is not counter or not _same_method(state.impl, counter._run)
                or not any(owned is counter for owned, route, code in state.observers)):
            raise RuntimeError("Selected front implementation/counter is absent from the actual dispatcher")
    except BaseException:
        _failed(counter)
        raise
