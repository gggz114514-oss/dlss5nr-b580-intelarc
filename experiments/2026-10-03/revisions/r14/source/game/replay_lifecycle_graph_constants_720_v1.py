"""CPU-only constants scan scheduling for the existing default-OFF lifecycle.

The admitted v1-v6 GraphFront owns an immutable model for one session. Product
updates must exit the suite, close the old graph/session and open a new one.
Private in-place Tensor writes, including mutation followed by restoration,
are outside that contract: without reading _version they cannot be detected
on every replay. Explicit _validate/full_audit and new graph builds still scan.
No graph method body, capture, arithmetic, key, copy or completion is replaced.
"""
from __future__ import annotations

import sys
from types import MethodType


class GraphConstants:
    """One instance hook, using the parent's actual frame and entry authority."""

    def __init__(self, state, module):
        self.state, self.graph, self.module = state, state.graph, module
        self.model, self.constants = self.graph.model, self.graph.constants
        self.scan, self.validate = self.graph._constants, self.graph._validate
        self.validate_code = self.validate.__func__.__code__
        state.sources.function(self.scan, module=module, qualname='GraphFront._constants')
        state.sources.function(self.validate, module=module, qualname='GraphFront._validate')
        state.sources.watch(self.graph, '_validate')
        self.installed = False

        def constants(graph):
            return self.read(graph, sys._getframe(1))

        self.hook = MethodType(constants, self.graph)
        state.sources.function(self.hook)

    def require_owners(self):
        g, s = self.graph, self.state
        validate = g._validate
        if (g.model is not self.model or s.stack.model is not self.model or
                g.constants is not self.constants or
                getattr(validate, '__self__', None) is not g or
                getattr(validate, '__func__', None) is not self.validate.__func__ or
                self.validate.__func__.__code__ is not self.validate_code or
                (self.installed and g.__dict__.get('_constants') is not self.hook)):
            s.fail('Immutable graph model/constants/validation owner changed: retire the session')

    def audit(self):
        """Cold/explicit full validation, never a reused validate return value."""
        try:
            self.require_owners()
            self.state.sources.verify()
            # Fullsize selects and seals the suite before session._installed().
            # Only immutable checks belong to admission. The unchanged
            # GraphFront.forward -> _validate checks the temporary provider
            # on every real dispatch, including sealed replay.
            g = self.graph
            if g.closed:
                raise RuntimeError('Graph session is closed')
            if g._signature() != g.signature:
                raise RuntimeError('Arithmetic changed: create a new graph session')
            self.cold(True)
        except BaseException:
            self.state.invalidate('Immutable graph constants/source cold audit failed')
            self.state.session._failed = True
            raise

    def install(self):
        self.require_owners()
        if not self.installed:
            # Admission already audited the original scan. Do not hook a class
            # or any other session, and watch the instance only after binding.
            self.graph._constants = self.hook
            self.installed = True
            self.state.sources.watch(self.graph, '_constants')

    def cold(self, validating):
        value = self.scan()
        if validating and value != self.constants:
            self.state.fail('Model constants changed: create a new graph session')
        return value

    def read(self, graph, caller):
        if graph is not self.graph:
            self.state.fail('Constants hook called with a different graph owner')
        self.require_owners()
        validating = (caller.f_code is self.validate_code and
                      caller.f_globals is self.module.__dict__ and
                      caller.f_locals.get('self') is graph)
        forward = caller.f_back
        from replay_lifecycle_audit_base_720_v1 import trial_requested
        s = self.state
        if (not trial_requested() or not s.armed or not validating or forward is None or
                forward.f_code is not s.forward_code or
                forward.f_globals is not self.module.__dict__ or
                forward.f_locals.get('self') is not graph):
            return self.cold(validating)

        s.require_frame()
        # Read only dynamic argument descriptors. The real forward constructs
        # and consumes its original key unchanged; lookup independently checks
        # that exact key before replay. A miss cannot use the sealed snapshot.
        values = forward.f_locals
        key = tuple(s.descriptor(values[name]) for name in
                    ('rgb', 'front', 'previous', 'history_reciprocal')) + (
                        bool(values['return_float32']),)
        entry = dict.get(s.entries, key)
        if entry is None:
            return self.cold(True)
        seal = s.sealed.get(key)
        if (s.pending is not None or s.consumed is not None or seal is None or
                seal.entry is not entry or seal.constants is not self.constants or
                not seal.current()):
            s.fail('Constants scan shortcut requires the actual sealed replay entry')
        return self.constants

    def release_retired(self):
        if self.graph.closed and not self.graph.entries and self.installed:
            if self.graph.__dict__.get('_constants') is self.hook:
                del self.graph._constants
            self.installed = False
