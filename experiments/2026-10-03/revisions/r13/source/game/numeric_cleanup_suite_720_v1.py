"""Compose independent, owned numerical candidates on one fresh C512+K8 session.

Each selected scope owns a separate operator family. ViT's changes share one
scope because its live forward otherwise cannot be patched independently.
No tensor/GPU library is imported until a selected child scope needs it.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
import importlib
from pathlib import Path
import hashlib
import sys
from threading import RLock, get_ident
from types import MethodType

from numeric_cleanup_options_720_v1 import NumericCleanupOptions


_SERIAL_GUARD_TYPE = type(RLock())
_SERIAL_CHILD_TYPES = {
    "decoder_input": ("decoder_input_full_k_720_v1", "DecoderInputCounter"),
    "branch_accum": ("branch_accum_native_720_v1", "BranchAccumCounter"),
    "vit": ("vit_numeric_suite_720_v1", "NumericSuite720Counter"),
    "history": ("history_numeric_suite_720_v1", "HistoryNumericCounter720"),
    "post": ("post_numeric_suite_720_v1", "PostNumericCounters"),
    "front": ("front_noise_native_720_v1", "FrontNoiseCounters"),
}


class NumericCleanupCounter:
    def __init__(self, modes, options, native_rtz_receipt=None):
        self.modes = modes
        self.options = options
        self._options_frozen = options
        self.identity = options.identity
        hooks = getattr(modes, "implementation_hooks_720", ())
        if hooks:
            from implementation_hooks_720_v1 import hooks_identity
            self.identity += ":" + hooks_identity(hooks)
        self.configured_options = getattr(modes, "numeric_cleanup_720", None)
        self.native_rtz_receipt = native_rtz_receipt
        self._native_rtz_receipt_frozen = native_rtz_receipt
        self.configured_rtz_receipt = getattr(modes, "native_rtz_receipt", None)
        self.session = modes.session
        self.stack = self.session._stack if self.session is not None else None
        self.graph = self.stack.graph if self.stack is not None else None
        self.children = {}
        self.sources = {}
        self.preflights = {}
        self.closed = False
        self.signature_hook = None
        self.signature_value = None
        self._game_serial_adapter = None
        self._game_serial_guard = None
        self._validation_bindings = []

    def _validate_suite_owner(self):
        self.session._ready()
        if (self.closed or self.options is not self._options_frozen or
                self.modes.session is not self.session or
                self.session._stack is not self.stack or
                self.stack.graph is not self.graph or
                self.graph.closed or
                self.modes.height != 720 or self.modes.variant != "unrounded" or
                tuple(self.modes.source) != (720, 1280) or
                getattr(self.modes, "_numeric_cleanup_owner", None) is not self or
                self.graph.__dict__.get("_signature") is not self.signature_hook or
                self.graph.signature != self.signature_value or
                not self.modes.c512_qkv_library_720 or not self.modes.native_k8_720):
            self.session._failed = True
            raise RuntimeError("Numerical-cleanup fixed session/config/graph/signature owner changed")
        configured = getattr(self.modes, "numeric_cleanup_720", None)
        if configured != self.configured_options:
            self.session._failed = True
            raise RuntimeError("Cannot change numerical options on a captured session")
        if (getattr(self.modes, "native_rtz_receipt", None) != self.configured_rtz_receipt
                or self.native_rtz_receipt != self._native_rtz_receipt_frozen):
            self.session._failed = True
            raise RuntimeError("Cannot change native RTZ evidence on a captured session")

    def validate_context(self):
        if not self._options_frozen.active:
            return
        self._validate_suite_owner()
        for label, child in self.children.items():
            operation = getattr(child, "validate", None)
            if not callable(operation):
                raise TypeError(f"{label} lacks an explicit replay validation interface")
            operation()

    def validate_frame_context(self):
        """Authorized Base/Branch live interface; other families retain their guards."""
        if not self._options_frozen.active:
            return
        self._validate_suite_owner()
        for label, child in self.children.items():
            operation = child._validate_replay
            operation()

    def transfer_serial_thread(self, *, game_adapter, serial_guard, previous_thread):
        """Explicit Cyberpunk process-only handoff, after its bridge stream bind.

        The actual adapter's distinct RLock must cover the whole process call.
        Only exact selected child types with an explicit handoff may participate.
        All children move as one transaction. This never binds/moves an XPU stream.
        Ordinary validate_context()/child.validate() retain their thread guard.
        """
        serial_children, saved_threads, implementation_owners = [], [], []
        try:
            caller = sys._getframe(1)
            host = getattr(game_adapter, "host", None)
            if (game_adapter is not sys.modules.get("cyberpunk_nr_adapter") or
                    host is None or host is not sys.modules.get("nr_game_pre_xess_host") or
                    caller.f_globals is not game_adapter.__dict__ or
                    caller.f_code is not getattr(getattr(game_adapter, "process", None), "__code__", None)):
                raise RuntimeError("Numeric serial handoff is restricted to the actual Cyberpunk adapter.process entry")
            if (type(serial_guard) is not _SERIAL_GUARD_TYPE or
                    serial_guard is not getattr(game_adapter, "_process_serial_lock", None) or
                    serial_guard is getattr(game_adapter, "_settings_lock", None) or
                    not serial_guard._is_owned()):
                raise RuntimeError("Numeric serial handoff requires the adapter's distinct process RLock held by this thread")
            if (self._game_serial_adapter is not None and
                    (self._game_serial_adapter is not game_adapter or
                     self._game_serial_guard is not serial_guard)):
                raise RuntimeError("Numeric serial handoff adapter/process guard changed after binding")
            if (not self.options.active or getattr(host, "_modes", None) is not self.modes or
                    getattr(host, "_failed", False)):
                raise RuntimeError("Numeric serial handoff requires this actual active game host/modes owner")
            if type(previous_thread) is not int:
                raise RuntimeError("Numeric serial handoff requires the previous actual CPU thread")
            self._validate_suite_owner()
            selected = self.options.selected_scopes()
            if not selected or set(self.children) != {label for label, _, _ in selected}:
                raise RuntimeError("Numeric serial handoff selected children changed")
            for label, module_name, _ in selected:
                expected = _SERIAL_CHILD_TYPES.get(label)
                module = sys.modules.get(module_name)
                child = self.children[label]
                if (expected is None or expected[0] != module_name or module is None or
                        type(child) is not getattr(module, expected[1], None) or
                        child.modes is not self.modes or child.session is not self.session or
                        child.graph is not self.graph or
                        not callable(getattr(child, "_transfer_serial_thread", None))):
                    raise RuntimeError(f"Numeric serial handoff lost its exact selected {label} child/interface")
                serial_children.append((label, child))
                if hasattr(child, "thread"):
                    # Save only known, exact counter types; never arbitrary objects.
                    if type(child.thread) is not int or child.thread != previous_thread:
                        raise RuntimeError(f"Numeric serial handoff {label} CPU owner mismatch")
                    saved_threads.append((child, child.thread))
            current_thread = get_ident()
            bridge = getattr(host, "_bridge", None)
            torch = sys.modules.get("torch")
            if (bridge is None or getattr(host, "_thread", None) != current_thread or
                    bridge.thread != current_thread or torch is None or bridge.torch is not torch or
                    bridge.device != torch.device("xpu", torch.xpu.current_device()) or
                    bridge.torch.xpu.current_stream().sycl_queue != bridge.stream.sycl_queue):
                raise RuntimeError("Numeric serial handoff requires the same owned bridge XPU stream already bound to the current host thread")
            if getattr(self.modes,"implementation_hooks_720",()):
                from implementation_hooks_720_v1 import HookCounter
                for hook_owner in self.modes.implementation_calls_720.values():
                    if (type(hook_owner) is not HookCounter or hook_owner.modes is not self.modes or
                            hook_owner.session is not self.session or hook_owner.graph is not self.graph):
                        raise RuntimeError("Numeric handoff lost an exact implementation owner")
                    participants=hook_owner.serial_participants(previous_thread)
                    implementation_owners.append((hook_owner,participants))
                    saved_threads.extend((child,child.thread) for _,child,_ in participants)
            # Other families may reference the selected history owner at idle.
            serial_children.sort(key=lambda item: 0 if item[0] == "history" else 1)
            for label, child in serial_children:
                transferred = child._transfer_serial_thread(
                    self, serial_guard=serial_guard, previous_thread=previous_thread)
                if transferred is not None and transferred != current_thread:
                    raise RuntimeError(f"Numeric serial handoff {label} returned a different CPU owner")
            for hook_owner,participants in implementation_owners:
                hook_owner.transfer_serial_thread(
                    numeric_owner=self,game_adapter=game_adapter,serial_guard=serial_guard,
                    previous_thread=previous_thread)
                for _,child,direct in participants:
                    if direct:
                        transferred=child.transfer_serial_thread(
                            self,game_adapter=game_adapter,serial_guard=serial_guard,
                            previous_thread=previous_thread)
                        if transferred is not None and transferred!=current_thread:
                            raise RuntimeError("Direct implementation handoff returned another CPU owner")
            # The trial's migrated children use live gates on the new thread;
            # stream/idle/caller checks and the whole-suite rollback remain.
            self.validate_frame_context()
            for hook_owner,_ in implementation_owners:
                hook_owner.validate_frame_context()
            self._game_serial_adapter = game_adapter
            self._game_serial_guard = serial_guard
            return current_thread
        except BaseException:
            for child, old_thread in reversed(saved_threads):
                child.thread = old_thread
            if self.session is not None:
                self.session._failed = True
                self.session._numeric_cleanup_retired_owner = self
                if implementation_owners:
                    self.session._implementation_failed_transfer_anchor = tuple(
                        owner for owner,_ in implementation_owners)
            raise

    def preflight(self):
        self.validate_context()
        for label, counter in self.children.items():
            if label in self.preflights:
                continue
            operation = getattr(counter, "preflight", None)
            if not callable(operation):
                raise TypeError(f"{label} lacks an explicit preflight interface")
            self.preflights[label] = operation()
        from replay_lifecycle_audit_base_720_v1 import prepare_suite
        prepare_suite(self)
        return dict(self.preflights)

    def snapshot(self):
        self.validate_context()
        reports = {}
        for label, counter in self.children.items():
            operation = getattr(counter, "snapshot", None)
            if not callable(operation):
                raise TypeError(f"{label} lacks an explicit snapshot interface")
            reports[label] = operation()
        return {"scope": "720p C512+K8 numerical cleanup",
                "options": self.options.to_dict(), "identity": self.identity,
                "native_rtz_receipt": self.native_rtz_receipt,
                "sources": dict(self.sources), "preflight": dict(self.preflights),
                "children": reports}


def _require_baseline(modes):
    session = getattr(modes, "session", None)
    if (session is None or getattr(modes, "height", None) != 720 or
            getattr(modes, "variant", None) != "unrounded" or
            tuple(getattr(modes, "source", ())) != (720, 1280) or
            not getattr(modes, "controlled", False) or
            not getattr(modes, "c512_qkv_library_720", False) or
            not getattr(modes, "native_k8_720", False) or
            getattr(modes, "c512_library_calls", None) is None or
            getattr(modes, "native_k8_calls", None) is None):
        raise RuntimeError("Select the actual 720p C512+K8 unrounded baseline first")
    conflicts = ("c32_window_chain_720", "post_rgb_tail_720", "history_compact_720",
                 "vit_head_720", "c512_encoder_window_blocks_720")
    if any(getattr(modes, name, False) for name in conflicts):
        raise RuntimeError("Numerical candidates cannot be mixed with dormant layout experiments")
    session._ready()
    stack = session._stack
    if (tuple(session.fullsize_geometry) != (720, 1280) or
            tuple(session.fullsize_padding["padding"]) != (768, 1280) or
            stack.provider.mode != "fp16_xmx" or stack.graph.entries):
        raise RuntimeError("Expected a fresh uncaptured 720p/768 C512+K8 session")
    return session


@contextmanager
def installed(modes, options=None, *, preflight=True, native_rtz_receipt=None):
    from numeric_frame_validation_720_v1 import bind_child_validation, restore_child_validation
    options = NumericCleanupOptions.parse(options)
    if native_rtz_receipt is not None and options.post_store != "native_rtz":
        raise ValueError("Native RTZ evidence cannot be attached to a different mathematical arm")
    if not options.active:
        yield NumericCleanupCounter(modes, options)
        return
    session = _require_baseline(modes)
    if "_numeric_cleanup_owner" in modes.__dict__:
        raise RuntimeError("A numerical suite is already installed on this session")
    counter = NumericCleanupCounter(modes, options, native_rtz_receipt)
    graph = counter.graph
    original_signature = graph._signature
    original_signature_value = graph.signature
    had_signature_hook = "_signature" in graph.__dict__
    previous_signature_hook = graph.__dict__.get("_signature")

    def signature(self):
        return original_signature() + (counter.identity,)

    counter.signature_hook = MethodType(signature, graph)
    counter.signature_value = original_signature_value + (counter.identity,)
    modes._numeric_cleanup_owner = counter
    graph._signature = counter.signature_hook
    graph.signature = counter.signature_value
    try:
        with ExitStack() as scopes:
            for label, module_name, kwargs in options.selected_scopes():
                if label == "post" and options.post_store == "native_rtz":
                    kwargs = dict(kwargs, native_rtz_receipt=native_rtz_receipt)
                module = importlib.import_module(module_name)
                source = Path(module.__file__).resolve(strict=True)
                if source.stem != module_name:
                    raise RuntimeError(f"Unexpected selected module source: {source}")
                counter.sources[label] = {"path": str(source),
                                          "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
                child = scopes.enter_context(module.installed(modes, **kwargs))
                if child is None:
                    raise RuntimeError(f"{label} returned no candidate execution receipt")
                counter.children[label] = child
                counter._validation_bindings.append(bind_child_validation(counter, label, child))
            try:
                if preflight:
                    counter.preflight()
                yield counter
            finally:
                from replay_lifecycle_audit_base_720_v1 import invalidate_suite
                invalidate_suite(counter, 'Suite scope exited / retired')
                # This anchor survives a child or final graph.close failure.
                # It retains all constants and loaded binaries during unwind.
                session._numeric_cleanup_retired_owner = counter
    except BaseException:
        session._failed = True
        raise
    finally:
        valid = (getattr(modes, "_numeric_cleanup_owner", None) is counter and
                 graph.__dict__.get("_signature") is counter.signature_hook and
                 graph.signature == counter.signature_value)
        # Retire eager history too. Keep children (and their constants) strongly
        # referenced until the recorded commands are destroyed.
        session._failed = True
        graph_retired = False
        try:
            if graph.entries or not graph.closed:
                graph.close()
            graph_retired = bool(graph.closed and not graph.entries)
        finally:
            if had_signature_hook:
                graph._signature = previous_signature_hook
            else:
                graph.__dict__.pop("_signature", None)
            graph.signature = original_signature_value
            if getattr(modes, "_numeric_cleanup_owner", None) is counter:
                del modes._numeric_cleanup_owner
            counter.closed = True
            for binding in reversed(counter._validation_bindings):
                restore_child_validation(binding)
        if graph_retired and getattr(session, "_numeric_cleanup_retired_owner", None) is counter:
            del session._numeric_cleanup_retired_owner
        if graph_retired:
            from replay_lifecycle_audit_base_720_v1 import state_for
            state = state_for(counter)
            if state is not None:
                state.release_retired()
        if not valid:
            session._failed = True
            raise RuntimeError("Numerical suite ownership changed before cleanup")
