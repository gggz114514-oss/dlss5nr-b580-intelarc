"""Bounded CPU evidence for one NR frame; never loads a DLL or touches a tensor.

The native patch and optional host hooks are reviewed/staged separately. Reads
use only the web module's already-held CDLL. IDs, not uncalibrated timestamps,
join a native SR evaluation to its Python NR call. No polling or file I/O here.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import ctypes as C
from functools import wraps
import threading
import time

VERSION, CAPACITY, FRAME_CAPACITY = 2, 64, 128
EXPORT = "NRB_GetPeriodicFlashDiagV2"
CONTEXT_EXPORT = "NRB_GetPeriodicFlashContextV2"
COUNTERS = (
    "seen", "eligible", "source_proven", "delegated", "submitted", "nr_composited",
    "original_fallback", "excluded_original", "game_reset", "generation_mismatch",
    "invalidated", "failure", "nr_started", "nr_return_ok", "nr_processed",
    "nr_retired", "nr_skipped",
)
REASONS = (
    "none", "scope_or_feature", "source_state", "record_rejected", "tail_unverified",
    "disarmed", "arguments", "attempt_budget", "textures", "pending_full", "allocation",
    "motion_contract", "device", "record_resources", "hdr_missing", "motion_proxy",
    "record_close", "original_evaluate", "processor_missing", "disabled", "latched_failure",
    "prepared_wait", "processor_result", "composite_or_wait", "completion_signal",
    "retire", "generation", "list_reset", "cpp_exception", "prepare_signal",
    "prepare_event", "prepare_registration", "prepare_timeout", "queue_wait",
    "composite_record", "composite_close", "completion_event", "completion_registration",
    "completion_timeout", "retire_callback",
)
EVENT_KINDS = ("record_reject", "fallback", "reset", "failure", "generation", "invalidated")
SOURCE_BITS = (
    "reset_observed", "closed", "oversized_barrier", "color_seen", "plain_barrier",
    "all_subresources", "exact_psr", "age_le_64", "proven", "incomplete",
)
STAGE_BITS = (
    "recorded", "submitted", "processor_called", "processor_return_ok", "result_valid",
    "composite_submitted", "sr_submitted", "retired", "raw_fallback", "invalidated",
    "motion_probe_recorded", "original_called",
)


class Context(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        "eval_id", "list", "generation", "feature_id", "nr_frame_id", "sr_sequence",
        "evaluate_sequence", "color_sequence", "submit_batch",
    )] + [(name, C.c_uint32) for name in (
        "route", "evidence", "effective_route", "eligible", "sr_slot", "source_bits",
        "color_after", "color_age", "game_reset", "enabled", "history", "controls_known",
        "nr_reset", "stage_bits", "motion_probe_index", "record_reason",
    )]


class Event(C.Structure):
    _fields_ = [(name, C.c_uint64) for name in (
        "serial", "steady_us", "composited_before", "detail", "detail2",
    )] + [("context", Context), ("kind", C.c_uint32), ("reason", C.c_uint32)]


class Frame(C.Structure):
    _fields_ = [("serial", C.c_uint64), ("steady_us", C.c_uint64),
                ("context", Context), ("reason", C.c_uint32), ("reserved", C.c_uint32)]


class Snapshot(C.Structure):
    _fields_ = [("abi_size", C.c_uint32), ("version", C.c_uint32)] + [
        (name, C.c_uint64) for name in (
            "retained", "overwritten", "dropped", "frames_retained", "frames_overwritten",
            "frames_dropped", "sampled_steady_us",
        )] + [("counters", C.c_uint64 * len(COUNTERS)),
              ("reasons", C.c_uint64 * len(REASONS)), ("current", Context),
              ("events", Event * CAPACITY), ("frames", Frame * FRAME_CAPACITY)]


ABI_SIZE = 32912
if tuple(C.sizeof(t) for t in (Context, Event, Frame, Snapshot)) != (136, 184, 160, ABI_SIZE):
    raise RuntimeError("unsupported v2 native diagnostic ABI")
_CACHE_KEY = "_nrb_periodic_flash_v2_bindings"
_BIND_LOCK = threading.Lock()


def _binding(dll, export, structure):
    if dll is None:
        return "unavailable", "no_existing_dll"
    try:
        cache = vars(dll)
    except TypeError:
        return "unavailable", "no_dll_object"
    with _BIND_LOCK:
        bindings = cache.setdefault(_CACHE_KEY, {})
        if export not in bindings:
            try:
                function = getattr(dll, export)
                if not callable(function):
                    raise TypeError("snapshot export is not callable")
                function.argtypes, function.restype = [C.POINTER(structure)], C.c_int
                bindings[export] = "ready", function
            except AttributeError:
                bindings[export] = "unavailable", "missing_export"
            except Exception as exc:
                bindings[export] = "error", "binding_failed: " + str(exc)[:240]
        return bindings[export]


def _flags(mask, names):
    return {"mask": mask, "flags": {name: bool(mask & (1 << i)) for i, name in enumerate(names)},
            "unknown_mask": mask & ~((1 << len(names)) - 1)}


def _context(context):
    value = {name: int(getattr(context, name)) for name, _ in Context._fields_}
    value["source"] = _flags(value["source_bits"], SOURCE_BITS)
    value["stages"] = _flags(value["stage_bits"], STAGE_BITS)
    return value


def _reason(value):
    return REASONS[value] if value < len(REASONS) else "unknown"


def read_current_context(dll):
    """Called on the NR callback thread; TLS is unavailable on the web thread."""
    state, function = _binding(dll, CONTEXT_EXPORT, Context)
    if state != "ready":
        return {"status": state, "reason": function}
    context = Context()
    try:
        returned = function(C.byref(context))
    except Exception as exc:
        return {"status": "error", "reason": "context_read_failed: " + str(exc)[:240]}
    if returned == 0:
        return {"status": "unavailable", "reason": "outside_native_nr_callback"}
    if returned != 1:
        return {"status": "error", "reason": "invalid_export_result"}
    return {"status": "ok", "context": _context(context)}


def read_snapshot(dll):
    result = {"status": "unavailable", "export": EXPORT, "events": [], "frames": [],
              "counters": {}, "reasons": {}, "clock_domains_aligned": False,
              "counters_transactional": False}
    state, function = _binding(dll, EXPORT, Snapshot)
    if state != "ready":
        result.update(status=state, reason=function)
        return result
    snapshot = Snapshot()
    snapshot.abi_size, snapshot.version = ABI_SIZE, VERSION
    started = time.monotonic_ns()
    try:
        returned = function(C.byref(snapshot))
    except Exception as exc:
        result.update(status="error", reason="read_failed: " + str(exc)[:240])
        return result
    finally:
        result["python_sample_time"] = {"monotonic_ns_start": started,
                                        "monotonic_ns_end": time.monotonic_ns()}
    if returned == 0:
        result["status"] = "busy"
        return result
    if returned != 1:
        result.update(status="error", reason="invalid_export_result")
        return result
    if (snapshot.abi_size != ABI_SIZE or snapshot.version != VERSION or
            snapshot.retained > CAPACITY or snapshot.frames_retained > FRAME_CAPACITY):
        result.update(status="error", reason="invalid_native_header")
        return result
    result.update(status="ok", native_header={name: int(getattr(snapshot, name)) for name in (
        "abi_size", "version", "retained", "overwritten", "dropped", "frames_retained",
        "frames_overwritten", "frames_dropped", "sampled_steady_us",
    )}, current=_context(snapshot.current),
        counters={name: int(snapshot.counters[i]) for i, name in enumerate(COUNTERS)},
        reasons={name: int(snapshot.reasons[i]) for i, name in enumerate(REASONS)})
    for event in snapshot.events[:snapshot.retained]:
        result["events"].append({name: int(getattr(event, name)) for name in (
            "serial", "steady_us", "composited_before", "detail", "detail2",
        )} | {"context": _context(event.context), "reason": _reason(event.reason),
             "reason_id": int(event.reason), "kind_id": int(event.kind),
             "kind": EVENT_KINDS[event.kind] if event.kind < len(EVENT_KINDS) else "unknown"})
    for frame in snapshot.frames[:snapshot.frames_retained]:
        result["frames"].append({"serial": int(frame.serial), "steady_us": int(frame.steady_us),
                                 "context": _context(frame.context), "reason": _reason(frame.reason)})
    return result


def model_cpu_state(modes):
    """Only plain counters/identities and scalar controls; never tensor values."""
    session = getattr(modes, "session", None)
    stack = getattr(session, "_stack", None)
    model, graph = getattr(stack, "model", None), getattr(stack, "graph", None)
    seed, replays, entries = (getattr(model, "_next_seed", None),
                              getattr(graph, "replays", None), getattr(graph, "entries", None))
    controls = getattr(model, "_controls", None)
    scalar_controls = {}
    for name in ("style", "intensity", "local_tone", "local_structure", "auto_mask", "skin_structure"):
        value = getattr(controls, name, None)
        scalar_controls[name] = value if type(value) in (int, float, bool, type(None)) else None
    return {"next_seed": seed if type(seed) is int else None,
            "graph_entries": len(entries) if isinstance(entries, dict) else None,
            "graph_replays": replays if type(replays) is int else None,
            "session_id": id(session) if session is not None else None,
            "private_history_present": getattr(model, "_previous", None) is not None if model is not None else None,
            "controls": scalar_controls}


def _observational(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            # Diagnostics must never disable a good NR call or mask its error.
            self._errors += 1
            self._last_error = {"method": method.__name__, "type": type(exc).__name__,
                                "message": str(exc)[:240]}
            return None
    return guarded


class PythonFrameRecorder:
    """Keep 128 returned/failed NR calls in RAM and 32 anomalous events."""
    def __init__(self):
        self._lock = threading.Lock()
        self._frames, self._events = deque(maxlen=128), deque(maxlen=32)
        self._serial = self._event_serial = 0
        self._errors, self._last_error = 0, None

    @_observational
    def start(self, frame_id, game_reset, dll):
        native = read_current_context(dll)
        joined = (native.get("status") == "ok" and
                  native["context"]["nr_frame_id"] == frame_id and
                  native["context"]["sr_sequence"] > 0)
        return {"frame_id": frame_id, "game_reset": bool(game_reset),
                "started_monotonic_ns": time.monotonic_ns(), "native": native,
                "native_join_valid": joined, "before": None, "reset_causes": [],
                "selection": {}, "stage": "entry"}

    @_observational
    def prepare(self, row, modes, causes, selection):
        row.update(before=model_cpu_state(modes), reset_causes=list(causes),
                   selection=dict(selection), stage="model")

    @_observational
    def complete(self, row, modes, *, stage="returned", exception=None):
        after = model_cpu_state(modes)
        before = row.get("before") or {}
        reasons = list(row["reset_causes"])
        same_session = before.get("session_id") is not None and before.get("session_id") == after["session_id"]
        seed_before, seed_after = before.get("next_seed"), after["next_seed"]
        if before.get("session_id") is not None and not same_session:
            reasons.append("model_session_changed")
        if same_session and type(seed_before) is int and type(seed_after) is int:
            expected = (seed_before + 1) & 0xffffffff
            if seed_before > 0 and seed_after == 1 and expected != 1:
                reasons.append("model_seed_restart")
            elif exception is None and seed_after != expected:
                reasons.append("model_seed_non_unit_step")
        if (type(before.get("graph_entries")) is int and type(after["graph_entries"]) is int
                and after["graph_entries"] > before["graph_entries"]):
            reasons.append("new_graph_entry")
        if row["selection"].get("graph_requested") and exception is None:
            old_replays, new_replays = before.get("graph_replays"), after["graph_replays"]
            if same_session and type(old_replays) is int and new_replays == old_replays:
                reasons.append("graph_not_replayed")
        prior, requested = before.get("controls", {}), row["selection"]
        if same_session:
            for names, cause in ((("style", "local_tone", "local_structure"), "front_controls_changed"),
                                  (("auto_mask", "skin_structure"), "aux_controls_changed")):
                if any(name in requested and prior.get(name) != requested[name] for name in names):
                    reasons.append(cause)
        if exception is not None:
            reasons.append("python_exception")
        row.update(after=after, stage=stage, returned=exception is None, observations=list(dict.fromkeys(reasons)),
                   ended_monotonic_ns=time.monotonic_ns(),
                   exception=None if exception is None else {"type": type(exception).__name__, "message": str(exception)[:240]})
        with self._lock:
            self._serial += 1
            row["serial"] = self._serial
            self._frames.append(deepcopy(row))
            if reasons:
                self._event_serial += 1
                event = deepcopy(row)
                event["event_serial"] = self._event_serial
                self._events.append(event)

    def snapshot(self):
        with self._lock:
            return {"enabled": True, "serial": self._serial, "event_serial": self._event_serial,
                    "diagnostic_errors": self._errors, "last_diagnostic_error": deepcopy(self._last_error),
                    "frames": deepcopy(list(self._frames)), "events": deepcopy(list(self._events))}


def correlate(native_snapshot, python_snapshot):
    """Exact NR-ID join only. Missing observations never prove a healthy frame."""
    python_by_id = {row["frame_id"]: row for row in python_snapshot.get("frames", [])}
    result = []
    for frame in native_snapshot.get("frames", []):
        context = frame["context"]
        python = python_by_id.get(context["nr_frame_id"])
        joined = bool(python and python.get("native_join_valid") and
                      python["native"]["context"]["eval_id"] == context["eval_id"] and
                      python["native"]["context"]["sr_sequence"] == context["sr_sequence"])
        flags = context["stages"]["flags"]
        result.append({"eval_id": context["eval_id"], "sr_sequence": context["sr_sequence"],
                       "nr_frame_id": context["nr_frame_id"], "nr_processed": flags["result_valid"],
                       "nr_composed": flags["composite_submitted"], "nr_retired": flags["retired"],
                       "nr_skipped": not flags["processor_called"], "raw_fallback": flags["raw_fallback"],
                       "invalidated_without_submit": flags["invalidated"] and not flags["submitted"],
                       "native_reason": frame["reason"], "python_join": "exact" if joined else "unavailable",
                       "game_reset": bool(context["game_reset"]), "native_nr_reset": bool(context["nr_reset"]),
                       "color_age": context["color_age"], "source_mask": context["source_bits"],
                       "motion_probe_index": context["motion_probe_index"],
                       "motion_probe_recorded": flags["motion_probe_recorded"],
                       "reset_causes": python.get("reset_causes", []) if joined else None,
                       "observations": python.get("observations", []) if joined else None,
                       "exception": python.get("exception") if joined else None})
        if joined:
            result[-1]["model_cpu"] = {name: python.get(name) for name in ("before", "after", "stage", "selection")}
    return {"frames": result, "negative_evidence_complete": False,
            "limitation": "bounded rings and nonblocking drops; SR submission is not Present/pixel evidence"}


class WindowSampler:
    """An occasional existing health read emits at most eight anomaly windows.

    The two frame rings stay in RAM. Consumers can persist a window once its
    revision changes, plus one start/end counter sample. This schedules no poll.
    """
    def __init__(self):
        self._lock = threading.Lock()
        self._windows = {}
        self._native_serial = self._python_serial = 0
        self._revision = 0

    def sample(self, dll, python_snapshot):
        native = read_snapshot(dll)
        if native["status"] != "ok":
            return {"status": native["status"], "reason": native.get("reason"),
                    "revision": self._revision, "windows": [], "negative_evidence_complete": False}
        with self._lock:
            incoming = [event for event in native["events"] if event["serial"] > self._native_serial]
            if native["events"]:
                self._native_serial = native["events"][-1]["serial"]
            python_events = [event for event in python_snapshot.get("events", [])
                             if event["event_serial"] > self._python_serial]
            if python_snapshot.get("events"):
                self._python_serial = python_snapshot["events"][-1]["event_serial"]
            for event in incoming:
                context = event["context"]
                if context["sr_sequence"]:
                    key = context["sr_sequence"]
                    window = self._windows.setdefault(key, {"sr_sequence": key, "triggers": [], "frames": []})
                    window["triggers"].append({"kind": event["kind"], "reason": event["reason"],
                                               "eval_id": context["eval_id"], "nr_frame_id": context["nr_frame_id"],
                                               "source_mask": context["source_bits"], "color_age": context["color_age"],
                                               "native_steady_us": event["steady_us"],
                                               "detail": event["detail"], "detail2": event["detail2"]})
            unjoined_python_events = []
            for event in python_events:
                if event.get("native_join_valid"):
                    context = event["native"]["context"]
                    key = context["sr_sequence"]
                    window = self._windows.setdefault(key, {"sr_sequence": key, "triggers": [], "frames": []})
                    window["triggers"].append({"kind": "python", "nr_frame_id": event["frame_id"],
                                               "observations": event["observations"], "exception": event["exception"]})
                else:
                    unjoined_python_events.append({"nr_frame_id": event["frame_id"],
                                                    "observations": event["observations"],
                                                    "exception": event["exception"], "join": "unavailable"})
            self._windows = {key: self._windows[key] for key in sorted(self._windows)[-8:]}
            joined = correlate(native, python_snapshot)["frames"]
            changed = bool(incoming or python_events)
            for key, window in self._windows.items():
                adjacent = [row for row in joined if key - 1 <= row["sr_sequence"] <= key + 1]
                retained = {row["sr_sequence"]: row for row in window["frames"]}
                for row in adjacent:
                    old = retained.get(row["sr_sequence"])
                    if not old or old["python_join"] != "exact" or row["python_join"] == "exact":
                        retained[row["sr_sequence"]] = row
                merged = [retained[sequence] for sequence in sorted(retained)]
                if merged != window["frames"]:
                    window["frames"] = merged
                    changed = True
            if changed:
                self._revision += 1
            return {"status": "ok", "revision": self._revision, "changed": changed,
                    "current": native["current"], "native_header": native["native_header"],
                    "counters": native["counters"], "counters_transactional": False,
                    "windows": deepcopy(list(self._windows.values())),
                    "recent_frames": joined[-3:],
                    "unjoined_python_events": unjoined_python_events[-4:],
                    "python_diagnostic_errors": python_snapshot.get("diagnostic_errors", 0),
                    "negative_evidence_complete": False, "clock_domains_aligned": False}
