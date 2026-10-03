"""Decode the v1 native diagnostic ABI using an already-held DLL object.

Call read_snapshot(existing_web_module._native) from an occasional CPU sampler.
This module loads no DLL, schedules no polling, and performs no file/GPU I/O.
The first call caches the export and its ctypes signature on that DLL object.
Native steady_us and Python monotonic_ns retain separate clock domains; their
absolute epochs/precision have not been calibrated and must not be equated.
"""
from __future__ import annotations

import ctypes as C
import threading
import time

EXPORT = "NRB_GetPeriodicFlashDiag"
VERSION = 1
CAPACITY = 64
COUNTERS = (
    "seen", "eligible", "source_proven", "delegated", "submitted", "nr_composited",
    "original_fallback", "excluded_original", "game_reset", "generation_mismatch",
    "invalidated", "failure",
)
REASONS = (
    "none", "scope_or_feature", "source_state", "record_rejected", "tail_unverified",
    "disarmed", "arguments", "attempt_budget", "textures", "pending_full", "allocation",
    "motion_contract", "device", "record_resources", "hdr_missing", "motion_proxy",
    "record_close", "original_evaluate", "processor_missing", "disabled", "latched_failure",
    "prepared_wait", "processor_result", "composite_or_wait", "completion_signal",
    "retire", "generation", "list_reset",
)
EVENT_KINDS = ("record_reject", "fallback", "reset", "failure", "generation", "invalidated")
SOURCE_BITS = (
    "reset_observed", "closed", "oversized_barrier", "color_seen", "plain_barrier",
    "all_subresources", "exact_psr", "age_le_64", "proven", "incomplete",
)


class Context(C.Structure):
    _fields_ = [
        ("eval_id", C.c_uint64), ("list", C.c_uint64), ("generation", C.c_uint64),
        ("feature_id", C.c_uint64), ("route", C.c_uint32), ("evidence", C.c_uint32),
        ("effective_route", C.c_uint32), ("eligible", C.c_uint32), ("sr_slot", C.c_uint32),
        ("source_bits", C.c_uint32), ("color_after", C.c_uint32), ("color_age", C.c_uint32),
        ("game_reset", C.c_uint32), ("enabled", C.c_uint32), ("history", C.c_uint32),
    ]


class Event(C.Structure):
    _fields_ = [
        ("serial", C.c_uint64), ("steady_us", C.c_uint64),
        ("composited_before", C.c_uint64), ("detail", C.c_uint64), ("detail2", C.c_uint64),
        ("context", Context), ("kind", C.c_uint32), ("reason", C.c_uint32),
    ]


class Snapshot(C.Structure):
    _fields_ = [
        ("abi_size", C.c_uint32), ("version", C.c_uint32),
        ("retained", C.c_uint64), ("overwritten", C.c_uint64), ("dropped", C.c_uint64),
        ("counters", C.c_uint64 * len(COUNTERS)), ("reasons", C.c_uint64 * len(REASONS)),
        ("events", Event * CAPACITY),
    ]


# Fixed-width fields, MSVC x64 default alignment; no packed layout. The sizes
# follow the patch header, including Context's four trailing padding bytes.
ABI_SIZE = 8544
if (C.sizeof(Context), C.sizeof(Event), C.sizeof(Snapshot)) != (80, 128, ABI_SIZE):
    raise RuntimeError("unsupported v1 native diagnostic structure layout")
if any(C.alignment(cls) != 8 for cls in (Context, Event, Snapshot)):
    raise RuntimeError("unsupported v1 native diagnostic structure alignment")

_CACHE_KEY = "_nrb_periodic_flash_snapshot_v1_binding"
_CACHE_MISSING = object()
_BIND_LOCK = threading.Lock()


def _binding(dll):
    if dll is None:
        return ("unavailable", "no_existing_dll")
    try:
        cache = vars(dll)
    except TypeError:
        return ("unavailable", "no_dll_object")
    bound = cache.get(_CACHE_KEY, _CACHE_MISSING)
    if bound is not _CACHE_MISSING:
        return bound
    with _BIND_LOCK:
        bound = cache.get(_CACHE_KEY, _CACHE_MISSING)
        if bound is _CACHE_MISSING:
            try:
                function = getattr(dll, EXPORT)
                if not callable(function):
                    raise TypeError("native snapshot export is not callable")
                function.argtypes = [C.POINTER(Snapshot)]
                function.restype = C.c_int
                bound = ("ready", function)
            except AttributeError:
                bound = ("unavailable", "missing_export")
            except Exception as exc:
                bound = ("error", "binding_failed: " + str(exc))
            cache[_CACHE_KEY] = bound
    return bound


def _decode_event(event):
    context = {name: int(getattr(event.context, name)) for name, _ in Context._fields_}
    mask = context["source_bits"]
    return {
        "serial": int(event.serial), "native_steady_us": int(event.steady_us),
        "composited_before": int(event.composited_before),
        "detail": int(event.detail), "detail2": int(event.detail2),
        "kind_id": int(event.kind),
        "kind": EVENT_KINDS[event.kind] if event.kind < len(EVENT_KINDS) else "unknown",
        "reason_id": int(event.reason),
        "reason": REASONS[event.reason] if event.reason < len(REASONS) else "unknown",
        "context": context,
        "source_bits": {
            "mask": mask,
            "flags": {name: bool(mask & (1 << i)) for i, name in enumerate(SOURCE_BITS)},
            "unknown_mask": mask & ~((1 << len(SOURCE_BITS)) - 1),
        },
    }


def read_snapshot(dll):
    """Read once; return JSON-serializable ok/unavailable/busy/error evidence.

    dll is the existing ctypes CDLL object, not a path or numeric OS handle.
    Passing a web module's _native=None returns unavailable without initializing
    its lazy _api(). Signature binding is cached, including missing exports.
    Counters are lifetime values, not inferred per-frame deltas or acceptance.
    """
    result = {
        "status": "unavailable", "export": EXPORT,
        "expected_abi": {"size": ABI_SIZE, "version": VERSION, "capacity": CAPACITY},
        "python_sample_time": None, "clock_domains_aligned": False,
        "counters": {}, "reasons": {}, "events": [],
    }
    state, value = _binding(dll)
    if state != "ready":
        result["status"] = state
        result["error" if state == "error" else "reason"] = value
        return result
    snapshot = Snapshot()
    snapshot.abi_size = ABI_SIZE
    snapshot.version = VERSION
    started = time.monotonic_ns()
    try:
        returned = value(C.byref(snapshot))
    except Exception as exc:
        result.update(status="error", error="read_failed: " + str(exc))
        return result
    finally:
        result["python_sample_time"] = {
            "monotonic_ns_start": started, "monotonic_ns_end": time.monotonic_ns(),
        }
    if returned == 0:
        result["status"] = "busy"
        return result
    if returned != 1:
        result.update(status="error", error="invalid_export_result")
        return result
    result["native_header"] = {
        name: int(getattr(snapshot, name))
        for name in ("abi_size", "version", "retained", "overwritten", "dropped")
    }
    if snapshot.abi_size != ABI_SIZE:
        result.update(status="error", error="invalid_abi_size")
    elif snapshot.version != VERSION:
        result.update(status="error", error="invalid_version")
    elif snapshot.retained > CAPACITY:
        result.update(status="error", error="invalid_retained")
    else:
        result.update(
            status="ok",
            counters={name: int(snapshot.counters[i]) for i, name in enumerate(COUNTERS)},
            reasons={name: int(snapshot.reasons[i]) for i, name in enumerate(REASONS)},
            events=[_decode_event(snapshot.events[i]) for i in range(snapshot.retained)],
        )
    return result
