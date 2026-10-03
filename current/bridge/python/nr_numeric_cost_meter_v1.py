"""Opt-in live cost attribution; no model/capture edits and no GPU imports here.

Timestamps surround already captured XPUGraph.replay, not Python validation.
That interval includes the graph's kernels AND internal memory operations. It
is never called pure matrix time. CPU guards retain their original operations.
Only completed steady replay frames are published, with inclusive CPU nesting.
"""
from __future__ import annotations

from collections import deque
from contextlib import contextmanager
from functools import wraps
import os
import threading
import time


class CostMeter:
    def __init__(self):
        self._lock = threading.RLock()
        self._samples = deque(maxlen=128)
        self._patches = []
        self._modes = None
        self._binding = None
        self._torch = None
        self._replay_original = None
        self._replay_wrapper = None
        self._current = None
        self._thread = None
        self._epoch = 0
        self._total = 0
        self._error = None
        self._selected = ()
        self._handoff_ms = 0.0

    def note_handoff(self, value):
        self._handoff_ms = float(value)

    def _cpu_wrapper(self, target, name, label):
        original = getattr(target, name)
        had = name in target.__dict__
        previous = target.__dict__.get(name)

        @wraps(original)
        def measured(*args, **kwargs):
            row = self._current
            if row is None or self._thread != threading.get_ident():
                return original(*args, **kwargs)
            start = time.perf_counter_ns()
            try:
                return original(*args, **kwargs)
            finally:
                row[label] = row.get(label, 0.0) + (time.perf_counter_ns() - start) / 1e6

        setattr(target, name, measured)
        self._patches.append((target, name, had, previous, measured))

    def _detach(self):
        for target, name, had, previous, wrapper in reversed(self._patches):
            if target.__dict__.get(name) is wrapper:
                if had:
                    setattr(target, name, previous)
                else:
                    delattr(target, name)
        self._patches.clear()
        self._modes = None
        self._binding = None

    def _identity(self, host):
        modes = getattr(host, "_modes", None)
        session = getattr(modes, "session", None)
        if session is None or getattr(modes, "height", None) != 720:
            return None
        return (id(modes), id(session), id(session._stack.graph),
                id(getattr(modes, "numeric_cleanup_calls", None)))

    def _install_replay(self, torch):
        if self._torch is not None:
            if self._torch is not torch:
                raise RuntimeError("cost meter torch owner changed")
            return
        original = torch.xpu.XPUGraph.replay

        @wraps(original)
        def measured(graph, *args, **kwargs):
            row = self._current
            if (row is None or self._thread != threading.get_ident() or
                    id(graph) not in row["_network_ids"]):
                return original(graph, *args, **kwargs)
            try:
                capturing = torch.xpu.is_current_stream_capturing()
            except Exception as error:
                self._error = "capture-query: " + type(error).__name__ + ": " + str(error)
                return original(graph, *args, **kwargs)
            if capturing:
                return original(graph, *args, **kwargs)
            try:
                start = torch.xpu.Event(enable_timing=True)
                end = torch.xpu.Event(enable_timing=True)
                start.record()
            except Exception as error:
                self._error = "event-start: " + type(error).__name__ + ": " + str(error)
                return original(graph, *args, **kwargs)
            submitted = time.perf_counter_ns()
            # Exceptions from the actual production replay must propagate.
            value = original(graph, *args, **kwargs)
            submit_ms = (time.perf_counter_ns() - submitted) / 1e6
            try:
                end.record()
            except Exception as error:
                self._error = "event-end: " + type(error).__name__ + ": " + str(error)
                return value
            # A web change may retire the old session inside host.process.
            # Classification uses the pre-frame IDs, never a retired session.
            label = "network_graph_gpu_ms" if id(graph) in row["_network_ids"] else "other_graph_gpu_ms"
            cpu_label = "network_replay_submit_cpu_ms" if id(graph) in row["_network_ids"] else "other_replay_submit_cpu_ms"
            row[cpu_label] = row.get(cpu_label, 0.0) + submit_ms
            row["_events"].append((label, start, end))
            return value

        torch.xpu.XPUGraph.replay = measured
        self._torch = torch
        self._replay_original, self._replay_wrapper = original, measured

    def _attach(self, host):
        modes = getattr(host, "_modes", None)
        binding = self._identity(host)
        if binding is not None and binding == self._binding:
            return
        self._detach()
        self._epoch += 1
        with self._lock:
            self._samples.clear()
        if modes is None or modes.session is None or modes.height != 720:
            return
        torch = host._bridge.torch
        self._install_replay(torch)
        self._modes = modes
        self._binding = binding
        graph = modes.session._stack.graph
        owner = modes.numeric_cleanup_calls
        lifecycle = getattr(owner, "_replay_lifecycle_base_720", None)
        constants = getattr(lifecycle, "graph_constants", None)
        protected_validate = (constants is not None and constants.installed and
                              constants.graph is graph)
        # The constants lifecycle authenticates the real _validate call and
        # its direct forward caller. A timing wrapper would alter both. Keep
        # that callee untouched; an absent CPU metric is explicitly unavailable.
        if not protected_validate:
            self._cpu_wrapper(graph, "_validate", "graph_constant_guard_cpu_ms")
        if owner is not None:
            self._cpu_wrapper(owner, "validate_context", "numeric_guard_cpu_ms")
            for label, child in owner.children.items():
                self._cpu_wrapper(child, "validate", "child_" + label + "_guard_cpu_ms")
        self._selected = tuple(sorted(getattr(host, "_active_optimizations", ())))

    @contextmanager
    def frame(self, host, frame_id):
        # This context is entered inside the adapter's existing serial lock.
        # Never replace adapter.process: thread handoff checks its exact code.
        if self._error is not None:
            yield
            return
        try:
            self._attach(host)
        except Exception as error:
            self._error = "attach: " + type(error).__name__ + ": " + str(error)
            self.close()
            yield
            return
        modes = self._modes
        binding = self._binding
        usable = (modes is not None and modes.session is not None and
                  bool(modes.session._stack.graph.entries))
        row = {"frame_id": int(frame_id), "_events": [],
               "adapter_handoff_cpu_ms": self._handoff_ms,
               "_network_ids": {id(entry.graph) for entry in modes.session._stack.graph.entries.values()}} if usable else None
        self._current, self._thread = row, threading.get_ident()
        start = time.perf_counter_ns()
        success = False
        try:
            yield
            success = True
        finally:
            wall = (time.perf_counter_ns() - start) / 1e6
            self._current, self._thread = None, None
            if (self._error is None and success and row is not None and self._identity(host) == binding and
                    modes.last_combo_frame_route == "replay"):
                finished = time.perf_counter_ns()
                try:
                    self._finish(row, modes, wall)
                    stages = host.stage_times() if callable(getattr(host, "stage_times", None)) else None
                    if stages is not None:
                        for label in ("prepare", "model", "export"):
                            row["runtime_" + label + "_wall_ms"] = stages[label + "_last_ms"]
                except Exception as error:
                    self._error = "finish: " + type(error).__name__ + ": " + str(error)
                row["observer_finish_cpu_ms"] = (time.perf_counter_ns() - finished) / 1e6
                if self._error is None:
                    with self._lock:
                        self._total += 1
                        self._samples.append(row)
            if self._error is not None:
                self.close()

    def _finish(self, row, modes, wall):
        events = row.pop("_events")
        row.pop("_network_ids")
        counts = {}
        for label, start, end in events:
            # Production completes the network before history commit/export.
            # Do not add a probe synchronize or wait to the frame.
            if not end.query():
                raise RuntimeError("production returned before timed graph event completed")
            value = float(start.elapsed_time(end))
            if not 0 <= value < 10000:
                raise RuntimeError("invalid GPU timestamp interval")
            row[label] = row.get(label, 0.0) + value
            counts[label] = counts.get(label, 0) + 1
        if counts.get("network_graph_gpu_ms") != 1:
            raise RuntimeError("expected exactly one already captured network graph replay")
        row["process_wall_ms"] = wall
        row["network_replay_count"] = counts["network_graph_gpu_ms"]
        row["other_replay_count"] = counts.get("other_graph_gpu_ms", 0)
        # Shared static input/output traffic counts, not claimed traffic timings.
        entry = modes.session._stack.graph.last_entry
        row["static_input_copy_bytes"] = sum(t.numel() * t.element_size()
                                              for t in entry.inputs.values() if t is not None)
        row["output_clone_bytes"] = entry.output.numel() * entry.output.element_size()

    def snapshot(self):
        with self._lock:
            rows = [dict(row) for row in self._samples]
            epoch, total, selected, error = self._epoch, self._total, self._selected, self._error
        keys = sorted({key for row in rows for key in row
                       if key.endswith("_ms") or key.endswith("_bytes")})
        means = {key: sum(row.get(key, 0.0) for row in rows) / len(rows) for key in keys} if rows else {}
        return {"enabled": True, "error": error, "epoch": epoch,
                "completed_frames": total, "window_frames": len(rows),
                "optimizations_720": list(selected), "last": rows[-1] if rows else None,
                "averages": means,
                "semantics": {"network_graph_gpu_ms": "GPU timestamps around captured network replay, including internal memory operations; excludes external input copies and output clone",
                              "network_replay_submit_cpu_ms": "CPU duration inside original XPUGraph.replay submission; can overlap GPU time and is not additive",
                              "numeric_guard_cpu_ms": "inclusive production numeric validate_context CPU duration; child guard totals include other host-frame entries too and must not be added to this inclusive value",
                              "process_wall_ms": "host.process plus successful validation-frame exit wall duration, including bridge prepare/model/export; timestamps add observation overhead; excludes adapter handoff and observer readout",
                              "observer_finish_cpu_ms": "CPU readout after host.process, included in native bridge's outer frame timing",
                              "copy_bytes": "logical existing graph input-copy/output-clone bytes, not measured bandwidth or pure copy time"}}

    def close(self):
        self._detach()
        if self._torch is not None and self._torch.xpu.XPUGraph.replay is self._replay_wrapper:
            self._torch.xpu.XPUGraph.replay = self._replay_original
        self._torch = self._replay_original = self._replay_wrapper = None


_METER = None


def optional_meter():
    global _METER
    if os.environ.get("CYBERPUNK_NR_COST_METER", "0") != "1":
        return None
    if _METER is None:
        _METER = CostMeter()
    return _METER
