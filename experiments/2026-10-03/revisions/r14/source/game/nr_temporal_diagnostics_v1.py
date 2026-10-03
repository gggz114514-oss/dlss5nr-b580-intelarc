"""Small CPU event buffer for diagnosing an occasional game NR flash.

Reads Python counters only. It does not inspect tensor values, synchronize a
GPU, change history, or write a per-frame log.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
import threading
import time


def reset_causes(*, candidate_changed, geometry_changed, game_reset,
                 last_frame, frame_id, last_history_mode, history_mode,
                 last_graph_replay, graph_replay, last_backend_variant,
                 backend_variant):
    checks = (
        ("candidate_changed", candidate_changed),
        ("source_geometry_changed", geometry_changed),
        ("game_requested_reset", bool(game_reset)),
        ("first_nr_frame" if last_frame is None else "nr_frame_gap",
         last_frame != frame_id - 1),
        ("history_mode_changed", last_history_mode != history_mode),
        ("graph_setting_changed", last_graph_replay != graph_replay),
        ("backend_variant_changed", last_backend_variant != backend_variant),
        ("reset_every_frame_mode", history_mode == "reset"),
    )
    return tuple(name for name, active in checks if active)


def model_state(modes):
    session = getattr(modes, "session", None)
    stack = getattr(session, "_stack", None)
    model = getattr(stack, "model", None)
    graph = getattr(stack, "graph", None)
    seed = getattr(model, "_next_seed", None)
    replays = getattr(graph, "replays", None)
    entries = getattr(graph, "entries", None)
    return {
        "next_seed": seed if type(seed) is int else None,
        "graph_entries": len(entries) if isinstance(entries, dict) else None,
        "graph_replays": replays if type(replays) is int else None,
    }


class TemporalDiagnostics:
    def __init__(self, *, enabled=False, event_limit=32):
        self.enabled = bool(enabled)
        self._events = deque(maxlen=event_limit)
        self._lock = threading.Lock()
        self._completed = 0
        self._resets = 0
        self._captures = 0
        self._last = None
        self._sequence = 0

    def record(self, frame_id, causes, before, after, *, input_height, selection=None):
        if not self.enabled:
            return
        # Reading these integers has no device transfer or GPU synchronization.
        restarted = (type(before["next_seed"]) is int and
                     before["next_seed"] > 1 and after["next_seed"] == 1)
        prior_entries, new_entries = before["graph_entries"], after["graph_entries"]
        captured = (type(new_entries) is int and
                    new_entries > (prior_entries if type(prior_entries) is int else 0))
        event = None
        if causes or restarted or captured:
            event = {"monotonic_ms": round(time.monotonic() * 1000, 3),
                     "frame_id": frame_id, "input_height": input_height,
                     "requested_reset_causes": list(causes),
                     "model_seed_restart": restarted,
                     "new_graph_entry": captured,
                     "before": dict(before), "after": dict(after),
                     "selection": deepcopy(selection)}
        with self._lock:
            self._completed += 1
            self._resets += bool(causes)
            self._captures += captured
            self._last = {"frame_id": frame_id, "input_height": input_height,
                          **after}
            if event is not None:
                self._sequence += 1
                event["sequence"] = self._sequence
                self._events.append(event)

    def snapshot(self):
        with self._lock:
            return {"enabled": self.enabled,
                    "completed_python_frames": self._completed,
                    "requested_reset_frames": self._resets,
                    "graph_capture_frames": self._captures,
                    "event_sequence": self._sequence,
                    "last": dict(self._last) if self._last is not None else None,
                    "events": deepcopy(list(self._events))}
