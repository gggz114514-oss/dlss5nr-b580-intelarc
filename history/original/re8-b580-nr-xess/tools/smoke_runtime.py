"""Exercise the relocated RE8 runtime; optionally collect a steady baseline."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import statistics
import sys
import time
import traceback


def tree_sha256(folder: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(folder.rglob("*.py")):
        digest.update(path.relative_to(folder).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile without samples")
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def validate_preflight(path: Path) -> dict:
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("game_closed") is not True or evidence.get("re8_processes") != []:
        raise RuntimeError("Preflight did not confirm that RE8 is closed")
    if (evidence.get("gpu_idle") is not True or
            evidence.get("competing_gpu_load") is not False):
        raise RuntimeError("Preflight found meaningful competing GPU load")
    policy = evidence.get("idle_policy")
    if not isinstance(policy, dict) or not policy.get("definition"):
        raise RuntimeError("Preflight did not record its GPU-idle decision policy")
    samples = evidence.get("gpu_engine_samples")
    if not isinstance(samples, list) or len(samples) < 5:
        raise RuntimeError("Preflight needs at least five GPU-idle samples")
    for sample in samples:
        maximum = sample.get("max_utilization_percent")
        if (maximum is None or not isinstance(sample.get("active_engines"), list) or
                sample.get("engine_rows", 0) < 1):
            raise RuntimeError("Preflight GPU activity samples are incomplete")
    if policy.get("accepted_status") == "background-display-load-accepted":
        basis = evidence.get("acceptance_basis")
        if not isinstance(basis, dict):
            raise RuntimeError("Accepted desktop-load preflight has no audit basis")
        threshold = basis.get("display_only_threshold_percent")
        if threshold is None and (basis.get("source_file") == "preflight.json" and
                                  policy.get("threshold_percent") == 4.0):
            # Preserve the reviewed first ABBA fixture, written before the
            # geometry-only acceptance field was introduced.
            threshold = 4.0
        if (not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or
                not 0 < threshold <= 6.0):
            raise RuntimeError("Accepted desktop-load ceiling must be an acceptance-basis value <=6 percent")
        if policy.get("threshold_percent") != threshold:
            raise RuntimeError("Accepted preflight policy threshold does not match its audit basis")
        if policy.get("activity_reporting_threshold_percent") != 0.5:
            raise RuntimeError("Accepted preflight must document its activity threshold")
        processes = basis.get("attributed_processes")
        if (not isinstance(processes, list) or len(processes) != 2 or
                any(not isinstance(item, dict) for item in processes) or
                [(item.get("name"), item.get("role")) for item in processes] != [
                    ("dwm.exe", "Windows desktop compositor"),
                    ("ChatGPT.exe", "Codex UI process")]):
            raise RuntimeError("Accepted GPU activity is not limited to reviewed DWM/Codex roles")
        allowed = {}
        for item in processes:
            pid = item.get("pid")
            if type(pid) is not int or pid <= 0 or pid in allowed:
                raise RuntimeError("Accepted DWM/Codex PIDs are invalid")
            allowed[pid] = item["name"]
        if threshold == 6.0 and (
                basis.get("acceptance_scope") != "geometry-A-C-C-A-20260925-session-only" or
                not basis.get("noise_caveat")):
            raise RuntimeError("Six-percent acceptance needs its session scope and noise caveat")
        if evidence.get("competing_compute_processes") != []:
            raise RuntimeError("Accepted preflight lists competing compute processes")
        raw_path = path.parent / str(basis.get("source_file", ""))
        if not raw_path.is_file():
            raise RuntimeError("Accepted preflight raw source file is missing")
        raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        if raw_hash.casefold() != str(basis.get("source_sha256", "")).casefold():
            raise RuntimeError("Accepted preflight raw source hash does not match")
        raw = json.loads(raw_path.read_text(encoding="utf-8"))
        raw_max = max(sample["max_utilization_percent"]
                      for sample in raw.get("gpu_engine_samples", []))
        if (raw.get("gpu_idle") is not False or
                raw.get("competing_gpu_load") is not True or
                raw_max > threshold or
                abs(raw_max - basis.get("raw_max_engine_utilization_percent", -1)) > 1e-6):
            raise RuntimeError("Accepted preflight does not faithfully preserve the strict raw result")
        if samples != raw.get("gpu_engine_samples"):
            raise RuntimeError("Accepted preflight changed raw GPU samples")
        for sample in samples:
            if sample["max_utilization_percent"] > threshold:
                raise RuntimeError(f"Accepted background GPU utilization exceeds {threshold:g} percent")
            for engine in sample["active_engines"]:
                match = re.match(r"pid_(\d+)_", str(engine.get("instance", "")))
                if match is None or int(match.group(1)) not in allowed:
                    raise RuntimeError("Accepted preflight contains an unreviewed active GPU process")
                if engine["utilization_percent"] > threshold:
                    raise RuntimeError(f"Accepted active-engine utilization exceeds {threshold:g} percent")
        if abs(max(sample["max_utilization_percent"] for sample in samples) - raw_max) > 1e-6:
            raise RuntimeError("Accepted samples differ from their raw preflight source")
    b580 = evidence.get("b580")
    if not isinstance(b580, dict):
        raise RuntimeError("Preflight did not identify the Intel Arc B580")
    device_name = b580.get("name", b580.get("Name", ""))
    driver_version = b580.get("driver_version", b580.get("DriverVersion"))
    if "B580" not in str(device_name):
        raise RuntimeError("Preflight did not identify the Intel Arc B580")
    if not driver_version:
        raise RuntimeError("Preflight did not record the B580 driver version")
    b580["name"] = device_name
    b580["driver_version"] = driver_version
    return evidence


def validate_byte_audit_preflight(path: Path) -> dict:
    """Permit desktop activity for shape/replay evidence, never for timings."""
    evidence = json.loads(path.read_text(encoding="utf-8"))
    if evidence.get("game_closed") is not True or evidence.get("re8_processes") != []:
        raise RuntimeError("Byte audit requires RE8 to be closed")
    if evidence.get("competing_compute_processes") != []:
        raise RuntimeError("Byte audit found a competing compute process")
    samples = evidence.get("gpu_engine_samples")
    if not isinstance(samples, list) or len(samples) < 5:
        raise RuntimeError("Byte audit requires five raw GPU samples")
    if any(not isinstance(sample.get("max_utilization_percent"), (int, float))
           for sample in samples):
        raise RuntimeError("Byte-audit GPU sample is incomplete")
    b580 = evidence.get("b580")
    if not isinstance(b580, dict) or "B580" not in str(b580.get("Name", b580.get("name"))):
        raise RuntimeError("Byte audit did not identify the B580")
    device_name = b580.get("Name", b580.get("name"))
    driver_version = b580.get("DriverVersion", b580.get("driver_version"))
    if not driver_version:
        raise RuntimeError("Byte audit did not record the B580 driver")
    evidence["b580"] = {"name": device_name, "driver_version": driver_version,
                        "pnp_device_id": b580.get("PNPDeviceID", b580.get("pnp_device_id"))}
    evidence["byte_audit_only"] = True
    evidence["timing_valid"] = False
    evidence["timing_reason"] = (
        "Raw preflight exceeded the idle threshold; graph buffer sizes, "
        "replay counters and output hashes remain interpretable, timings do not")
    return evidence


def invoke_frame(host, device, queue, color, motion, frame_id: int,
                 game_reset: int, width: int, height: int) -> dict:
    started = time.perf_counter_ns()
    resource, fence, value = host.process(
        device, queue, color, motion, frame_id, game_reset, width, height)
    if not all((resource, fence, value)):
        raise RuntimeError(f"Frame {frame_id} returned an invalid GPU output")
    host.retire(fence, value)
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    return {
        "frame_id": frame_id,
        "elapsed_ms": elapsed_ms,
        "graph_replayed": bool(host._modes.last_graph_used),
        "mode_height": host._modes.height,
    }


def tensor_sha256(torch, tensor) -> dict:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("The production model path did not return a tensor")
    cpu = tensor.detach().contiguous().cpu()
    return {
        "sha256": hashlib.sha256(cpu.numpy().tobytes(order="C")).hexdigest(),
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
        "stage": "composited model output before bridge export",
    }


class StageProbe:
    """Test-only wall-clock boundaries around production calls."""

    def __init__(self, host, synchronize_xpu, *, geometry_profile=False):
        if host._bridge is None or host._modes is None:
            raise RuntimeError("Stage probe requires a warmed production host")
        if not callable(synchronize_xpu):
            raise TypeError("Stage probe requires an XPU synchronization callback")
        self.current = None
        self.originals = []
        self.synchronize_xpu = synchronize_xpu
        self.geometry_profile = geometry_profile
        self._wrap(host._bridge, "prepare", "bridge_prepare")
        self._wrap(host._modes, "process", "nr_process")
        if geometry_profile:
            if host._modes.geometry is None:
                raise RuntimeError("Geometry probe requires an active warmed mode")
            self._wrap(host._modes.geometry, "prepare", "geometry_prepare")
            self._wrap(host._modes.geometry, "composite", "geometry_composite")
        self._wrap(host._bridge, "export", "bridge_export")
        self._wrap(host._bridge, "output", "bridge_output")
        self._wrap(host, "retire", "retire_registration")

    def _wrap(self, owner, method, label):
        original = getattr(owner, method)
        self.originals.append((owner, method, original))

        def timed(*args, **kwargs):
            started = time.perf_counter_ns()
            try:
                result = original(*args, **kwargs)
                if ((label == "nr_process" and not self.geometry_profile) or
                        (self.geometry_profile and label in
                         ("geometry_prepare", "geometry_composite"))):
                    # The production export call waits on the same XPU queue.
                    # Move that deferred model work to this boundary so the
                    # NR stage includes execution, not just graph submission.
                    self.synchronize_xpu()
                return result
            finally:
                if self.current is not None:
                    if label in self.current:
                        raise RuntimeError(f"Stage {label} called twice in one frame")
                    self.current[label] = (
                        time.perf_counter_ns() - started) / 1_000_000.0

        setattr(owner, method, timed)

    def begin(self):
        self.current = {}

    def end(self):
        result = self.current
        self.current = None
        if result is None or len(result) != (7 if self.geometry_profile else 5):
            raise RuntimeError(f"Missing expected stage timings: {result}")
        return result

    def close(self):
        self.current = None
        for owner, method, original in reversed(self.originals):
            setattr(owner, method, original)
        self.originals.clear()


def summarize_stages(samples: list[dict]) -> dict:
    names = tuple(samples[0]["stages_ms"])
    result = {}
    for name in names:
        values = [sample["stages_ms"][name] for sample in samples]
        shares = [100 * sample["stages_ms"][name] / sample["elapsed_ms"]
                  for sample in samples]
        result[name] = {
            "median_ms": statistics.median(values),
            "p95_ms": percentile(values, 0.95),
            "median_share_percent": statistics.median(shares),
        }
    return result


def graph_io_snapshot(host) -> dict:
    """Read live graph buffers and replay counts without touching tensor data."""
    session = host._modes.session
    if session is None or session._stack is None:
        raise RuntimeError("Graph I/O audit requires an active session")
    graph = session._stack.graph
    entries = []
    for index, (key, entry) in enumerate(graph.entries.items()):
        input_bytes = {
            name: int(t.numel() * t.element_size())
            for name, t in entry.inputs.items() if t is not None
        }
        output_bytes = int(entry.output.numel() * entry.output.element_size())
        entries.append({
            "index": index,
            "key": str(key),
            "replays": int(entry.replays),
            "input_bytes": input_bytes,
            "output_bytes": output_bytes,
        })
    return {"total_replays": int(graph.replays), "entries": entries,
            "history_warp_graph_replays": int(session._stack.warp.replays)}


def summarize_graph_io(before: dict, after: dict, measured_frames: int) -> dict:
    """Count known explicit copies, not inferred memory-controller traffic."""
    prior = {entry["key"]: entry for entry in before["entries"]}
    items = []
    known_bytes = 0
    replay_delta = after["total_replays"] - before["total_replays"]
    if replay_delta != measured_frames:
        raise RuntimeError(f"Expected {measured_frames} graph replays, got {replay_delta}")
    for entry in after["entries"]:
        old = prior.get(entry["key"])
        if old is None:
            raise RuntimeError("Graph captured a new shape during the measured interval")
        if (old["input_bytes"] != entry["input_bytes"] or
                old["output_bytes"] != entry["output_bytes"]):
            raise RuntimeError("Captured graph buffer sizes changed")
        count = entry["replays"] - old["replays"]
        if count < 0:
            raise RuntimeError("Graph replay count decreased")
        input_copy_bytes = sum(entry["input_bytes"].values())
        # graph_front_v5 captures output.copy_(temporary_result) inside replay;
        # graph_front_v1 also clones the persistent output after replay.
        output_copy_bytes = 2 * entry["output_bytes"]
        per_replay = input_copy_bytes + output_copy_bytes
        known_bytes += count * per_replay
        items.append({**entry, "measured_replays": count,
                      "explicit_input_copy_bytes_per_replay": input_copy_bytes,
                      "explicit_output_copy_bytes_per_replay": output_copy_bytes,
                      "known_explicit_copy_bytes_per_replay": per_replay})
    if sum(item["measured_replays"] for item in items) != replay_delta:
        raise RuntimeError("Per-entry graph replay counts do not add up")
    warp_delta = (after["history_warp_graph_replays"] -
                  before["history_warp_graph_replays"])
    if warp_delta:
        raise RuntimeError("Unexpected square history-warp graph activity")
    return {
        "scope": "explicit graph-front static-input copies, captured output copy, and returned output clone",
        "excludes": "all other model operators, implicit memory traffic, geometry, bridge, and non-graph copies",
        "no_tensor_readback_or_added_synchronization_in_measurement": True,
        "measured_frames": measured_frames,
        "measured_graph_replays": replay_delta,
        "square_history_warp_graph_replays": warp_delta,
        "known_explicit_copy_bytes_total": known_bytes,
        "known_explicit_copy_bytes_per_frame": known_bytes / measured_frames,
        "entries": items,
    }


def run_eager_group_probe(host, torch, device, queue, color, motion, *,
                          width: int, height: int, input_height: int,
                          first_frame_id: int) -> dict:
    """Use the model's existing progress boundary for an exploratory group map.

    The progress path is eager and synchronizes after every group. It can rank
    candidate groups, but its milliseconds are not graph-replay timings.
    """
    from nr_game_controls import Settings

    class Panel:
        def __init__(self, replay):
            self.replay = replay

        def snapshot(self):
            return Settings(input_size=input_height, display_strength=1.0,
                            history_mode="fused", graph_replay=self.replay)

        def mark_active(self, *args, **kwargs):
            pass

    modes = host._modes
    model = modes.session._stack.model
    previous_panel = host._panel
    had_front = "_forward_front" in model.__dict__
    previous_front = model.__dict__.get("_forward_front")
    front = model._forward_front
    original_process = modes.process
    captures = []
    current = None
    eager = []
    graph = []

    def observed_process(*args, **kwargs):
        result = original_process(*args, **kwargs)
        captures.append(result)
        return result

    def marked_front(*args, **kwargs):
        if current is None:
            raise RuntimeError("Group mark outside a probe frame")
        if kwargs.get("progress") is not None:
            raise RuntimeError("Unexpected pre-existing progress callback")
        started = time.perf_counter_ns()
        last = started

        def mark(name):
            nonlocal last
            now = time.perf_counter_ns()
            current["groups_ms"].append({"name": name,
                                         "elapsed_ms": (now - last) / 1_000_000.0})
            last = now

        kwargs["progress"] = mark
        result = front(*args, **kwargs)
        if not current["groups_ms"] or current["groups_ms"][-1]["name"] != "RGB":
            raise RuntimeError("Model did not finish its progress-marked front")
        return result

    modes.process = observed_process
    try:
        for replay in (False, True):
            host._panel = Panel(replay)
            if replay:
                if had_front:
                    model._forward_front = previous_front
                else:
                    del model._forward_front
            else:
                model._forward_front = marked_front
            for offset, reset in enumerate((1, 0)):
                current = {"reset": bool(reset), "groups_ms": []}
                frame_id = first_frame_id + (2 if replay else 0) + offset
                sample = invoke_frame(host, device, queue, color, motion,
                                      frame_id, reset, width, height)
                if sample["graph_replayed"] != replay:
                    raise RuntimeError("Group probe executed the wrong graph/eager route")
                if len(captures) != (2 if replay else 0) + offset + 1:
                    raise RuntimeError("Group probe missed a model output")
                current["output_sha256"] = tensor_sha256(torch, captures[-1])["sha256"]
                (graph if replay else eager).append(current)
                current = None
    finally:
        modes.process = original_process
        host._panel = previous_panel
        if had_front:
            model._forward_front = previous_front
        else:
            model.__dict__.pop("_forward_front", None)

    matched = [a["output_sha256"] == b["output_sha256"]
               for a, b in zip(eager, graph)]
    if matched != [True, True]:
        raise RuntimeError(f"Eager and graph probe output differs: {matched}")
    names = [part["name"] for part in eager[0]["groups_ms"]]
    if names != [part["name"] for part in eager[1]["groups_ms"]]:
        raise RuntimeError("Group progress names changed between reset and temporal")
    return {
        "scope": "2 eager diagnostic frames and 2 graph replays after the baseline",
        "timing_interpretation": (
            "Eager progress callback synchronizes after every model group; CPU dispatch, "
            "GPU waits and competing desktop activity are included. Do not interpret "
            "as production graph-replay group times or add these to the graph baseline."),
        "group_names": names,
        "eager_and_graph_output_byte_identical": True,
        "eager_frames": eager,
        "graph_frames": graph,
    }


def install_game_fullsize_override(path: Path, runtime_root: Path) -> None:
    """Load a test-only game module without changing the production runtime."""
    module_name = "nr_game_fullsize"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, path.resolve())
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load game module override: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    # The candidate file is isolated in E:, but its unchanged collaborators
    # and duplicate-backend checks must resolve from the actual G: runtime.
    module.ROOT = runtime_root.resolve()
    module.GEOMETRY_MODULE = module.ROOT / "game"


def run_steady_baseline(host, torch, device, queue, color, motion, *,
                        width: int, height: int, input_height: int,
                        warmup_frames: int, measure_frames: int, row: dict,
                        stage_profile: bool = False,
                        geometry_profile: bool = False,
                        graph_io_audit: bool = False) -> None:
    from nr_game_controls import Settings
    from nr_game_fullsize import mode_for

    class BaselinePanel:
        def snapshot(self):
            return Settings(input_size=input_height, display_strength=1.0,
                            history_mode="fused", graph_replay=True)

        def mark_active(self, *args, **kwargs):
            pass

    host._panel = BaselinePanel()
    selected = host._panel.snapshot()
    if (selected.input_size != input_height or selected.history_mode != "fused" or
            selected.graph_replay is not True or selected.display_strength != 1.0):
        raise RuntimeError("Steady baseline settings do not match the requested RE8 path")

    row["mode"] = "steady-baseline"
    row["pipeline"] = {
        "source": [width, height],
        "input_height": input_height,
        "active_model_image": [width, input_height],
        "history_mode": "fused",
        "graph_replay_requested": True,
        "display_strength": 1.0,
        "style": 0,
        "model_intensity": 1.0,
        "controls": "RE8 control defaults",
        "fixture": "nr_texture_fixture_re8",
        "runtime_route": "G runtime nr_game_pre_xess_host.process",
        "execution_scope": (
            "offline production RE8 bridge/model path using a synthetic native fixture; "
            "game and downstream XeSS consumer are not running"),
        "production_output_path": "source prepare -> controlled model -> bridge export",
    }
    mode = mode_for(input_height)
    row["pipeline"]["geometry"] = {
        "active_hw": list(mode.active),
        "model_hw_with_production_padding": list(mode.model),
        "internal_hw": list(mode.internal),
        "inset": list(mode.inset),
    }
    if input_height == 540 and mode.active != (height, width):
        raise RuntimeError("540p mode is not the full 960x540 active source")

    warmup = []
    warmup_started = time.perf_counter()
    for frame_id in range(1, warmup_frames + 1):
        sample = invoke_frame(host, device, queue, color, motion, frame_id,
                              int(frame_id == 1), width, height)
        if (not sample["graph_replayed"] and frame_id == warmup_frames):
            raise RuntimeError("Graph replay was not active by the end of warm-up")
        warmup.append(sample)
        row["frames"] += 1
    if not any(sample["graph_replayed"] for sample in warmup):
        raise RuntimeError("Warm-up did not observe an actual graph replay")

    disk_only = getattr(host, "_disk_only", None)
    if disk_only is None or not isinstance(getattr(disk_only, "hits", None), int):
        raise RuntimeError("Production DiskOnly cache guard was not active")
    hits_before = disk_only.hits
    graph_before = graph_io_snapshot(host) if graph_io_audit else None
    row["warmup"] = {
        "frames": len(warmup),
        "minimum_required": 16,
        "first_frame_reset": True,
        "remaining_frames_reset": False,
        "elapsed_s": time.perf_counter() - warmup_started,
        "graph_replays_observed": sum(sample["graph_replayed"] for sample in warmup),
        "samples": warmup,
        "disk_only_hits_after_warmup": hits_before,
    }

    samples = []
    probe = (StageProbe(host, torch.xpu.synchronize,
                        geometry_profile=geometry_profile)
             if stage_profile else None)
    try:
        for frame_id in range(warmup_frames + 1,
                              warmup_frames + measure_frames + 1):
            if probe is not None:
                probe.begin()
            sample = invoke_frame(host, device, queue, color, motion, frame_id,
                                  0, width, height)
            if probe is not None:
                sample["stages_ms"] = probe.end()
                if geometry_profile:
                    sample["stages_ms"]["model_segment"] = (
                        sample["stages_ms"]["nr_process"] -
                        sample["stages_ms"]["geometry_prepare"] -
                        sample["stages_ms"]["geometry_composite"])
                sample["stages_ms"]["unclassified"] = (
                    sample["elapsed_ms"] - sum(
                        sample["stages_ms"][name] for name in (
                            "bridge_prepare", "nr_process", "bridge_export",
                            "bridge_output", "retire_registration")))
            row["frames"] += 1
            if not sample["graph_replayed"]:
                raise RuntimeError(
                    f"Measured frame {frame_id} requested replay but did not replay")
            if sample["mode_height"] != input_height:
                raise RuntimeError(
                    f"Measured frame {frame_id} selected {sample['mode_height']}p, "
                    f"expected {input_height}p")
            samples.append(sample)
    finally:
        if probe is not None:
            probe.close()

    if graph_io_audit:
        row["graph_io_audit"] = summarize_graph_io(
            graph_before, graph_io_snapshot(host), measure_frames)

    if stage_profile:
        row["stage_profile"] = {
            "timing_kind": "CPU wall-clock boundaries with explicit XPU synchronization",
            "geometry_profile": geometry_profile,
            "synchronization_boundary": (
                "In geometry-profile runs, XPU synchronization follows geometry.prepare "
                "and geometry.composite; the model itself synchronizes within execute. "
                "Otherwise, stage-profile synchronizes after _modes.process. These "
                "test-only syncs change host overhead and may wait on unrelated work."),
            "interpretation": (
                "bridge_prepare includes D3D pack/submit and XPU transfer; "
                "nr_process includes geometry preparation, model graph replay, "
                "and completion of queued XPU work; bridge_export includes any "
                "contiguous-output work plus XPU transfer and D3D unpack/submit. "
                "Bridge timings combine conversion, transfer, synchronization, "
                "and submission. Geometry-profile's model_segment subtracts the "
                "synchronized geometry stages from nr_process and still includes "
                "Python, graph dispatch, and model-internal data movement. These are "
                "not GPU kernel times or an arithmetic-vs-memory split."),
            "stages": summarize_stages(samples),
        }

    timings = [sample["elapsed_ms"] for sample in samples]
    row["steady_timing"] = {
        "measured_frames": len(samples),
        "wall_scope": (
            "offline G-runtime RE8 bridge/model path: host.process plus host.retire "
            "call-return wall time, including source preparation, model, and export"),
        "retire_semantics": (
            "host.retire registers the returned retirement fence/value and returns; "
            "it does not wait for the downstream XeSS consumer"),
        "downstream_xess_consumer_waited": False,
        "full_in_game_xess_frame_wall": False,
        "sample_clock": "perf_counter_ns from immediately before host.process through host.retire call return",
        "measured_frames_reset": False,
        "history_mode": "fused",
        "graph_replay_requested": True,
        "samples": samples,
        "median_ms": statistics.median(timings),
        "p50_ms": percentile(timings, 0.50),
        "p90_ms": percentile(timings, 0.90),
        "p95_ms": percentile(timings, 0.95),
        "p99_ms": percentile(timings, 0.99),
        "percentile_method": "linear interpolation at p*(n-1)",
        "min_ms": min(timings),
        "max_ms": max(timings),
        "graph_replayed_frames": sum(sample["graph_replayed"] for sample in samples),
    }
    if row["steady_timing"]["graph_replayed_frames"] != measure_frames:
        raise RuntimeError("At least one timed frame did not use graph replay")

    # Hash two identical reset-frame outputs outside the measured interval.
    # The model tensor is the final composited image passed to bridge export.
    original_process = host._modes.process
    captured = []

    def capture_output(*args, **kwargs):
        result = original_process(*args, **kwargs)
        captured.append(result)
        return result

    host._modes.process = capture_output
    hashes = []
    try:
        first_probe = warmup_frames + measure_frames + 1
        for frame_id in (first_probe, first_probe + 1):
            sample = invoke_frame(host, device, queue, color, motion, frame_id,
                                  1, width, height)
            row["frames"] += 1
            if not sample["graph_replayed"]:
                raise RuntimeError("Determinism probe did not use graph replay")
            if len(captured) != len(hashes) + 1:
                raise RuntimeError("Did not capture exactly one model output per probe")
            hashes.append(tensor_sha256(torch, captured[-1]))
    finally:
        host._modes.process = original_process

    row["output_check"] = {
        "method": "SHA-256 of two repeated reset-frame model outputs",
        "hashes": hashes,
        "deterministic": len(hashes) == 2 and hashes[0]["sha256"] == hashes[1]["sha256"],
        "included_in_timing_samples": False,
    }
    if not row["output_check"]["deterministic"]:
        raise RuntimeError("Repeated reset-frame output hashes differ")

    hits_after = disk_only.hits
    row["cache"] = {
        "mode": "DiskOnly",
        "cache_hits_after_warmup": hits_before,
        "cache_hits_after_measurement_and_output_check": hits_after,
        "missing_artifact": False,
        "compile_allowed": False,
    }
    row["steady_timing"]["graph_replay_verified_every_frame"] = True


def main(root: Path, report: Path, *, width: int = 960, height: int = 540,
         game_default: bool = False, steady_baseline: bool = False,
         input_height: int | None = None, warmup_frames: int = 16,
         measure_frames: int = 240, preflight_json: Path | None = None,
         game_fullsize_override: Path | None = None,
         triton_cache_dir: Path | None = None,
         stage_profile: bool = False, geometry_profile: bool = False,
         graph_io_audit: bool = False, byte_audit_only: bool = False,
         eager_group_probe: bool = False, graph_stage_markers: bool = False,
         post_substage_markers: bool = False,
         stage_marker_cache: Path | None = None,
         expected_reset_sha256: str | None = None) -> int:
    root = root.resolve()
    report = report.resolve()
    selected_height = input_height if input_height is not None else (
        540 if steady_baseline else (360 if game_default else 480))
    row = {
        "passed": False,
        "mode": "steady-baseline" if steady_baseline else "smoke",
        "runtime": str(root),
        "report": str(report),
        "command_argv": sys.argv,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "frames": 0,
        "source": [width, height],
        "input_height": selected_height,
        "game_default": game_default,
        "game_tested": False,
    }
    if game_fullsize_override is not None:
        override = game_fullsize_override.resolve()
        row["game_fullsize_override"] = str(override)
        row["game_fullsize_override_sha256"] = hashlib.sha256(
            override.read_bytes()).hexdigest()
    if triton_cache_dir is not None:
        row["triton_cache_dir_override"] = str(triton_cache_dir.resolve())
    host = None
    bridge = None
    try:
        if stage_profile and not steady_baseline:
            raise ValueError("--stage-profile requires --steady-baseline")
        if geometry_profile and not stage_profile:
            raise ValueError("--geometry-profile requires --stage-profile")
        if graph_io_audit and not steady_baseline:
            raise ValueError("--graph-io-audit requires --steady-baseline")
        if byte_audit_only and (not (graph_io_audit or graph_stage_markers) or stage_profile):
            raise ValueError("--byte-audit-only requires graph I/O or stage-marker "
                             "audit without stage timing")
        if eager_group_probe and not steady_baseline:
            raise ValueError("--eager-group-probe requires --steady-baseline")
        if graph_stage_markers and (not steady_baseline or stage_marker_cache is None or
                                    not expected_reset_sha256):
            raise ValueError("Graph stage markers require steady baseline, isolated marker "
                             "cache and expected reset SHA-256")
        if graph_stage_markers and (stage_profile or geometry_profile or graph_io_audit or
                                    eager_group_probe):
            raise ValueError("Graph stage markers cannot be combined with other probes")
        if post_substage_markers and (not graph_stage_markers or selected_height != 540):
            raise ValueError("Post substage markers require 540p fullsize graph stage markers")
        if steady_baseline:
            if (width, height) != (960, 540):
                raise ValueError("Steady baseline requires a 960x540 RE8 source")
            if selected_height not in (360, 480, 540):
                raise ValueError("--input-height must be 360, 480, or 540")
            if warmup_frames < 16:
                raise ValueError("Steady baseline requires at least 16 warm-up frames")
            if measure_frames < 1:
                raise ValueError("Steady baseline requires at least one measured frame")
            if preflight_json is None:
                raise ValueError("Steady baseline requires --preflight-json")
            row["preflight"] = (
                validate_byte_audit_preflight(preflight_json.resolve())
                if byte_audit_only else validate_preflight(preflight_json.resolve()))

        assert (root / "python/python313.dll").is_file()
        assert (root / "native/nr_texture_bridge_re8_v1.dll").is_file()
        assert (root / "exact/model-assets/style-sm89-v1/manifest.json").is_file()
        sys.path[:0] = [str(root / "game"), str(root), str(root / "modules"),
                        str(root / "fast/backend")]
        from runtime_environment import isolate
        hold = isolate()
        import fast_cached_runtime_v1 as fast_runtime
        if triton_cache_dir is not None:
            cache_path = triton_cache_dir.resolve()
            if not cache_path.is_dir():
                raise FileNotFoundError(f"Isolated Triton cache does not exist: {cache_path}")
            fast_runtime.CACHE = cache_path
        fast_runtime.bootstrap()
        if game_fullsize_override is not None:
            install_game_fullsize_override(game_fullsize_override, root)
        import torch
        if graph_stage_markers:
            from graph_stage_markers import prewarm_markers
            marker_cache = stage_marker_cache.resolve()
            if marker_cache.drive.casefold() != "d:":
                raise ValueError("Stage marker cache must be on D:")
            prewarm_markers(marker_cache)
            if post_substage_markers:
                from post_stage_markers import prewarm as prewarm_post_markers
                prewarm_post_markers(marker_cache)
        from nr_texture_bridge_v1 import TextureBridge
        import nr_game_pre_xess_host as host

        # Keep test logs and any spike diagnostics with task output on D:.
        host.LOG = report.parent / "runtime-logs"
        if steady_baseline or input_height is not None or game_default:
            from nr_game_controls import Settings

            class TestPanel:
                def snapshot(self):
                    return Settings(input_size=selected_height, display_strength=1.0,
                                    history_mode="fused", graph_replay=True)

                def mark_active(self, *args, **kwargs):
                    pass

            host._panel = TestPanel()
        if selected_height is not None:
            row["input_height"] = selected_height

        bridge = TextureBridge(root / "native/nr_texture_bridge_re8_v1.dll", width, height)
        fixture = bridge.dll.nr_texture_fixture_re8
        fixture.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                            ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p,
                            ctypes.c_uint32]
        fixture.restype = ctypes.c_int
        color, motion = ctypes.c_void_p(), ctypes.c_void_p()
        error = ctypes.create_string_buffer(8192)
        if fixture(bridge.handle, ctypes.byref(color), ctypes.byref(motion),
                   error, len(error)):
            raise RuntimeError(error.value.decode("utf-8", errors="replace"))
        device, queue = bridge.test_context()

        if steady_baseline:
            row["source_code_sha256"] = {
                "game_py_tree": tree_sha256(root / "game"),
                "fast_nr_backend_py_tree": tree_sha256(
                    root / "fast/backend/nr_backend"),
            }
            device_index = torch.xpu.current_device()
            row["environment"] = {
                "python": sys.version,
                "executable": sys.executable,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "xpu_device": torch.xpu.get_device_name(device_index),
                "xpu_properties": str(torch.xpu.get_device_properties(device_index)),
                "b580_driver_version": row["preflight"]["b580"]["driver_version"],
                "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"),
            }
            if graph_stage_markers:
                from graph_stage_markers import installed as stage_markers_installed
                marker_scope = stage_markers_installed(marker_cache)
            else:
                marker_scope = nullcontext()
            if post_substage_markers:
                from post_stage_markers import installed as post_markers_installed
                post_scope = post_markers_installed(marker_cache)
            else:
                post_scope = nullcontext()
            with marker_scope, post_scope:
                run_steady_baseline(
                    host, torch, device, queue, color.value, motion.value,
                    width=width, height=height, input_height=selected_height,
                    warmup_frames=warmup_frames, measure_frames=measure_frames,
                    row=row, stage_profile=stage_profile,
                    geometry_profile=geometry_profile,
                    graph_io_audit=graph_io_audit)
            if graph_stage_markers:
                actual = row["output_check"]["hashes"][0]["sha256"]
                if actual.casefold() != expected_reset_sha256.casefold():
                    raise RuntimeError("Stage markers changed frozen reset output: "
                                       f"{actual} != {expected_reset_sha256}")
                row["stage_marker_probe"] = {
                    "frozen_reset_output_byte_identical": True,
                    "marker_cache": str(marker_cache),
                    "post_substage_markers": post_substage_markers,
                    "timing_interpretation": (
                        "Graph includes 14 extra marker kernels absent from production. "
                        "Use VTune per-instance timestamps only after validating task "
                        "order and comparing whole-frame overhead with unmarked graph."),
                }
            row["steady_timing"]["valid_for_performance"] = not byte_audit_only
            if byte_audit_only:
                row["steady_timing"]["invalid_reason"] = row["preflight"]["timing_reason"]
            if eager_group_probe:
                row["eager_group_probe"] = run_eager_group_probe(
                    host, torch, device, queue, color.value, motion.value,
                    width=width, height=height, input_height=selected_height,
                    first_frame_id=warmup_frames + measure_frames + 3)
                row["frames"] += 4
        else:
            for frame_id in (1, 2):
                sample = invoke_frame(host, device, queue, color.value, motion.value,
                                      frame_id, int(frame_id == 1), width, height)
                row["frames"] += 1
            row["passed"] = True
        row["finished_utc"] = datetime.now(timezone.utc).isoformat()
        row["passed"] = True
    except BaseException as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()[-8000:]
    finally:
        cleanup_errors = []
        if host is not None:
            for name, action in (
                    ("modes", lambda: host._modes.close() if host._modes is not None else None),
                    ("bridge", lambda: host._bridge.close() if host._bridge is not None else None),
                    ("disk_only", lambda: host._disk_only.__exit__(None, None, None)
                     if host._disk_only is not None else None)):
                try:
                    action()
                except BaseException as exc:
                    cleanup_errors.append(f"{name}: {type(exc).__name__}: {exc}")
        if bridge is not None:
            try:
                bridge.close()
            except BaseException as exc:
                cleanup_errors.append(f"fixture bridge: {type(exc).__name__}: {exc}")
        if cleanup_errors:
            row["cleanup_errors"] = cleanup_errors
            row["passed"] = False
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in row.items() if k != "traceback"},
                     ensure_ascii=False), flush=True)
    return 0 if row["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=540)
    parser.add_argument("--game-default", action="store_true")
    parser.add_argument("--steady-baseline", action="store_true",
                        help="run the production-host steady-state benchmark")
    parser.add_argument("--stage-profile", action="store_true",
                        help="time bridge prepare, NR, export and registration")
    parser.add_argument("--geometry-profile", action="store_true",
                        help="also split NR geometry and model segment")
    parser.add_argument("--graph-io-audit", action="store_true",
                        help="count explicit graph-front copy bytes and replay deltas")
    parser.add_argument("--byte-audit-only", action="store_true",
                        help="permit recorded desktop load; timing becomes invalid")
    parser.add_argument("--eager-group-probe", action="store_true",
                        help="two eager progress-marked frames and paired graph outputs")
    parser.add_argument("--graph-stage-markers", action="store_true",
                        help="capture named stage markers inside the fullsize NR graph")
    parser.add_argument("--post-substage-markers", action="store_true",
                        help="also split the 540p post stage into seven internal groups")
    parser.add_argument("--stage-marker-cache", type=Path,
                        help="D: cache for test-only marker kernels")
    parser.add_argument("--expected-reset-sha256",
                        help="frozen unmarked reset model-output digest")
    parser.add_argument("--input-height", type=int, choices=(360, 480, 540),
                        default=None, help="RE8 active model input height")
    parser.add_argument("--warmup-frames", type=int, default=16)
    parser.add_argument("--measure-frames", type=int, default=240)
    parser.add_argument("--preflight-json", type=Path,
                        help="idle/game-closed evidence required by steady mode")
    parser.add_argument("--game-fullsize-override", type=Path,
                        help="test-only nr_game_fullsize.py loaded in this process")
    parser.add_argument("--triton-cache-dir", type=Path,
                        help="use an isolated prepopulated Triton cache directory")
    args = parser.parse_args()
    raise SystemExit(main(args.runtime, args.report, width=args.width,
                          height=args.height, game_default=args.game_default,
                          steady_baseline=args.steady_baseline,
                          input_height=args.input_height,
                          warmup_frames=args.warmup_frames,
                          measure_frames=args.measure_frames,
                          preflight_json=args.preflight_json,
                          game_fullsize_override=args.game_fullsize_override,
                          triton_cache_dir=args.triton_cache_dir,
                          stage_profile=args.stage_profile,
                          geometry_profile=args.geometry_profile,
                          graph_io_audit=args.graph_io_audit,
                          byte_audit_only=args.byte_audit_only,
                          eager_group_probe=args.eager_group_probe,
                          graph_stage_markers=args.graph_stage_markers,
                          post_substage_markers=args.post_substage_markers,
                          stage_marker_cache=args.stage_marker_cache,
                          expected_reset_sha256=args.expected_reset_sha256))
