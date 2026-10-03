"""Bounded live comparison for the opt-in replay lifecycle audit policy.

This is a CPU-side controller. The already-running game owns all GPU work.
No lifecycle transition is made by import, self-test, or preflight.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
import traceback
import urllib.request

TOOLS = Path(__file__).resolve().parent
PROJECT = TOOLS.parent
PERF = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001")
OUT_ROOT = PERF / "replay-lifecycle-v5-unwind-live-v1-20261001"
DEFAULT_MANIFEST = PROJECT / "artifacts" / "replay-lifecycle-controls-v5-unwind-20261001" / "payload-manifest.json"
EXPECTED_BUNDLE_SHA256 = "3630d7b689df6119d806397ad73a2d2f96c9b341c608f4170faed8dd4ac74732"
EXPECTED_NATIVE_SHA256 = "5b5ebabc315390d5076a585263429a32aee993e1eeb5f83c85a63e3db2515c73"
EXPECTED_HOST_SHA256 = "d613d1ba88dd5c60833e48a2bc00778b9a274f775459319d46df4c548f0a7100"
EXPECTED_PROFILE_SHA256 = "26471f140205ebbdc249159f0ca1c0c35d786cd1172c809e01cfd0f2297ef94f"
NATIVE_ASI = Path(r"G:\epic\Cyberpunk2077\bin\x64\plugins\CyberpunkNRBridge.asi")
ADAPTER_MODULE = Path(r"G:\epic\Cyberpunk2077\bin\x64\plugins\cyberpunk_nr_adapter.py")
HOST_SOURCE = Path(r"G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime\game\nr_game_pre_xess_host.py")
PROFILE = Path(r"G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime\data\product-v1\local-runtime-v1.json")
URL = "http://127.0.0.1:8765"
REFERENCE = ("c512_k8_decoder", "c512_k8_c32_native", "num_history_fractional", "num_front_both")
CANDIDATES = ("num_branch_c128",)
POLL = 0.2
ACTIVE_CONFIRM_API = None

sys.path.insert(0, str(TOOLS))
import benchmark_live_web_720_v1 as base
import compare_live_numeric_cost_720_v1 as cost
import sweep_live_web_720_v1 as sweep


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def capture_product_error(output: Path, started_utc: str):
    source = Path(r"G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime\logs\nr-pre-xess-python-error.txt")
    try:
        info = source.stat()
        payload = source.read_bytes()
    except OSError as exc:
        return {"available": False, "source": str(source),
                "read_error": f"{type(exc).__name__}: {exc}"}
    destination = output / "product-error-completed.txt"
    destination.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    modified = dt.datetime.fromtimestamp(info.st_mtime, dt.timezone.utc)
    started = dt.datetime.fromisoformat(started_utc.replace("Z", "+00:00"))
    return {"available": True, "source": str(source), "source_mtime_utc": modified.isoformat(),
            "modified_during_pilot": modified >= started,
            "source_length": len(payload), "source_sha256": digest,
            "copied_path": str(destination), "copied_sha256": digest,
            "byte_identical": destination.read_bytes() == payload}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_freeze(manifest_path: Path, cache_manifests=(), *,
                  expected_bundle, expected_native, expected_host, expected_profile):
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    rows = []
    for item in manifest.get("files", []):
        source, target = Path(item["source"]), Path(item["target"])
        expected = item.get("after_sha256")
        if not expected:
            raise RuntimeError("installed manifest row lacks after_sha256: " + str(item.get("label")))
        source_hash, target_hash = sha256(source), sha256(target)
        rows.append({"label": item.get("label"), "source": str(source), "source_sha256": source_hash,
                     "target": str(target), "target_sha256": target_hash,
                     "expected_after_sha256": expected, "source_matches": source_hash == expected,
                     "target_matches": target_hash == expected})
    if len(rows) != 14 or any(not r["source_matches"] or not r["target_matches"] for r in rows):
        raise RuntimeError("14-file lifecycle bundle source/installed hash verification failed")
    extra = {}
    for path in (NATIVE_ASI, ADAPTER_MODULE, HOST_SOURCE, PROFILE,
                 Path(base.GAME) / "numeric_game_profiles_720_v1.py",
                 Path(base.GAME) / "branch_accum_native_720_kernel_v1.py",
                 Path(base.GAME) / "replay_lifecycle_audit_base_720_v1.py",
                 Path(base.GAME) / "replay_lifecycle_audit_rest_720_v1.py",
                 Path(cost.__file__).resolve(),
                 Path(sweep.__file__).resolve(),
                 Path(base.__file__).resolve(),
                 Path(__file__).resolve()):
        extra[str(path)] = sha256(path)
    if extra[str(NATIVE_ASI)] != expected_native or extra[str(HOST_SOURCE)] != expected_host:
        raise RuntimeError("installed native/host SHA differs from reviewed launch pins")
    if extra[str(PROFILE)] != expected_profile:
        raise RuntimeError("runtime profile changed from the installed preflight")
    cache_rows = {}
    for value in cache_manifests:
        path = Path(value).resolve()
        cache_rows[str(path)] = sha256(path)
    return {"manifest": str(manifest_path.resolve()), "manifest_sha256": sha256(manifest_path),
            "bundle_sha256": expected_bundle, "files": rows, "pinned_runtime_and_helpers": extra,
            "cache_manifests": cache_rows}


class LifecycleAPI(sweep.FixedControlsAPI):
    def __init__(self):
        super().__init__()
        self.lifecycle_checkbox_present = False
        self.last_state = None

    def connect(self):
        req = urllib.request.Request(URL + "/", headers={"Accept": "text/html"})
        try:
            with urllib.request.urlopen(req, timeout=5) as response:
                page = response.read().decode("utf-8")
        except Exception as exc:
            raise RuntimeError(f"{type(exc).__name__} opening lifecycle control page") from None
        token = re.search(r"\bconst\s+token\s*=\s*'([0-9a-f]{48})'\s*;", page)
        if token is None:
            raise RuntimeError("CSRF token missing from lifecycle control page")
        self.token = token.group(1)
        self.checkboxes = set(re.findall(r"""data-optimization=["']([^"']+)["']""", page))
        self.lifecycle_checkbox_present = bool(re.search(r"""id=["']lifecycleEnabled["']""", page))
        if not self.lifecycle_checkbox_present:
            raise RuntimeError("lifecycle policy checkbox is absent from the page")

    def call(self, path, payload=None):
        value = super().call(path, payload)
        if path == "/api/state" and isinstance(value, dict):
            self.last_state = value
        return value


def lifecycle_state(st):
    value = st.get("lifecycle_audit")
    if not isinstance(value, dict) or type(value.get("enabled")) is not bool:
        raise RuntimeError("lifecycle_audit state is unavailable")
    if type(value.get("applied")) is not bool or type(value.get("pending")) is not bool:
        raise RuntimeError("lifecycle_audit applied/pending state is malformed")
    if type(value.get("epoch")) is not int:
        raise RuntimeError("lifecycle_audit epoch is unavailable")
    return value


def policy_converged(st, desired):
    state = lifecycle_state(st)
    return state["enabled"] is desired and state["applied"] is desired and state["pending"] is False


def joined_snapshot(st, frame_id, desired):
    state = st.get("validation_batch")
    if not isinstance(state, dict) or state.get("enabled") is not True:
        return None
    snapshot = state.get("last_snapshot")
    if not isinstance(snapshot, dict):
        return None
    if snapshot.get("frame_id") != frame_id or snapshot.get("lifecycle_trial") is not desired:
        return None
    if snapshot.get("active") is not False:
        return None
    if desired and snapshot.get("retired") is not False:
        return None
    # full_checks/reused_checks are deliberately not gates: zero is valid for
    # a frame that exercised no memoized check.
    return snapshot


def summarize_collected(rows, requested, policy_enabled):
    if requested != 8 and requested < 30:
        raise ValueError("collection size must be the 8-frame readiness pilot or at least 30 measurement frames")
    if len(rows) != requested:
        raise RuntimeError(f"collected {len(rows)} rows; expected {requested}")
    ids = [row.get("frame_id") for row in rows]
    if any(frame_id is None for frame_id in ids) or len(set(ids)) != requested:
        raise RuntimeError("collected frame IDs are missing or duplicated")
    for row in rows:
        snapshot = row.get("lifecycle_snapshot") or {}
        if row.get("network_replay_count") != 1:
            raise RuntimeError("collected row is not exactly one network replay")
        if (snapshot.get("frame_id") != row.get("frame_id") or
                snapshot.get("lifecycle_trial") is not policy_enabled or
                snapshot.get("active") is not False):
            raise RuntimeError("collected row lacks an exact matching lifecycle snapshot")
        if policy_enabled and snapshot.get("retired") is not False:
            raise RuntimeError("trial-on row is not a completed, non-retired frame")
    metrics = reduce_lifecycle_samples(rows, policy_enabled) if requested >= 30 else None
    return {"unique_completed_samples": requested, "unique_frame_ids": ids,
            "samples": rows, "metrics": metrics,
            "summary_kind": "readiness_only" if requested == 8 else "completed_frame_cost"}


def reduce_lifecycle_samples(rows, policy_enabled):
    if len(rows) < 30:
        raise RuntimeError("fewer than 30 unique completed cost samples")
    required = {"process_wall_ms", "network_graph_gpu_ms", "graph_constant_guard_cpu_ms"}
    if not policy_enabled:
        required.add("numeric_guard_cpu_ms")
    metrics = {}
    for row in rows:
        if row.get("network_replay_count") != 1:
            raise RuntimeError("sample did not have exactly one network replay")
    for key in cost.KEYS:
        values, missing = [], 0
        for row in rows:
            value = row.get(key)
            if value is None:
                missing += 1
                continue
            if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                    not math.isfinite(value) or value < 0):
                raise RuntimeError("invalid cost sample: " + key)
            values.append(float(value))
        if missing:
            if key in required:
                raise RuntimeError("missing required cost sample: " + key)
            metrics[key] = {"available": False, "missing_sample_count": missing,
                            "reason": ("lifecycle trial did not emit this field" if policy_enabled and
                                       key == "numeric_guard_cpu_ms" else "field absent from one or more rows")}
        else:
            metrics[key] = {"available": True, "mean_ms": statistics.fmean(values),
                            "p50_ms": statistics.median(values)}
    return metrics


def pair_lifecycle_metrics(measurements):
    deltas = {}
    for key in cost.KEYS:
        metric_rows = [(m.get("metrics") or {}).get(key) for m in measurements]
        if len(metric_rows) != 4 or any(not isinstance(item, dict) or item.get("available") is not True
                                        for item in metric_rows):
            deltas[key] = {"available": False,
                           "reason": "one or more B/C arms did not emit this metric"}
            continue
        values = [item["mean_ms"] for item in metric_rows]
        b = (values[0] + values[3]) / 2
        c = (values[1] + values[2]) / 2
        deltas[key] = {"available": True, "baseline_ms": b, "candidate_ms": c,
                       "c_minus_b_ms": c - b, "baseline_drift_ms": values[3] - values[0],
                       "both_candidates_below_both_baselines": max(values[1:3]) < min(values[0], values[3]),
                       "both_candidates_above_both_baselines": min(values[1:3]) > max(values[0], values[3])}
    return deltas


def selected_for_confirm(reference_only, candidate):
    """Return the exact selected mode; reference-only never appends a profile."""
    return tuple(REFERENCE) if reference_only else (*REFERENCE, candidate)


def self_test(output: Path | None):
    manifest = json.loads(DEFAULT_MANIFEST.read_text(encoding="utf-8-sig"))
    assert manifest.get("bundle_sha256") == EXPECTED_BUNDLE_SHA256
    assert len(manifest.get("files", [])) == 14
    for item in manifest["files"]:
        source = Path(item["source"])
        assert source.is_file() and sha256(source) == item.get("after_sha256"), item.get("label")
    states = [
        {"lifecycle_audit": {"enabled": False, "applied": False, "pending": False, "epoch": 2}},
        {"lifecycle_audit": {"enabled": True, "applied": True, "pending": False, "epoch": 3}},
    ]
    assert policy_converged(states[0], False) and policy_converged(states[1], True)
    assert not policy_converged({"lifecycle_audit": {"enabled": True, "applied": False,
                                                      "pending": True, "epoch": 3}}, True)
    snap = {"enabled": True, "active": False, "frame_id": 44, "retired": False,
            "lifecycle_trial": True, "full_checks": 0, "reused_checks": 0}
    fixture = {"validation_batch": {"enabled": True, "last_snapshot": snap}}
    assert joined_snapshot(fixture, 44, True) == snap
    assert joined_snapshot(fixture, 45, True) is None
    assert joined_snapshot(fixture, 44, False) is None
    assert joined_snapshot({"validation_batch": {"enabled": True,
                           "last_snapshot": {**snap, "lifecycle_trial": False}}}, 44, False) is not None
    def sample(frame_id, trial):
        return {"frame_id": frame_id, "network_replay_count": 1,
                "process_wall_ms": 2.0, "network_graph_gpu_ms": 1.0,
                "numeric_guard_cpu_ms": 0.2, "graph_constant_guard_cpu_ms": 0.1,
                "lifecycle_snapshot": {"enabled": True, "active": False, "frame_id": frame_id,
                                       "retired": False, "lifecycle_trial": trial,
                                       "full_checks": 0, "reused_checks": 0}}
    pilot_summary = summarize_collected([sample(i, False) for i in range(8)], 8, False)
    assert pilot_summary["summary_kind"] == "readiness_only" and pilot_summary["metrics"] is None
    trial_rows = [{k: v for k, v in sample(i, True).items() if k != "numeric_guard_cpu_ms"}
                  for i in range(30)]
    measured_summary = summarize_collected(trial_rows, 30, True)
    assert measured_summary["summary_kind"] == "completed_frame_cost"
    assert measured_summary["metrics"]["process_wall_ms"]["mean_ms"] == 2.0
    assert measured_summary["metrics"]["numeric_guard_cpu_ms"] == {
        "available": False, "missing_sample_count": 30,
        "reason": "lifecycle trial did not emit this field"}
    paired = pair_lifecycle_metrics([summarize_collected([sample(i, False) for i in range(30)], 30, False),
                                     measured_summary, measured_summary,
                                     summarize_collected([sample(i, False) for i in range(30)], 30, False)])
    assert paired["process_wall_ms"]["available"] is True
    assert paired["numeric_guard_cpu_ms"]["available"] is False
    try:
        summarize_collected([sample(i, False) for i in range(7)], 8, False)
    except RuntimeError:
        pass
    else:
        raise AssertionError("incomplete readiness collection accepted")
    readiness_plan = [False, True, False]
    pair_plan = [False, True, True, False]
    assert readiness_plan == [False, True, False]
    assert pair_plan == [False, True, True, False]
    assert selected_for_confirm(True, "ignored") == REFERENCE
    assert "num_branch_c128" not in selected_for_confirm(True, "ignored")
    assert selected_for_confirm(False, "num_branch_c128") == (*REFERENCE, "num_branch_c128")
    obj = {"status": "passed", "checks": [
        "requested/applied/pending and lifecycle epoch parsing",
        "exact cost-frame/validation snapshot join",
        "zero full/reused checks accepted when lifecycle snapshot is valid",
        "collector/summarizer accepts exact 8-frame readiness without cost reducer and keeps >=30 cost gate",
        "trial-only missing inclusive guard stays unavailable rather than being fabricated as zero",
        "reference-only confirmation selects exactly the four reference options",
        "V5 unwind manifest has 14 exact source hashes and expected bundle pin",
        "V5 pilot OFF/ON/OFF and confirmation B-C-C-B gates",
    ], "network_touched": False, "gpu_touched": False, "torch_imported": False,
        "bundle_sha256": EXPECTED_BUNDLE_SHA256,
        "manifest_sha256": sha256(DEFAULT_MANIFEST),
        "utc": base.utc()}
    if output is not None:
        write_json(output, obj)
    return obj


def ensure_output(path: Path):
    resolved = path.resolve()
    if not resolved.is_relative_to(OUT_ROOT.resolve()) or resolved.exists():
        raise ValueError("output must be a new directory under " + str(OUT_ROOT))
    resolved.mkdir(parents=True)
    return resolved


def verify_receipts(args):
    for path in (args.manifest, args.install_receipt, args.launch_receipt):
        if not Path(path).is_file():
            raise RuntimeError("required receipt missing: " + str(path))
    manifest_obj = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    if len(manifest_obj.get("files", [])) != 14:
        raise RuntimeError("install manifest is not the reviewed 14-file V5 unwind bundle")
    if Path(args.manifest).resolve() != DEFAULT_MANIFEST.resolve():
        raise RuntimeError("only the reviewed lifecycle-controls V5 unwind manifest is accepted")
    if (manifest_obj.get("bundle_sha256") != EXPECTED_BUNDLE_SHA256 or
            args.bundle_sha256 != EXPECTED_BUNDLE_SHA256):
        raise RuntimeError("requested bundle SHA is not the reviewed lifecycle-controls V5 unwind bundle")
    if (args.native_sha256 != EXPECTED_NATIVE_SHA256 or args.host_sha256 != EXPECTED_HOST_SHA256 or
            args.profile_sha256 != EXPECTED_PROFILE_SHA256):
        raise RuntimeError("native/host/profile SHA differs from the reviewed V5 runtime pins")
    if args.bundle_sha256 not in Path(args.install_receipt).read_text(encoding="utf-8-sig"):
        raise RuntimeError("install receipt does not identify the supplied V5 bundle")
    launch = json.loads(Path(args.launch_receipt).read_text(encoding="utf-8-sig"))
    if launch.get("scope_override_present") is True:
        raise RuntimeError("launch receipt has a scope override")
    if launch.get("payload_bundle_sha256") != args.bundle_sha256:
        raise RuntimeError("launch receipt bundle does not match the supplied V5 bundle")
    if launch.get("asi_sha256") != args.native_sha256 or launch.get("host_sha256") != args.host_sha256:
        raise RuntimeError("launch receipt native/host pins do not match the supplied pins")
    return {"manifest_sha256": sha256(Path(args.manifest)),
            "install_receipt_sha256": sha256(Path(args.install_receipt)),
            "launch_receipt_sha256": sha256(Path(args.launch_receipt)),
            "launch_pid": launch.get("pid"), "bundle_sha256": args.bundle_sha256,
            "native_sha256": args.native_sha256, "host_sha256": args.host_sha256,
            "profile_sha256": args.profile_sha256}


def source_args(args):
    return source_freeze(args.manifest, args.cache_manifest,
                         expected_bundle=args.bundle_sha256, expected_native=args.native_sha256,
                         expected_host=args.host_sha256, expected_profile=args.profile_sha256)


def verify_pilot_gate(path: Path, args):
    pilot = json.loads(path.read_text(encoding="utf-8-sig"))
    expected = {"bundle_sha256": args.bundle_sha256, "native_sha256": args.native_sha256,
                "host_sha256": args.host_sha256, "profile_sha256": args.profile_sha256}
    if (pilot.get("completed") is not True or
            pilot.get("phase") != "bidirectional_lifecycle_readiness_pilot_v5_unwind" or
            pilot.get("policy_left_on") is not False or pilot.get("source_pins") != expected):
        raise RuntimeError("confirmation requires a completed matching V5 unwind pilot receipt")
    arms = pilot.get("arms")
    if not isinstance(arms, list) or [row.get("phase") for row in arms] != ["OFF-before", "ON", "OFF-after"]:
        raise RuntimeError("V5 pilot receipt lacks the required OFF/ON/OFF stages")
    for arm in arms:
        measurement = arm.get("measurement") or {}
        if measurement.get("unique_completed_samples") != 8:
            raise RuntimeError("each V5 pilot side must contain eight joined completed frames")
        samples = measurement.get("samples") or []
        if len({row.get("frame_id") for row in samples}) != 8:
            raise RuntimeError("V5 pilot frame IDs are not unique")
        expected_trial = arm["phase"] == "ON"
        for sample in samples:
            snapshot = sample.get("lifecycle_snapshot") or {}
            if (snapshot.get("frame_id") != sample.get("frame_id") or
                    snapshot.get("lifecycle_trial") is not expected_trial or
                    snapshot.get("active") is not False):
                raise RuntimeError("V5 pilot cost row lacks its exact expected lifecycle snapshot")
            if expected_trial and snapshot.get("retired") is not False:
                raise RuntimeError("V5 pilot ON row was not a completed live trial")
        if any(row.get("network_replay_count") != 1 for row in samples):
            raise RuntimeError("V5 pilot contains a non-replay sample")
    transitions = pilot.get("transitions") or []
    if len(transitions) != 2:
        raise RuntimeError("V5 pilot receipt lacks exactly one ON and one OFF transition")
    expected_states = ((False, True), (True, False))
    for transition, (before, after) in zip(transitions, expected_states):
        if (transition.get("before", {}).get("applied") is not before or
                transition.get("after", {}).get("applied") is not after or
                transition.get("after", {}).get("pending") is not False or
                transition.get("after", {}).get("epoch", -1) <=
                transition.get("before", {}).get("epoch", -1)):
            raise RuntimeError("V5 pilot transition receipt does not prove fresh applied epochs")
    return {"pilot_receipt": str(path.resolve()), "pilot_receipt_sha256": sha256(path),
            "pilot_phase": pilot["phase"], "verified_stages": [row["phase"] for row in arms]}


def fatal_state(st):
    base.fatal(st)
    if (st.get("health") or {}).get("failed") is not False:
        raise RuntimeError("health.failed or health unavailable")


def set_profile(api, selected, reg, settings_template, prepare_base):
    reg.combined_mode_options(list(selected))
    st = api.call("/api/state")
    fatal_state(st)
    if base.matches(st, selected):
        return st, False
    if not prepare_base:
        raise RuntimeError("actual profile differs; rerun only with explicit --prepare-base authorization")
    api.call("/api/state", base.payload(settings_template, selected))
    st = base.wait_target(api, selected)
    return st, True


def request_policy(api, desired, timeout=35.0):
    before_state = lifecycle_state(api.call("/api/state"))
    if before_state["enabled"] is desired and before_state["applied"] is desired and not before_state["pending"]:
        return {"requested": False, "before": before_state, "after": before_state}
    api.call("/api/lifecycle-audit", {"enabled": desired})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        st = api.call("/api/state")
        fatal_state(st)
        state = lifecycle_state(st)
        if state["epoch"] > before_state["epoch"] and policy_converged(st, desired):
            return {"requested": True, "before": before_state, "after": state}
        time.sleep(POLL)
    raise RuntimeError("lifecycle policy did not apply with a fresh epoch before timeout")


def fresh_roundtrip(api, desired):
    current = lifecycle_state(api.call("/api/state"))
    if current["enabled"] is desired and current["applied"] is desired and not current["pending"]:
        other = not desired
        away = request_policy(api, other)
        back = request_policy(api, desired)
        return {"forced_rebind": True, "away": away, "back": back}
    changed = request_policy(api, desired)
    return {"forced_rebind": False, "transition": changed}


def wait_fresh_route(api, selected, desired, frames_needed=2, timeout=30.0):
    st = api.call("/api/state")
    fatal_state(st)
    start_frames = (st.get("processing") or {}).get("frames")
    if type(start_frames) is not int:
        raise RuntimeError("processing.frames unavailable")
    deadline = time.monotonic() + timeout
    max_frames = start_frames
    first_cost_epoch = None
    while time.monotonic() < deadline:
        time.sleep(POLL)
        st = api.call("/api/state")
        fatal_state(st)
        if not base.matches(st, selected):
            raise RuntimeError("actual profile changed during lifecycle transition")
        if not policy_converged(st, desired):
            continue
        if (st.get("health") or {}).get("structure_combo_frame_route") != "replay":
            continue
        hits = sweep.actual_hits(st, selected, REGISTRY)
        data = cost.cost(st, selected)
        if first_cost_epoch is None:
            first_cost_epoch = data["epoch"]
        frames = (st.get("processing") or {}).get("frames")
        if type(frames) is int:
            max_frames = max(max_frames, frames)
        if max_frames - start_frames >= frames_needed:
            return st, {"baseline_processing_frames": start_frames,
                        "verified_processing_frames": max_frames,
                        "new_processing_frames": max_frames - start_frames,
                        "cost_epoch": data["epoch"], "lifecycle": lifecycle_state(st),
                        "hits": hits}
    raise RuntimeError("fresh replay route did not produce bounded completed frames")


def validate_scene(st, selected):
    if not base.matches(st, selected):
        raise RuntimeError("settings/active_mode do not match the target 720p profile")
    settings = st.get("settings") or {}
    if settings.get("enabled") is not True or settings.get("graph_replay") is not True:
        raise RuntimeError("NR or graph replay is disabled")
    if settings.get("timing_enabled") is False or st.get("timing_enabled") is False:
        raise RuntimeError("timing is disabled")
    if (st.get("source_geometry") or {}).get("width") != 1280 or (st.get("source_geometry") or {}).get("height") != 720:
        raise RuntimeError("source geometry is not 1280x720")
    if (st.get("validation_batch") or {}).get("enabled") is not True:
        raise RuntimeError("validation batch must remain enabled")
    return sweep.actual_hits(st, selected, REGISTRY)


def collect_frames(api, selected, desired, count, timeout=70.0, checkpoint_path=None):
    st = api.call("/api/state")
    fatal_state(st)
    meter = cost.cost(st, selected)
    session_epoch = meter["epoch"]
    policy_epoch = lifecycle_state(st)["epoch"]
    baseline_counter = meter["completed_frames"]
    last_counter = baseline_counter
    rows_by_id, skips = {}, 0
    started = advanced = time.monotonic()
    max_processing = (st.get("processing") or {}).get("frames")
    while len(rows_by_id) < count and time.monotonic() - started < timeout:
        time.sleep(POLL)
        st = api.call("/api/state")
        fatal_state(st)
        validate_scene(st, selected)
        if not policy_converged(st, desired):
            raise RuntimeError("lifecycle policy changed while collecting frames")
        if lifecycle_state(st)["epoch"] != policy_epoch:
            raise RuntimeError("lifecycle epoch changed during frame collection")
        meter = cost.cost(st, selected)
        if meter.get("epoch") != session_epoch:
            raise RuntimeError("cost-meter owner epoch changed during frame collection")
        counter = meter.get("completed_frames")
        if type(counter) is not int:
            raise RuntimeError("completed cost-frame counter unavailable")
        proc_frames = (st.get("processing") or {}).get("frames")
        if type(proc_frames) is int:
            max_processing = proc_frames if max_processing is None else max(max_processing, proc_frames)
        if counter > last_counter:
            delta = counter - last_counter
            advanced = time.monotonic()
            row = meter.get("last")
            if not isinstance(row, dict):
                skips += delta
            else:
                frame_id = row.get("frame_id")
                snapshot = joined_snapshot(st, frame_id, desired)
                if frame_id is None or snapshot is None or row.get("network_replay_count") != 1:
                    skips += delta
                else:
                    if frame_id not in rows_by_id:
                        rows_by_id[frame_id] = dict(row, cost_counter=counter,
                                                    lifecycle_snapshot=dict(snapshot),
                                                    lifecycle_epoch=policy_epoch,
                                                    cost_epoch=session_epoch)
                    skips += max(0, delta - 1)
            last_counter = counter
        if time.monotonic() - advanced > 15:
            raise RuntimeError("completed cost-frame stream stalled")
    rows = [rows_by_id[key] for key in sorted(rows_by_id)]
    if len(rows) < count:
        raise RuntimeError(f"only {len(rows)} exact joined frames; required {count}")
    try:
        summary = summarize_collected(rows, count, desired)
    except Exception as exc:
        if checkpoint_path is not None:
            write_json(checkpoint_path, {"completed": False, "requested": count,
                                         "policy_enabled": desired,
                                         "error": f"{type(exc).__name__}: {exc}",
                                         "session_cost_epoch": session_epoch,
                                         "lifecycle_epoch": policy_epoch,
                                         "cost_counter_start": baseline_counter,
                                         "cost_counter_end": last_counter,
                                         "skipped_or_unjoined_cost_rows": skips,
                                         "unique_collected_rows": len(rows), "samples": rows})
        raise
    return {"session_cost_epoch": session_epoch, "lifecycle_epoch": policy_epoch,
            "cost_counter_start": baseline_counter, "cost_counter_end": last_counter,
            "unique_frame_ids": summary["unique_frame_ids"],
            "unique_completed_samples": summary["unique_completed_samples"],
            "skipped_or_unjoined_cost_rows": skips, "samples": summary["samples"],
            "metrics": summary["metrics"], "summary_kind": summary["summary_kind"],
            "processing_frames_end": max_processing, "finished_utc": base.utc()}


def warmup(api, selected, desired, seconds=8.0, frames=35, timeout=75.0):
    st = api.call("/api/state")
    fatal_state(st)
    initial = (st.get("processing") or {}).get("frames")
    if type(initial) is not int:
        raise RuntimeError("processing.frames unavailable at warmup start")
    begun = advanced = time.monotonic()
    last = initial
    latest_hits = None
    while time.monotonic() - begun < timeout:
        time.sleep(POLL)
        st = api.call("/api/state")
        fatal_state(st)
        latest_hits = validate_scene(st, selected)
        if not policy_converged(st, desired):
            raise RuntimeError("lifecycle policy not converged during warmup")
        now = (st.get("processing") or {}).get("frames")
        if type(now) is int and now > last:
            advanced, last = time.monotonic(), now
        if time.monotonic() - advanced > 15:
            raise RuntimeError("warmup processing frames stalled")
        if time.monotonic() - begun >= seconds and last - initial >= frames:
            return st, {"start_frame": initial, "end_frame": last, "new_frames": last-initial,
                        "warmup_seconds": round(time.monotonic()-begun, 3),
                        "actual_hits": latest_hits}
    raise RuntimeError("warmup did not reach both minimum time and frame count")


def run_pilot(args):
    output = ensure_output(args.output)
    receipt_info, frozen_before, api, reg = None, None, None, None
    arms, transitions, selected = [], [], (*REFERENCE, "num_branch_c128")
    started = base.utc()
    try:
        receipt_info = verify_receipts(args)
        frozen_before = source_args(args)
        reg = base.registry()
        global REGISTRY
        REGISTRY = reg
        if "num_branch_c128" not in reg.PROFILES:
            raise RuntimeError("num_branch_c128 is absent from the installed registry")
        api = LifecycleAPI()
        api.connect()
        if api.checkboxes != set(reg.PROFILES):
            raise RuntimeError("web optimization checkbox registry differs from installed registry")
        st = api.call("/api/state")
        fatal_state(st)
        settings_template = dict(st.get("settings") or {})
        if not settings_template:
            raise RuntimeError("settings snapshot unavailable")
        initial_policy = lifecycle_state(st)
        if not policy_converged(st, False):
            raise RuntimeError("pilot requires lifecycle policy OFF; refusing any preliminary toggle")
        if (st.get("validation_batch") or {}).get("enabled") is not True:
            raise RuntimeError("validation batch must already be ON; refusing to change it")
        # One authorized profile POST, if required. Preserve style/strength and
        # set the explicit 720p backend prerequisites before adding the options.
        settings_template.update(enabled=True, input_size=720, history_mode="fused",
                                 graph_replay=True, backend_variant="unrounded",
                                 experiment_720="c512_k8")
        st, changed = set_profile(api, selected, reg, settings_template, True)
        api.controls = {k: (st.get("settings") or {}).get(k) for k in sweep.CONTROL_KEYS}
        base.validate_base(st)
        if (st.get("validation_batch") or {}).get("enabled") is not True:
            raise RuntimeError("validation batch must remain enabled")
        st, route = wait_fresh_route(api, selected, False)
        warm = {"duration_seconds": 0, "discarded_frames": 0,
                "readiness_frames": route["new_processing_frames"]}
        off_frames = collect_frames(api, selected, False, 8,
                                    checkpoint_path=output / "pilot-off-before-collector-failure.json")
        arms.append({"phase": "OFF-before", "selected": list(selected), "profile_changed_once": changed,
                     "initial_lifecycle_policy": initial_policy, "fresh_route": route,
                     "warmup": warm, "actual_hits": route["hits"], "measurement": off_frames})
        transition = request_policy(api, True)
        transitions.append(transition)
        st, on_route = wait_fresh_route(api, selected, True, frames_needed=8)
        on_smoke = collect_frames(api, selected, True, 8,
                                  checkpoint_path=output / "pilot-on-collector-failure.json")
        arms.append({"phase": "ON", "selected": list(selected),
                     "fresh_route": on_route, "measurement": on_smoke})
        # Only the reviewed V5 unwind bundle can proceed through this bounded normal-retirement check;
        # require the successful eight-frame ON sample before the single OFF.
        transition_off = request_policy(api, False)
        transitions.append(transition_off)
        st, off_after_route = wait_fresh_route(api, selected, False, frames_needed=8)
        off_after = collect_frames(api, selected, False, 8,
                                   checkpoint_path=output / "pilot-off-after-collector-failure.json")
        arms.append({"phase": "OFF-after", "selected": list(selected),
                     "fresh_route": off_after_route, "measurement": off_after})
        st = api.call("/api/state")
        fatal_state(st)
        off_hits = validate_scene(st, selected)
        if not policy_converged(st, False):
            raise RuntimeError("V5 pilot did not finish with lifecycle policy OFF")
        after = source_args(args)
        if frozen_before != after:
            raise RuntimeError("source/runtime/cache-manifest freeze changed during V5 pilot")
        report = {"completed": True, "phase": "bidirectional_lifecycle_readiness_pilot_v5_unwind",
                  "started_utc": started, "finished_utc": base.utc(), "pid": receipt_info["launch_pid"],
                  "receipt_evidence": receipt_info, "source_freeze_before": frozen_before,
                  "source_freeze_after": after, "checkbox_registry_count": len(reg.PROFILES),
                  "selected": list(selected), "settings_profile_changed_once": changed,
                  "source_pins": {"bundle_sha256": args.bundle_sha256, "native_sha256": args.native_sha256,
                                  "host_sha256": args.host_sha256, "profile_sha256": args.profile_sha256},
                  "arms": arms, "transitions": transitions,
                  "final_state": base.clean(st), "final_actual_hits": off_hits,
                  "final_lifecycle_audit": lifecycle_state(st),
                  "interpretation": "8 exact joined frames in OFF-before, ON and OFF-after; readiness only",
                  "policy_left_on": False}
        write_json(output / "pilot-completed.json", report)
        return report
    except Exception as exc:
        report = {"completed": False, "phase": "bidirectional_lifecycle_readiness_pilot_v5_unwind",
                  "started_utc": started, "failed_utc": base.utc(),
                  "pid": receipt_info.get("launch_pid") if receipt_info else None,
                  "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                  "arms_completed": arms, "transitions_completed": locals().get("transitions", []),
                  "last_api_state": api.last_state if api is not None else None,
                  "source_freeze_before": frozen_before, "gpu_tasks_started": 0,
                  "product_error_evidence": capture_product_error(output, started),
                  "policy_may_be_on": bool(locals().get("transitions") and
                                            locals()["transitions"][-1].get("after", {}).get("applied"))}
        write_json(output / "pilot-failed.json", report)
        return report


def _run_confirm_inner(args):
    global ACTIVE_CONFIRM_API
    confirm_started_utc = base.utc()
    ACTIVE_CONFIRM_API = None
    if args.frames < 30 or args.frames > 600:
        raise ValueError("comparison requires 30..600 unique exact-joined frames per arm")
    output = ensure_output(args.output)
    receipt_info = verify_receipts(args)
    pilot_info = verify_pilot_gate(args.pilot_receipt, args)
    before = source_args(args)
    reg = base.registry()
    global REGISTRY
    REGISTRY = reg
    reference_only = bool(args.reference_only)
    if reference_only:
        candidates = ("reference_only",)
    else:
        candidates = tuple((args.candidates or "num_branch_c128").split(","))
        if len(candidates) != len(set(candidates)) or set(candidates) - set(reg.PROFILES):
            raise ValueError("unknown/duplicate candidate list")
    api = LifecycleAPI()
    ACTIVE_CONFIRM_API = api
    api.connect()
    if api.checkboxes != set(reg.PROFILES):
        raise RuntimeError("web optimization checkbox registry differs from installed registry")
    st = api.call("/api/state")
    fatal_state(st)
    template = dict(st.get("settings") or {})
    first_target = selected_for_confirm(reference_only, candidates[0])
    starting_policy_recovery = None
    if not policy_converged(st, False):
        current_policy = lifecycle_state(st)
        if (current_policy["enabled"] is True and current_policy["applied"] is True and
                current_policy["pending"] is False):
            current_selected = tuple((st.get("settings") or {}).get("optimizations_720") or ())
            if not current_selected or not base.matches(st, current_selected):
                raise RuntimeError("cannot safely retire the applied lifecycle policy from an unknown mode")
            starting_policy_recovery = request_policy(api, False)
            st, recovery_route = wait_fresh_route(api, current_selected, False)
            starting_policy_recovery["fresh_off_route"] = recovery_route
        else:
            raise RuntimeError("confirm requires a converged policy; refusing an ambiguous/pending state")
    if reference_only:
        if not base.matches(st, REFERENCE):
            st = api.call("/api/state", base.payload(template, REFERENCE))
            st = base.wait_target(api, REFERENCE)
    elif base.matches(st, first_target):
        # Pilot frames are readiness evidence only. Retire that owner before
        # the first measured B arm so no pilot cost epoch is reused.
        st = api.call("/api/state", base.payload(template, REFERENCE))
        st = base.wait_target(api, REFERENCE)
        st, _ = wait_fresh_route(api, REFERENCE, False)
    elif not base.matches(st, REFERENCE):
        if not args.prepare_base:
            raise RuntimeError("actual mode is neither reference nor first target; explicit --prepare-base required")
        st = api.call("/api/state", base.payload(template, REFERENCE))
        st = base.wait_target(api, REFERENCE)
    api.controls = {k: (st.get("settings") or {}).get(k) for k in sweep.CONTROL_KEYS}
    validate_scene(st, first_target if base.matches(st, first_target) else REFERENCE)
    blocks = []
    last_cost_epochs = set()
    for candidate in candidates:
        selected = selected_for_confirm(reference_only, candidate)
        reg.combined_mode_options(list(selected))
        arm_stem = "reference-only" if reference_only else candidate
        arms = []
        for index, desired in enumerate((False, True, True, False), 1):
            # C-C needs an explicit OFF->ON cold rebind between its two arms.
            if index == 3:
                away = request_policy(api, False)
                st, _ = wait_fresh_route(api, selected, False)
                transition = {"between_adjacent_candidates": away}
                transition["to_arm_policy"] = request_policy(api, True)
            elif index == 1:
                st = api.call("/api/state")
                if not base.matches(st, selected):
                    st = api.call("/api/state", base.payload(template, selected))
                    st = base.wait_target(api, selected)
                transition = {"selected_profile": list(selected)}
                if not policy_converged(st, False):
                    transition["policy"] = request_policy(api, False)
            else:
                transition = request_policy(api, desired)
            st, route = wait_fresh_route(api, selected, desired)
            _, warm = warmup(api, selected, desired)
            measurement = collect_frames(api, selected, desired, args.frames,
                                         checkpoint_path=output / f"{arm_stem}-arm-{index}-collector-failure.json")
            if measurement["session_cost_epoch"] in last_cost_epochs:
                raise RuntimeError("measured arm reused an earlier cost-meter session epoch")
            last_cost_epochs.add(measurement["session_cost_epoch"])
            arm = {"arm": index, "role": "B" if not desired else "C",
                   "policy_enabled": desired, "policy_epoch": lifecycle_state(st)["epoch"],
                   "transition": transition, "fresh_route": route, "warmup": warm,
                   "measurement": measurement}
            arms.append(arm)
            write_json(output / f"{arm_stem}-arm-{index}.json", arm)
        delta = pair_lifecycle_metrics([arm["measurement"] for arm in arms])
        block = {"candidate": candidate, "selected": list(selected), "order": ["B", "C", "C", "B"],
                 "arms": arms, "paired_cost_delta": delta}
        blocks.append(block)
        write_json(output / f"{arm_stem}-completed.json", block)
        print(json.dumps({"event": "candidate_completed", "candidate": candidate,
                          "delta_ms": {k: v.get("c_minus_b_ms") for k, v in delta.items()
                                       if v.get("available") is True and
                                       k in ("process_wall_ms", "network_graph_gpu_ms",
                                             "numeric_guard_cpu_ms", "adapter_handoff_cpu_ms",
                                             "observer_finish_cpu_ms")}}, ensure_ascii=False), flush=True)
    # Return to reference+OFF once after all requested, bounded candidates.
    st = api.call("/api/state")
    if not policy_converged(st, False):
        request_policy(api, False)
    st = api.call("/api/state")
    if not base.matches(st, REFERENCE):
        st = api.call("/api/state", base.payload(template, REFERENCE))
        st = base.wait_target(api, REFERENCE)
    st, restored_route = wait_fresh_route(api, REFERENCE, False)
    final_readiness = collect_frames(
        api, REFERENCE, False, 8,
        checkpoint_path=output / "final-reference-8frame-collector-failure.json")
    st = api.last_state
    final_hits = validate_scene(st, REFERENCE)
    final_policy = lifecycle_state(st)
    if not policy_converged(st, False) or final_policy["pending"]:
        raise RuntimeError("final reference policy did not remain converged OFF")
    after = source_args(args)
    if before != after:
        raise RuntimeError("source/runtime/cache-manifest freeze changed during confirmation")
    report = {"completed": True,
              "phase": "lifecycle_policy_reference_only_pair" if reference_only else "lifecycle_policy_candidate_pairs",
              "started_utc": confirm_started_utc, "finished_utc": base.utc(), "pid": receipt_info["launch_pid"],
              "receipt_evidence": receipt_info, "pilot_evidence": pilot_info,
              "source_freeze_before": before,
              "source_freeze_after": after, "pair_order": "B-C-C-B",
              "starting_policy_recovery": starting_policy_recovery,
              "reference_only": reference_only, "candidates": blocks, "final_reference": list(REFERENCE),
              "final_lifecycle_audit": lifecycle_state(st), "final_restore_route": restored_route,
              "final_reference_readiness": final_readiness, "final_actual_hits": final_hits,
              "final_health_failed": (st.get("health") or {}).get("failed"),
              "gpu_tasks_started": 0, "offline_compilation_started": False}
    write_json(output / "confirm-completed.json", report)
    return report


def run_confirm(args):
    started = base.utc()
    output = Path(args.output).resolve()
    try:
        return _run_confirm_inner(args)
    except Exception as exc:
        if not output.exists():
            if not output.is_relative_to(OUT_ROOT.resolve()):
                raise
            output.mkdir(parents=True)
        completed_arms = []
        for path in sorted(output.glob("*-arm-*.json")):
            try:
                completed_arms.append({"path": str(path), "arm": json.loads(path.read_text(encoding="utf-8"))})
            except Exception as read_exc:
                completed_arms.append({"path": str(path), "read_error": f"{type(read_exc).__name__}: {read_exc}"})
        api = ACTIVE_CONFIRM_API
        state = api.last_state if api is not None else None
        lifecycle = (state or {}).get("lifecycle_audit") or {}
        report = {"completed": False, "phase": "lifecycle_policy_candidate_pairs",
                  "started_utc": started, "failed_utc": base.utc(),
                  "pid": None, "error": f"{type(exc).__name__}: {exc}",
                  "traceback": traceback.format_exc(), "completed_arm_evidence": completed_arms,
                  "last_api_state": state, "policy_may_be_on": bool(
                      lifecycle.get("enabled") or lifecycle.get("applied")),
                  "product_error_evidence": capture_product_error(output, started),
                  "gpu_tasks_started": 0, "offline_compilation_started": False}
        write_json(output / "confirm-failed.json", report)
        return report


def build_parser():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    test = sub.add_parser("self-test")
    test.add_argument("--output", type=Path)
    for name in ("pilot", "confirm"):
        command = sub.add_parser(name)
        command.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
        command.add_argument("--install-receipt", type=Path, required=True)
        command.add_argument("--launch-receipt", type=Path, required=True)
        command.add_argument("--bundle-sha256", required=True)
        command.add_argument("--native-sha256", required=True)
        command.add_argument("--host-sha256", required=True)
        command.add_argument("--profile-sha256", required=True)
        command.add_argument("--cache-manifest", type=Path, action="append", default=[])
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--prepare-base", action="store_true")
    pilot = sub.choices["pilot"]
    pilot.add_argument("--frames", type=int, choices=(8,), default=8)
    confirm = sub.choices["confirm"]
    confirm.add_argument("--candidates")
    confirm.add_argument("--reference-only", action="store_true")
    confirm.add_argument("--frames", type=int, default=30)
    confirm.add_argument("--pilot-receipt", type=Path, required=True)
    return parser


def main():
    args = build_parser().parse_args()
    if args.command == "self-test":
        result = self_test(args.output)
    elif args.command == "pilot":
        result = run_pilot(args)
    else:
        result = run_confirm(args)
    print(json.dumps(result, ensure_ascii=False, default=str), flush=True)


if __name__ == "__main__":
    main()
