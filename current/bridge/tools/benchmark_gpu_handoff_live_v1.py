"""Bounded live bridge comparison. Run mode is reserved for the GPU tester.

CPU-check mode reads source/receipts only and never opens the control port.
Four arms use unique completed-frame records, rather than rolling averages.
"""
import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.error
import urllib.request


PROJECT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8765"
TARGET = tuple(sorted(("c512_k8_decoder", "c512_k8_c32_native",
                       "num_history_fractional", "num_front_both",
                       "c512_k8_probability", "num_branch_c128")))
EXPECTED = {"enabled": True, "input_size": 720, "history_mode": "fused",
            "graph_replay": True, "backend_variant": "unrounded",
            "experiment_720": "c512_k8"}
METRICS = ("record_last_ms", "resources_last_ms", "hdr_initialize_last_ms",
           "xess_record_last_ms", "record_to_submit_gap_last_ms",
           "record_to_retire_span_last_ms")
CONTROL_KEYS = ("display_strength", "style", "model_intensity", "local_tone",
                "local_structure", "auto_mask", "skin_structure")


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical(path):
    for part in (path, *path.parents):
        if part.is_symlink() or part.is_junction():
            raise ValueError("Redirected task path: " + str(part))


def dump(root, name, value):
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def source_pins(receipt):
    install = read(receipt)
    if install.get("status") != "installed_trial_default_off" or len(install["files"]) not in (8, 9):
        raise ValueError("Expected the reviewed bridge trial and its optional import repair")
    pins = {}
    for row in install["files"]:
        target = Path(row["target"])
        physical(target)
        digest = sha(target)
        if digest != row["new_sha256"]:
            raise ValueError("Installed trial source changed: " + str(target))
        pins[str(target)] = digest
    # Freeze the actual mathematical sources without importing Torch or shaders.
    game = Path(install["files"][3]["target"]).parent
    for target in game.glob("*.py"):
        physical(target)
        pins[str(target)] = sha(target)
    overlay = game.parent / "experimental/fp8_unround_overlay"
    for target in (overlay / "bootstrap.py", overlay / "modules/nr_texture_bridge_v1.py"):
        physical(target)
        pins[str(target)] = sha(target)
    pins[str(Path(__file__).resolve())] = sha(Path(__file__).resolve())
    return pins


class API:
    def __init__(self):
        self.token = None

    def connect(self):
        with urllib.request.urlopen(URL + "/", timeout=5) as response:
            page = response.read().decode("utf-8")
        match = re.search(r"\bconst\s+token\s*=\s*'([0-9a-f]{48})'\s*;", page)
        if not match:
            raise ValueError("Control-page CSRF token unavailable")
        self.token = match.group(1)

    def call(self, path, payload=None):
        body = None if payload is None else json.dumps(payload).encode()
        headers = {"Accept": "application/json"}
        if body is not None:
            headers.update({"Content-Type": "application/json", "Origin": URL,
                            "X-NR-Token": self.token})
        request = urllib.request.Request(URL + path, data=body, headers=headers,
                                         method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            if self.token:
                detail = detail.replace(self.token, "<redacted>")
            raise ValueError("HTTP %d %s: %s" % (exc.code, path, detail[:1200])) from None


def frame(state):
    processing = state.get("processing") or {}
    stages = processing.get("stages") or {}
    return (stages.get("recording") or {}).get("last_nr_frame_id")


def payload(settings):
    result = copy.deepcopy(settings)
    result.update(EXPECTED)
    result["optimizations_720"] = list(TARGET)
    return result


def validate(state, controls=None, configured=True):
    if (state.get("health") or {}).get("failed") is not False:
        raise ValueError("NR health failed or unavailable")
    if controls is not None:
        for key, value in controls.items():
            if state["settings"].get(key) != value:
                raise ValueError("User control changed: " + key)
    if not configured:
        return
    if type(frame(state)) is not int or frame(state) < 1:
        raise ValueError("Completed NR frame identity unavailable")
    if state.get("route") != "pre-xess-fullsize":
        raise ValueError("Unexpected bridge route")
    settings, active = state["settings"], state.get("active_mode") or {}
    if any(settings.get(k) != v or active.get(k) != v for k, v in EXPECTED.items() if k != "enabled"):
        raise ValueError("720p settings and actual mode have not converged")
    if settings.get("enabled") is not True:
        raise ValueError("NR is disabled")
    if sorted(settings.get("optimizations_720", [])) != list(TARGET) or sorted(active.get("optimizations_720", [])) != list(TARGET):
        raise ValueError("Actual 720p option combination differs")
    if active.get("style") != settings["style"]:
        raise ValueError("Actual model style differs")
    for setting, actual in (("model_intensity", "intensity"), ("local_tone", "local_tone"),
                            ("local_structure", "local_structure"), ("auto_mask", "auto_mask"),
                            ("skin_structure", "skin_structure")):
        if (active.get("model_controls") or {}).get(actual) != settings[setting]:
            raise ValueError("Actual model control differs: " + setting)
    geometry = state.get("source_geometry") or {}
    if (geometry.get("width"), geometry.get("height")) != (1280, 720):
        raise ValueError("Game source is not the agreed 1280x720")
    health = state["health"]
    for key in ("c512_library_720_active", "native_k8_720_active",
                "decoder_gather_unround_720_active", "c32_hidden_native_720_active"):
        if health.get(key) is not True:
            raise ValueError("Actual backend hit unavailable: " + key)
    if health.get("structure_combo_frame_route") != "replay":
        raise ValueError("Actual model route is not graph replay")
    policy = state.get("lifecycle_audit") or {}
    if policy.get("enabled") is not True or policy.get("applied") is not True or policy.get("pending") is not False:
        raise ValueError("Accepted V6 lifecycle policy has not applied")
    if (state.get("bridge_cache") or {}).get("enabled") is not True or state.get("timing_enabled") is not True:
        raise ValueError("Keep accepted immutable cache and measuring enabled")


def applied(state, pool, handoff):
    p, h = state.get("bridge_resource_pool") or {}, state.get("gpu_handoff") or {}
    host = h.get("host") or {}
    if p.get("enabled") is not pool or h.get("requested") is not handoff:
        return False
    if handoff:
        return h.get("armed") is True and h.get("cap_healthy") is True and host.get("applied") is True and host.get("cap_healthy") is True and h.get("pending") is False
    return h.get("armed") is False and host.get("applied") is False and h.get("pending") is False


def wait_state(api, predicate, controls, configured=True, seconds=40):
    deadline = time.monotonic() + seconds
    last = None
    while time.monotonic() < deadline:
        last = api.call("/api/state")
        validate(last, controls, configured=False)
        try:
            validate(last, controls, configured)
        except ValueError:
            time.sleep(.12)
            continue
        if predicate(last):
            return last
        time.sleep(.12)
    error = ValueError("Mode/bridge transition did not actually apply before deadline")
    error.last_state = last
    raise error


def request(api, pool, handoff):
    state = api.call("/api/state")
    if (state.get("gpu_handoff") or {}).get("requested") is not handoff:
        api.call("/api/gpu-handoff", {"requested": handoff})
    if (state.get("bridge_resource_pool") or {}).get("enabled") is not pool:
        api.call("/api/bridge-resource-pool", {"enabled": pool})


def counters_delta(before, after, group):
    a, b = before.get(group) or {}, after.get(group) or {}
    return {key: b[key] - value for key, value in a.items()
            if type(value) is int and type(b.get(key)) is int and
            not any(term in key for term in ("epoch", "generation", "device", "queue", "thread"))}


def arm(api, root, name, controls, pool, handoff, count=30, warm_seconds=8):
    began = time.monotonic()
    request(api, pool, handoff)
    first = wait_state(api, lambda s: applied(s, pool, handoff), controls)
    first_id = frame(first)
    warm_started = time.monotonic()
    while True:
        if time.monotonic() - began > 90:
            raise ValueError("Arm warmup exceeded its 90-second limit")
        current = api.call("/api/state")
        validate(current, controls)
        if not applied(current, pool, handoff):
            raise ValueError("Bridge capability disappeared during warmup")
        if time.monotonic() - warm_started >= warm_seconds and frame(current) - first_id >= 35:
            break
        time.sleep(.12)
    warm = {"seconds": time.monotonic() - warm_started,
            "new_completed_frames": frame(current) - first_id}
    before = current
    rows, snapshots, last_id = [], [], frame(before)
    while len(rows) < count:
        if time.monotonic() - began > 90:
            raise ValueError("Arm sampling exceeded its 90-second limit")
        state = api.call("/api/state")
        validate(state, controls)
        if not applied(state, pool, handoff):
            raise ValueError("Bridge capability disappeared during measurement")
        fid = frame(state)
        if fid < last_id:
            raise ValueError("Completed frame identity regressed")
        if fid > last_id:
            row = state["processing"]["stages"]["recording"]
            if any(type(row.get(k)) not in (int, float) or not math.isfinite(row[k]) or row[k] < 0 for k in METRICS):
                raise ValueError("Missing/nonfinite completed-frame timing")
            rows.append(copy.deepcopy(row))
            snapshots.append(state)
            last_id = fid
        time.sleep(.12)
    after = snapshots[-1]
    delta = {group: counters_delta(before, after, group) for group in ("bridge_resource_pool", "gpu_handoff", "bridge_cache")}
    if pool and delta["bridge_resource_pool"].get("hits", 0) <= 0:
        raise ValueError("Pool ON did not reuse actual completed resources")
    if handoff and delta["gpu_handoff"].get("prepared_bypasses", 0) <= 0:
        raise ValueError("Handoff ON did not bypass prepared CPU waits")
    for group, keys in (("gpu_handoff", ("process_failures", "previous_consumer_wait_failures")),
                        ("bridge_resource_pool", ("creation_failures", "quarantines")),
                        ("bridge_cache", ("shader_compile_failures",))):
        if any(delta[group].get(key, 0) > 0 for key in keys):
            raise ValueError("New runtime/resource failures in " + group)
    result = {"completed": True, "name": name, "pool_enabled": pool,
              "handoff_enabled": handoff, "warmup": warm,
              "recording_samples": rows, "snapshots": snapshots,
              "before": before, "after": after, "counter_delta": delta,
              "missing_counters": "Unavailable counters are absent, never measured zero",
              "finished_utc": utc()}
    dump(root, name + ".json", result)
    print(json.dumps({"arm": name, "complete": True, "n": len(rows),
                      "record_first_ms": rows[0]["record_last_ms"],
                      "actual_pool_hits": delta["bridge_resource_pool"].get("hits"),
                      "actual_prepared_bypasses": delta["gpu_handoff"].get("prepared_bypasses")}), flush=True)
    return result


def run(args):
    root = args.output.resolve()
    physical(root)
    if root.drive.casefold() != "d:":
        raise ValueError("Live outputs belong on D")
    root.mkdir(parents=True, exist_ok=False)
    pins = source_pins(args.install_receipt)
    api = API()
    api.connect()
    initial = api.call("/api/state")
    dump(root, "initial.json", initial)
    controls = {k: initial["settings"][k] for k in CONTROL_KEYS}
    try:
        validate(initial, controls, configured=False)
        request(api, False, False)
        if (initial.get("bridge_cache") or {}).get("enabled") is not True:
            api.call("/api/bridge-cache", {"enabled": True})
        if initial.get("timing_enabled") is not True:
            api.call("/api/timing", {"enabled": True})
        target = payload(initial["settings"])
        if target != initial["settings"]:
            api.call("/api/state", target)
        life = (api.call("/api/state").get("lifecycle_audit") or {})
        if life.get("enabled") is not True:
            api.call("/api/lifecycle-audit", {"enabled": True})
        ready = wait_state(api, lambda s: applied(s, False, False), controls)
        dump(root, "720-ready.json", ready)
        # Resource reuse and cross-API synchronization are independent comparisons.
        for ordinal, on in enumerate((False, True, True, False), 1):
            arm(api, root, "pool-arm%02d" % ordinal, controls, on, False)
        for ordinal, on in enumerate((False, True, True, False), 1):
            arm(api, root, "handoff-arm%02d" % ordinal, controls, False, on)
        arm(api, root, "combined-arm", controls, True, True)
        # Ninety-second continuity window, bounded separately from a timed arm.
        start = api.call("/api/state")
        steady, started, first_id = [], time.monotonic(), frame(start)
        while time.monotonic() - started < 90:
            state = api.call("/api/state")
            validate(state, controls)
            if not applied(state, True, True):
                raise ValueError("Combined bridge capability lost during continuity")
            steady.append(state)
            time.sleep(.5)
        if frame(steady[-1]) - first_id < 100:
            raise ValueError("Insufficient actual completed NR frames during continuity")
        dump(root, "continuity-90s.json", {"seconds": time.monotonic() - started,
              "new_completed_frames": frame(steady[-1]) - first_id,
              "start": start, "snapshots": steady})
        request(api, False, False)
        restored = wait_state(api, lambda s: applied(s, False, False), controls)
        restored_id = frame(restored)
        restored = wait_state(api, lambda s: applied(s, False, False) and frame(s) - restored_id >= 8, controls)
        after_pins = source_pins(args.install_receipt)
        if after_pins != pins:
            raise ValueError("Source changed during the live comparison")
        result = {"status": "live_pairs_completed_pending_main_review", "completed": True,
                  "started_utc": initial.get("captured_utc"), "finished_utc": utc(),
                  "source_hashes_before": pins, "source_hashes_after": after_pins,
                  "target_settings": target, "preserved_user_controls": controls,
                  "final_state": restored, "final_experiments_OFF": True,
                  "visual_or_Present_acceptance": False, "executed_by": "assigned GPU tester"}
        dump(root, "RESULT.json", result)
        print(json.dumps({"status": result["status"], "output": str(root)}), flush=True)
    except BaseException as exc:
        recovered = None
        try:
            request(api, False, False)
            recovered = api.call("/api/state")
        except Exception as cleanup:
            recovered = {"recovery_error": type(cleanup).__name__}
        dump(root, "FAILED.json", {"status": "failed", "error_type": type(exc).__name__,
              "error": str(exc), "last_state": getattr(exc, "last_state", None),
              "recovered": recovered, "source_hashes_before": pins, "finished_utc": utc()})
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("cpu-check", "run"))
    parser.add_argument("--install-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "run":
        run(args)
        return
    pins = source_pins(args.install_receipt)
    prior = read(Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001\bridge-cache-live-720-pair-v1-20261001\pair\completed-checkpoint.json"))["initial"]
    validate(prior, {key: prior["settings"][key] for key in CONTROL_KEYS})
    candidate = payload(prior["settings"])
    assert candidate["optimizations_720"] == list(TARGET)
    assert all(candidate[k] == prior["settings"][k] for k in CONTROL_KEYS)
    cold = copy.deepcopy(prior)
    cold["active_mode"] = None
    cold["processing"]["stages"] = None
    validate(cold, configured=False)
    assert frame(cold) is None
    try:
        validate(cold)
    except ValueError:
        pass
    else:
        raise AssertionError("A cold transition must not be accepted as a completed frame")
    assert applied({"bridge_resource_pool": {"enabled": False},
                    "gpu_handoff": {"requested": False, "armed": False, "pending": False,
                                    "host": {"applied": False}}}, False, False)
    assert not applied({"bridge_resource_pool": {"enabled": False},
                        "gpu_handoff": {"requested": True, "armed": False,
                                        "host": {"applied": True}}}, False, True)
    physical(args.output)
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output, "CPU_RECEIPT.json", {"status": "cpu_protocol_checked", "pins": pins,
         "target": list(TARGET), "source_sha256": sha(Path(__file__).resolve()),
         "network_requests": 0, "GPU_executed": False, "live_acceptance": False})
    print(json.dumps({"status": "cpu_protocol_checked", "pin_count": len(pins),
                      "network_requests": 0, "GPU_executed": False}))


if __name__ == "__main__":
    main()
