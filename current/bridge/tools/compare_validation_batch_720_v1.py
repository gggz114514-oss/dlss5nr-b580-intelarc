"""Compare CPU validation policy on the same captured Cyberpunk GPU graph.

Run only by the assigned Luna tester. Changing /api/validation does not recapture
the graph or reset NR history. Candidate versus reference is a separate pairing.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_live_numeric_cost_720_v1 as cost
import benchmark_live_web_720_v1 as base
import sweep_live_web_720_v1 as sweep

REFERENCE = cost.REFERENCE
SLOW = (
    "c512_k8_merge_fma", "c512_k8_post_fma16", "num_decoder_full_k",
    "num_branch_c64", "num_branch_c128", "num_branch_c256",
    "num_vit_denominator_ordered", "num_vit_denominator_fp32",
    "num_vit_norm_fma", "num_vit_exp_fma", "num_vit_norm_exp_fma",
    "num_vit_qkv_full_k", "num_vit_projection_full_k", "num_vit_exp_zero_constant",
    "num_history_dimension_rcp", "num_history_direct_pixel", "num_history_reciprocal",
    "num_post_sigmoid", "num_post_rne", "num_post_rtz",
)


def policy(st):
    value = st.get("validation_batch")
    if not isinstance(value, dict) or type(value.get("enabled")) is not bool:
        raise RuntimeError("frame validation control is unavailable")
    return value


def require_row(snapshot, enabled, frame_id):
    if not isinstance(snapshot, dict) or snapshot.get("enabled") is not enabled:
        return False
    if snapshot.get("active") is not False:
        return False
    snapshot_frame = snapshot.get("frame_id")
    if snapshot_frame is not None and snapshot_frame != frame_id:
        return False
    if enabled:
        if snapshot_frame != frame_id or snapshot.get("retired") is not False:
            return False
        if snapshot.get("full_checks", 0) <= 0 or snapshot.get("reused_checks", 0) <= 0:
            raise RuntimeError("frame validation policy reports no actual reuse/full first check")
    return True


def set_policy(api, enabled):
    value = api.call("/api/validation", {"enabled": enabled})
    # This CPU-only endpoint returns its own scalar state and never takes the
    # native game state lock. The following frame polls check runtime health.
    if not isinstance(value, dict) or value.get("enabled") is not enabled:
        raise RuntimeError("validation policy did not commit")


def source_hashes():
    result = cost.source_hashes()
    plugins = Path(r"G:\epic\Cyberpunk2077\bin\x64\plugins")
    for name in ("CyberpunkNRBridge.asi", "CyberpunkNativeXeSSProbe.asi", "cyberpunk_nr_web.py"):
        path = plugins / name
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    names = ("numeric_frame_validation_720_v1.py", "decoder_input_full_k_720_v1.py",
             "nr_game_controls.py", "history_numeric_suite_720_v1.py",
             "post_numeric_suite_720_v1.py", "front_noise_native_720_v1.py",
             "nr_game_pre_xess_host.py", "periodic_flash_snapshot_v2.py",
             "nr_temporal_diagnostics_v1.py")
    for name in names:
        path = base.GAME / name
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (Path(__file__).resolve(), Path(cost.__file__).resolve()):
        result[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def select_mode(api, reg, selected):
    st = api.call("/api/state")
    base.fatal(st)
    if not base.matches(st, selected):
        api.call("/api/state", base.payload(st["settings"], selected))
    st = base.wait_target(api, selected)
    first = last = st["processing"]["frames"]
    begun = advanced = time.monotonic()
    while time.monotonic() - begun < 8 or last - first < 35:
        time.sleep(.2)
        st = api.call("/api/state")
        base.fatal(st)
        if not base.matches(st, selected):
            raise RuntimeError("actual mode changed during warmup")
        now = st["processing"]["frames"]
        if now > last:
            advanced, last = time.monotonic(), now
        if time.monotonic() - advanced > 15:
            raise RuntimeError("validation warmup stalled")
    sweep.actual_hits(st, selected, reg)
    return cost.cost(st, selected)["epoch"]


def arm(api, reg, selected, enabled, index, frames, expected_epoch):
    set_policy(api, enabled)
    rows, skips = [], 0
    st = api.call("/api/state")
    last_count = cost.cost(st, selected)["completed_frames"]
    start_count = last_count
    started = advanced = time.monotonic()
    while len(rows) < frames:
        time.sleep(.05)
        st = api.call("/api/state")
        base.fatal(st)
        if not base.matches(st, selected) or policy(st)["enabled"] is not enabled:
            raise RuntimeError("mode or validation policy changed during measurement")
        data = cost.cost(st, selected)
        if data["epoch"] != expected_epoch:
            raise RuntimeError("validation toggle changed captured graph/owner epoch")
        count = data["completed_frames"]
        if count > last_count:
            advanced = time.monotonic()
            # Two completed frames flush an in-flight pre-toggle invocation.
            if count - start_count > 2:
                snapshot = policy(st).get("last_snapshot")
                row = data["last"]
                if require_row(snapshot, enabled, row.get("frame_id")):
                    skips += count - last_count - 1
                    rows.append(dict(row, cost_counter=count, validation_frame=dict(snapshot),
                                     validation_frame_join=("exact_frame_id" if snapshot.get("frame_id") == row.get("frame_id")
                                                            else "not_available_while_validation_off"),
                                     web_last_ms=st["processing"].get("last_ms")))
                else:
                    skips += count - last_count
            else:
                skips += count - last_count
            last_count = count
        if time.monotonic() - advanced > 15 or time.monotonic() - started > 90:
            raise RuntimeError("validation arm stalled or exceeded bounded duration")
    return {"arm": index, "validation_enabled": enabled, "selected": list(selected),
            "epoch": expected_epoch, "unique_completed_samples": len(rows),
            "skipped_cost_frames": skips, "samples": rows, "metrics": cost.reduce_samples(rows),
            "actual_hits": sweep.actual_hits(st, selected, reg), "finished_utc": base.utc()}


def totals(delta):
    names = ("process_wall_ms", "adapter_handoff_cpu_ms", "observer_finish_cpu_ms")
    return {k: sum(delta[name][k] for name in names)
            for k in ("baseline_ms", "candidate_ms", "c_minus_b_ms", "baseline_drift_ms")}


def bracketed(candidate, before, after):
    result = {}
    for key in cost.KEYS:
        left, value, right = (row["metrics"][key]["mean_ms"] for row in (before, candidate, after))
        baseline = (left + right) / 2
        result[key] = {"baseline_ms": baseline, "candidate_ms": value,
                       "c_minus_b_ms": value - baseline, "baseline_drift_ms": right - left,
                       "below_both_baselines": value < min(left, right),
                       "above_both_baselines": value > max(left, right)}
    return result


def screen_arm(api, reg, selected, index, frames):
    set_policy(api, True)
    row = cost.arm(api, reg, selected, index, frames)
    # A poll can straddle snapshot publication and meter finish; retry a few
    # completed frames instead of mistaking that publication race for failure.
    for _ in range(10):
        st = api.call("/api/state")
        base.fatal(st)
        if policy(st)["enabled"] is not True or not base.matches(st, selected):
            raise RuntimeError("fixed-policy screen mode/policy changed")
        if require_row(policy(st).get("last_snapshot"), True,
                       cost.cost(st, selected)["last"].get("frame_id")):
            break
        time.sleep(.05)
    else:
        raise RuntimeError("fixed-policy screen lacks an actual completed validation frame")
    row["validation_enabled"] = True
    return row


def screen(api, reg, names, frames, output, frozen, result):
    # Screen every previously slow flag; brackets amortize baseline warmup.
    # A gain candidate can then receive its own longer B-C-C-B confirmation.
    result["semantics"] = "fixed validation reuse on; direct completed GPU/CPU samples; up to four candidate additions bracketed by the same four-flag reference; individual differences are not additive"
    left = screen_arm(api, reg, REFERENCE, "reference-open", frames)
    base.write_json(output / "reference-open.json", left)
    for offset in range(0, len(names), 4):
        rows = []
        for name in names[offset:offset + 4]:
            if name == "reference":
                continue
            row = screen_arm(api, reg, (*REFERENCE, name), name, frames)
            rows.append((name, row))
            base.write_json(output / f"{name}-screen.json", row)
            print(json.dumps({"screened_candidate": name,
                              "mean_ms": {key: value["mean_ms"] for key, value in row["metrics"].items()}},
                             ensure_ascii=False), flush=True)
        right = screen_arm(api, reg, REFERENCE, "reference-close", frames)
        group = []
        for name, row in rows:
            paired = bracketed(row, left, right)
            group.append({"candidate": name, "arms": [left, row, right],
                          "paired_metrics": paired, "total_tracked_adapter": totals(paired)})
        result["pairs"].extend(group)
        if source_hashes() != frozen:
            raise RuntimeError("source/config changed inside frozen candidate screening")
        base.write_json(output / f"block-{offset // 4 + 1:02d}-completed.json", group)
        left = right


def run(args):
    output = args.output.resolve()
    if not output.is_relative_to(base.OUT.parent.resolve()) or output.exists():
        raise ValueError("use a fresh task-owned PERF child directory")
    selected_names = list(SLOW) if args.candidates == "all-slow" else args.candidates.split(",")
    reg = base.registry()
    for name in selected_names:
        if name != "reference" and (name not in reg.PROFILES or name in REFERENCE):
            raise ValueError("invalid independent candidate: " + name)
        reg.combined_mode_options(REFERENCE if name == "reference" else (*REFERENCE, name))
    api = sweep.FixedControlsAPI()
    api.connect()
    initial = api.call("/api/state")
    base.fatal(initial)
    base.validate_base(initial)
    policy(initial)
    api.controls = {key: initial["settings"][key] for key in sweep.CONTROL_KEYS}
    if initial.get("timing_enabled") is not True:
        st = api.call("/api/timing", {"enabled": True})
        base.fatal(st)
        if st.get("timing_enabled") is not True:
            raise RuntimeError("cost timing did not enable")
    frozen = source_hashes()
    output.mkdir(parents=True)
    result = {"status": "running", "started_utc": base.utc(), "pairs": [],
              "source_hashes": frozen, "reference": list(REFERENCE),
              "semantics": "same captured graph, validation off/on/on/off; graph GPU includes memory; CPU submission overlaps GPU, do not sum; total tracked adapter is host wall + handoff + observer"}
    try:
        if args.command == "screen":
            screen(api, reg, selected_names, args.frames, output, frozen, result)
            result["status"] = "completed"
            return
        for name in selected_names:
            selected = REFERENCE if name == "reference" else (*REFERENCE, name)
            set_policy(api, False)
            epoch = select_mode(api, reg, selected)
            rows = []
            for index, enabled in enumerate((False, True, True, False), 1):
                row = arm(api, reg, selected, enabled, index, args.frames, epoch)
                rows.append(row)
                base.write_json(output / f"{name}-arm-{index}.json", row)
                print(json.dumps({"candidate": name, "completed_arm": index,
                                  "validation_enabled": enabled,
                                  "mean_ms": {key: value["mean_ms"] for key, value in row["metrics"].items()}},
                                 ensure_ascii=False), flush=True)
            paired = cost.paired(rows)
            entry = {"candidate": name, "arms": rows, "paired_metrics": paired,
                     "total_tracked_adapter": totals(paired)}
            result["pairs"].append(entry)
            if source_hashes() != frozen:
                raise RuntimeError("source/config changed inside frozen validation comparison")
            base.write_json(output / f"{name}-completed.json", entry)
        result["status"] = "completed"
    except Exception as error:
        result.update(status="stopped", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        try:
            set_policy(api, True)
            st = api.call("/api/state")
            base.fatal(st)
            count = st["processing"]["frames"]
            if not base.matches(st, REFERENCE):
                api.call("/api/state", base.payload(st["settings"], REFERENCE))
            st, hits, frames = base.wait_restore_evidence(api, REFERENCE, reg, count)
            result["restored"] = {"state": base.clean(st), "validation_batch": policy(st),
                                  "hits": hits, "frames": frames}
        except Exception as error:
            result["restore_error"] = type(error).__name__ + ": " + str(error)
            result["status"] = "stopped"
        result["finished_utc"] = base.utc()
        base.write_json(output / ("completed-checkpoint.json" if result["status"] == "completed"
                                  else "stopped-checkpoint.json"), result)


def self_test():
    class CPUOnlyAPI:
        def call(self, path, payload):
            assert path == "/api/validation" and type(payload["enabled"]) is bool
            return {"enabled": payload["enabled"], "last_snapshot": None}
    set_policy(CPUOnlyAPI(), True)
    set_policy(CPUOnlyAPI(), False)
    row = {"enabled": True, "active": False, "retired": False, "frame_id": 42,
           "full_checks": 2, "reused_checks": 3}
    assert require_row(row, True, 42)
    assert not require_row(row, True, 43)
    assert not require_row({**row, "active": True}, True, 42)
    assert require_row({"enabled": False, "active": False}, False, 42)
    assert require_row({"enabled": False, "frame_id": 42, "active": False}, False, 42)
    assert not require_row({"enabled": False, "frame_id": 41, "active": False, "retired": False}, False, 42)
    delta = {key: {"baseline_ms": 1., "candidate_ms": .5, "c_minus_b_ms": -.5,
                   "baseline_drift_ms": .1} for key in cost.KEYS}
    assert totals(delta)["c_minus_b_ms"] == -1.5
    arms = [{"metrics": {key: {"mean_ms": v} for key in cost.KEYS}} for v in (5, 4, 5)]
    assert bracketed(arms[1], arms[0], arms[2])["network_graph_gpu_ms"]["below_both_baselines"]
    assert bracketed(arms[1], arms[0], arms[2])["process_wall_ms"]["c_minus_b_ms"] == -1
    assert "torch" not in sys.modules
    print(json.dumps({"status": "passed", "API_touched": False, "GPU_imported": False,
                      "slow_candidate_count": len(SLOW)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "run", "screen"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidates", default="reference,num_branch_c64,num_branch_c128,num_branch_c256,num_vit_denominator_fp32,num_decoder_full_k,num_post_sigmoid,num_history_direct_pixel")
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    if args.command == "self-test":
        self_test()
    else:
        if args.output is None or not 30 <= args.frames <= 180:
            raise ValueError("provide --output and 30..180 completed samples per arm")
        run(args)
