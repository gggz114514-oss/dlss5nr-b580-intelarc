"""Pair actual Cyberpunk frames with timestamped graph and CPU guard durations.

The main process imports no GPU package. Only the game opt-in cost meter records
GPU timestamps. Raw samples are completed-frame values, not overlapping window
averages. The currently supported good combination is restored on completion.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import benchmark_live_web_720_v1 as base
import sweep_live_web_720_v1 as sweep

REFERENCE = (*base.OLD, "num_front_both")
KEYS = ("process_wall_ms", "network_graph_gpu_ms", "network_replay_submit_cpu_ms",
        "numeric_guard_cpu_ms", "graph_constant_guard_cpu_ms", "observer_finish_cpu_ms",
        "adapter_handoff_cpu_ms", "runtime_prepare_wall_ms", "runtime_model_wall_ms", "runtime_export_wall_ms",
        "child_branch_accum_guard_cpu_ms", "child_vit_guard_cpu_ms",
        "child_history_guard_cpu_ms", "child_front_guard_cpu_ms",
        "child_decoder_input_guard_cpu_ms", "child_post_guard_cpu_ms")


def cost(st, selected):
    data = ((st.get("processing") or {}).get("stages") or {}).get("cost_decomposition")
    if not isinstance(data, dict) or data.get("enabled") is not True:
        raise RuntimeError("live cost meter unavailable; diagnostic restart required")
    if data.get("error") is not None:
        raise RuntimeError("live cost meter failed: " + str(data["error"]))
    if tuple(sorted(data.get("optimizations_720", ()))) != tuple(sorted(selected)):
        raise RuntimeError("cost meter still belongs to another selected mode")
    return data


def reduce_samples(rows):
    if len(rows) < 30:
        raise RuntimeError("fewer than 30 unique completed cost samples")
    for row in rows:
        if row.get("network_replay_count") != 1:
            raise RuntimeError("sample did not have exactly one network replay")
        for key in ("process_wall_ms", "network_graph_gpu_ms", "numeric_guard_cpu_ms",
                    "graph_constant_guard_cpu_ms"):
            if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) or row[key] < 0:
                raise RuntimeError("invalid required cost sample: " + key)
    return {key: {"mean_ms": statistics.fmean(row.get(key, 0.0) for row in rows),
                  "p50_ms": statistics.median(row.get(key, 0.0) for row in rows)} for key in KEYS}


def paired(rows):
    deltas = {}
    for key in KEYS:
        values = [row["metrics"][key]["mean_ms"] for row in rows]
        b = (values[0] + values[3]) / 2
        c = (values[1] + values[2]) / 2
        deltas[key] = {"baseline_ms": b, "candidate_ms": c, "c_minus_b_ms": c - b,
                       "baseline_drift_ms": values[3] - values[0],
                       "both_candidates_below_both_baselines": max(values[1:3]) < min(values[0], values[3]),
                       "both_candidates_above_both_baselines": min(values[1:3]) > max(values[0], values[3])}
    return deltas


def arm(api, reg, selected, index, frames):
    st = api.call("/api/state")
    base.fatal(st)
    if not base.matches(st, selected):
        api.call("/api/state", base.payload(st["settings"], selected))
    st = base.wait_target(api, selected)
    initial = (st.get("processing") or {}).get("frames")
    begin, advanced, last = time.monotonic(), time.monotonic(), initial
    while time.monotonic() - begin < 8 or last - initial < 35:
        time.sleep(.2)
        st = api.call("/api/state")
        base.fatal(st)
        if not base.matches(st, selected):
            raise RuntimeError("actual mode changed during cost warmup")
        now = (st.get("processing") or {}).get("frames")
        if not isinstance(now, int):
            raise RuntimeError("completed processing counter unavailable")
        if now > last:
            advanced, last = time.monotonic(), now
        if time.monotonic() - advanced > 15:
            raise RuntimeError("cost warmup stalled")
    sweep.actual_hits(st, selected, reg)
    start = cost(st, selected)
    epoch, last_count = start["epoch"], start["completed_frames"]
    rows, skips, advanced = [], 0, time.monotonic()
    measure_started = time.monotonic()
    while len(rows) < frames:
        time.sleep(.05)
        st = api.call("/api/state")
        base.fatal(st)
        if not base.matches(st, selected):
            raise RuntimeError("actual mode changed during cost measurement")
        sample = cost(st, selected)
        if sample["epoch"] != epoch:
            raise RuntimeError("cost meter owner/epoch changed during steady measurement")
        count = sample["completed_frames"]
        if count > last_count:
            skips += count - last_count - 1
            rows.append(dict(sample["last"], cost_counter=count,
                             web_last_ms=(st.get("processing") or {}).get("last_ms")))
            advanced, last_count = time.monotonic(), count
        if time.monotonic() - advanced > 15 or time.monotonic() - measure_started > 90:
            raise RuntimeError("cost measurement stalled or exceeded its bounded duration")
    hits = sweep.actual_hits(st, selected, reg)
    return {"arm": index, "selected": list(selected), "epoch": epoch,
            "unique_completed_samples": len(rows), "skipped_cost_frames": skips,
            "metrics": reduce_samples(rows), "samples": rows,
            "actual_hits": hits, "finished_utc": base.utc()}


def source_hashes():
    project = Path(__file__).resolve().parents[1]
    installed = Path(r"G:\epic\Cyberpunk2077\bin\x64\plugins")
    paths = [installed / "cyberpunk_nr_adapter.py", installed / "nr_numeric_cost_meter_v1.py",
             base.GAME / "nr_game_fullsize.py", base.GAME / "numeric_cleanup_suite_720_v1.py",
             base.GAME / "branch_accum_native_720_v1.py", base.GAME / "branch_accum_native_720_kernel_v1.py",
             base.GAME / "vit_numeric_suite_720_v1.py", base.GAME / "numeric_game_profiles_720_v1.py",
             base.GAME.parent / "data/product-v1/local-runtime-v1.json"]
    result = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    for name in ("cyberpunk_nr_adapter.py", "nr_numeric_cost_meter_v1.py"):
        if hashlib.sha256((project / "game" / name).read_bytes()).hexdigest() != result[str(installed / name)]:
            raise RuntimeError("diagnostic installed/source mismatch: " + name)
    return result


def run(args):
    output = args.output.resolve()
    if not output.is_relative_to(base.OUT.parent.resolve()) or output.exists():
        raise ValueError("use a fresh task-owned PERF subdirectory")
    output.mkdir(parents=True)
    api, reg = sweep.FixedControlsAPI(), base.registry()
    api.connect()
    initial = api.call("/api/state")
    base.fatal(initial)
    base.validate_base(initial)
    api.controls = {key: initial["settings"][key] for key in sweep.CONTROL_KEYS}
    if initial.get("timing_enabled") is False:
        timing_state = api.call("/api/timing", {"enabled": True})
        base.fatal(timing_state)
        if timing_state.get("timing_enabled") is not True:
            raise RuntimeError("timing control did not enable")
    elif initial.get("timing_enabled") is not True:
        raise RuntimeError("timing control unavailable")
    frozen = source_hashes()
    result = {"status": "running", "started_utc": base.utc(), "pairs": [],
              "source_hashes": frozen, "reference": list(REFERENCE),
              "semantics": "same-scene Cyberpunk direct per-frame CPU guards and captured-graph GPU timestamps; graph GPU includes internal memory work; CPU submission overlaps GPU, do not sum"}
    try:
        for candidate in args.candidates.split(","):
            if candidate not in reg.PROFILES or candidate in REFERENCE:
                raise ValueError("invalid independent candidate: " + candidate)
            target = (*REFERENCE, candidate)
            reg.combined_mode_options(target)
            rows = []
            for index, selected in enumerate((REFERENCE, target, target, REFERENCE), 1):
                row = arm(api, reg, selected, index, args.frames)
                rows.append(row)
                base.write_json(output / f"{candidate}-arm-{index}.json", row)
                print(json.dumps({"completed_arm": index, "candidate": candidate,
                                  "mean_ms": {k: v["mean_ms"] for k, v in row["metrics"].items()}},
                                 ensure_ascii=False), flush=True)
            result["pairs"].append({"candidate": candidate, "arms": rows, "paired_metrics": paired(rows)})
            if source_hashes() != frozen:
                raise RuntimeError("source/config changed inside frozen paired measurement")
            base.write_json(output / f"{candidate}-completed.json", result["pairs"][-1])
        result["status"] = "completed"
    except Exception as error:
        result.update(status="stopped", error=type(error).__name__ + ": " + str(error))
        raise
    finally:
        try:
            st = api.call("/api/state")
            base.fatal(st)
            count = st["processing"]["frames"]
            if not base.matches(st, REFERENCE):
                api.call("/api/state", base.payload(st["settings"], REFERENCE))
            st, hits, frames = base.wait_restore_evidence(api, REFERENCE, reg, count)
            result["restored"] = {"state": base.clean(st), "hits": hits, "frames": frames}
        except Exception as error:
            result["restore_error"] = type(error).__name__ + ": " + str(error)
            result["status"] = "stopped"
        result["finished_utc"] = base.utc()
        base.write_json(output / ("completed-checkpoint.json" if result["status"] == "completed" else "stopped-checkpoint.json"), result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("self-test", "run"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--candidates", default="num_branch_c64,num_vit_denominator_fp32")
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    if args.command == "self-test":
        sample = {key: 1.0 for key in KEYS}
        sample["network_replay_count"] = 1
        stats = reduce_samples([sample] * 30)
        rows = [{"metrics": {key: {"mean_ms": value} for key in KEYS}} for value in (5, 6, 6, 5)]
        assert stats["network_graph_gpu_ms"]["mean_ms"] == 1
        assert paired(rows)["network_graph_gpu_ms"]["both_candidates_above_both_baselines"]
        print(json.dumps({"status": "passed", "gpu_imported": False, "API_touched": False}))
    else:
        if args.output is None or not 30 <= args.frames <= 180:
            raise ValueError("provide --output and 30..180 samples per arm")
        run(args)


if __name__ == "__main__":
    main()
