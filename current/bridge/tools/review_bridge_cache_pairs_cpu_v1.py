"""Recompute the live cache comparison from unique completed-frame samples.

Standard library only; never connects to the game, loads the runtime or uses GPU.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    root = args.input.resolve(strict=True)
    if root.drive.casefold() != "d:":
        raise ValueError("Read only the task's D experiment results")
    checkpoint = read(root / "completed-checkpoint.json")
    if checkpoint.get("completed") is not True or checkpoint.get("status") != "passed":
        raise RuntimeError("A completed, healthy live comparison is required")
    if checkpoint["source_hashes_before"] != checkpoint["source_hashes_after"]:
        raise RuntimeError("Source changed during the comparison")
    if checkpoint["initial"]["settings"] != checkpoint["restored"]["settings"]:
        raise RuntimeError("Non-cache settings were not restored")
    if checkpoint["restored"]["bridge_cache"]["enabled"] is not True:
        raise RuntimeError("Leave the accepted cache ON")
    arms, pins = [], {}
    for ordinal, enabled in enumerate((False, True, True, False), 1):
        path = root / f"arm{ordinal:02d}-completed.json"
        arm = read(path)
        pins[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if arm["completed"] is not True or arm["enabled"] is not enabled:
            raise RuntimeError("Wrong cache arm or incomplete result")
        if arm["warmup"]["new_completed_frames"] < 35 or arm["warmup"]["seconds"] < 8:
            raise RuntimeError("Cold frames were not discarded")
        samples = arm["recording_samples"]
        ids = [row["last_nr_frame_id"] for row in samples]
        if len(samples) < 30 or len(set(ids)) != len(ids) or ids != sorted(ids):
            raise RuntimeError("Need unique increasing completed-frame samples")
        metrics = {}
        for name in arm["recording_metrics"]:
            values = [row[name] for row in samples]
            if not all(math.isfinite(n) and n >= 0 for n in values):
                raise RuntimeError("Invalid wall-clock sample: " + name)
            metrics[name] = {"n": len(values), "mean_ms": statistics.fmean(values),
                             "p50_ms": statistics.median(values),
                             "p95_ms": sorted(values)[math.ceil(.95 * len(values)) - 1]}
            for field in ("mean_ms", "p50_ms", "p95_ms"):
                if abs(metrics[name][field] - arm["recording_metrics"][name][field]) > 1e-9:
                    raise RuntimeError("Published statistics do not match raw samples")
        delta = arm["cache_delta_measurement"]
        if enabled:
            unchanged = ("shader_compile_calls", "source_reads", "source_hashes",
                         "root_signature_creates", "pso_creates", "uncached_initializations")
            if any(delta[name] != 0 for name in unchanged) or delta["pipeline_hits"] <= 0:
                raise RuntimeError("Cache ON still rebuilt fixed resources")
        elif (delta["shader_compile_calls"] != 2 * delta["uncached_initializations"] or
              delta["pso_creates"] != 2 * delta["uncached_initializations"]):
            raise RuntimeError("The uncached control did not execute the expected work")
        if delta["shader_compile_failures"] != 0:
            raise RuntimeError("Shader compilation failed")
        arms.append({"enabled": enabled, "metrics": metrics})
    paired = {}
    for name in arms[0]["metrics"]:
        off = statistics.fmean(arms[i]["metrics"][name]["mean_ms"] for i in (0, 3))
        on = statistics.fmean(arms[i]["metrics"][name]["mean_ms"] for i in (1, 2))
        paired[name] = {"off_ms": off, "on_ms": on, "saved_ms": off - on,
                        "both_on_below_both_off":
                        max(arms[i]["metrics"][name]["mean_ms"] for i in (1, 2)) <
                        min(arms[i]["metrics"][name]["mean_ms"] for i in (0, 3))}
        published = checkpoint["paired"]["recording_metrics"][name]
        if abs(published["on_minus_off_ms"] - (on - off)) > 1e-9:
            raise RuntimeError("Paired delta mismatch")
    result = {"status": "main_raw_review_passed", "actual_NR_height": checkpoint["actual_NR_height"],
              "settings": checkpoint["initial"]["settings"], "paired": paired,
              "raw_source_pins": pins, "cache_restored_ON": True, "GPU_executed": False,
              "scope": "Completed-frame CPU recording and span. Span includes game and waits; no Present/FPS claim."}
    target = root / "MAIN_REVIEW.json"
    if target.exists():
        raise FileExistsError("Preserve the previous review receipt")
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "NR_height": result["actual_NR_height"],
                      "record": paired["record_last_ms"], "receipt": str(target)}))


if __name__ == "__main__":
    main()
