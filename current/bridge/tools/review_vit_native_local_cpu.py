"""Recompute local ViT timings from raw samples; no GPU or live API access."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def stats(values):
    if not values or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Invalid timing samples")
    ordered = sorted(values)
    return {"n": len(values), "mean_ms": statistics.fmean(values),
            "p50_ms": statistics.median(values),
            "p95_ms": ordered[math.ceil(0.95 * len(ordered)) - 1]}


def review(path):
    result = json.loads(path.read_text(encoding="utf-8-sig"))
    if result.get("status") != "local_gates_passed_quality_and_speed_unaccepted":
        raise ValueError("Local probe did not finish successfully")
    if not result.get("gpu_executed") or not result.get("local_graphs_retired_before_binary_owners_release"):
        raise ValueError("Actual execution and resource retirement are required")
    metrics = {}
    for label, rows in result["timing"].items():
        if [row["arm"] for row in rows] != ["B", "C", "C", "B"]:
            raise ValueError("Expected baseline/candidate/candidate/baseline arms")
        intervals = {}
        for clock in ("device_interval", "call_plus_completion_wall"):
            collected = {"B": [], "C": []}
            means = []
            for row in rows:
                values = row[clock]["samples_ms"]
                checked = stats(values)
                if not math.isclose(checked["mean_ms"], row[clock]["mean_ms"], rel_tol=1e-10, abs_tol=1e-10):
                    raise ValueError("Published mean differs from raw samples")
                collected[row["arm"]].extend(values)
                means.append(checked["mean_ms"])
            base, candidate = stats(collected["B"]), stats(collected["C"])
            intervals[clock] = {
                "baseline": base, "candidate": candidate,
                "saved_mean_ms": base["mean_ms"] - candidate["mean_ms"],
                "both_candidates_below_both_baselines": max(means[1:3]) < min(means[0], means[3]),
                "baseline_arm_drift_ms": means[3] - means[0],
            }
        metrics[label] = intervals
    numerical = result["numerical"]
    if not all(v["finite_baseline"] and v["finite_candidate"] for v in numerical.values()):
        raise ValueError("Nonfinite output")
    return {
        "status": "main_local_raw_review_passed",
        "input": str(path), "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "metrics": metrics, "numerical": numerical,
        "matrix_shapes": result["matrix_shapes"], "quality_acceptance": result["quality_acceptance"],
        "compile_policy": result["compile_policy"],
        "GPU_executed_by_main": False, "whole_model_or_game_acceptance": False,
        "scope": "Frozen local reference seam on authentic 720p-model QKV; not whole-game timing or current global-provider equivalence.",
        "p95_policy": "Nearest rank ceil(0.95*n) on combined raw samples; not the original small-sample floor percentile.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path("D:/Codex-NR-Experiments/cyberpunk-opt/vit-baseline-cache-prepare-v1-20261002").resolve()
    source, target = args.input.resolve(strict=True), args.output.resolve()
    if not source.is_relative_to(root) or not target.is_relative_to(root):
        raise ValueError("Use the owned ViT experiment directory")
    result = review(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": result["status"], "receipt": str(target),
                      "local_graph": result["metrics"]["complete_chain_and_layout_local_graph"]["device_interval"]}))


if __name__ == "__main__":
    main()
