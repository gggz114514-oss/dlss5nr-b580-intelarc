"""Recompute the recorded local timings without importing Torch or a GPU API."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import statistics

DATA = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\native-math-fixture-export-v1-20261001")


def main() -> None:
    rows = []
    for family in ("c128", "c256"):
        for schedule in ("wide", "fused"):
            path = DATA / f"mlp-report-{family}-{schedule}-01/OWNED_MLP_REPORT.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            if report["status"] != "measured_unaccepted_no_quality_gate" or len(report["sites"]) != 1:
                raise RuntimeError("Local measurement did not finish: " + str(path))
            site = report["sites"][0]
            blocks = site["total_blocks"]
            if [block["arm"] for block in blocks] != ["baseline", "candidate", "candidate", "baseline"]:
                raise RuntimeError("Timing order differs")
            arms = {arm: [value for block in blocks if block["arm"] == arm
                          for value in block["timing"]["device_ms"]]
                    for arm in ("baseline", "candidate")}
            if any(len(samples) != 60 for samples in arms.values()):
                raise RuntimeError("Expected two blocks of 30 raw samples per arm")
            means = {arm: statistics.mean(samples) for arm, samples in arms.items()}
            gain = site["local_gain"]
            if abs(means["baseline"] - gain["baseline_mean_ms"]) > 1e-12 or abs(
                    means["candidate"] - gain["candidate_mean_ms"]) > 1e-12:
                raise RuntimeError("Reported mean differs from raw timing")
            comparison = site["after_timing_comparison"]
            if comparison["excluded_nonfinite_pairs"] or not comparison["baseline"]["all_finite"] or not comparison["candidate"]["all_finite"]:
                raise RuntimeError("Nonfinite output in supplied fixture")
            if not all(site["repeatability"].values()):
                raise RuntimeError("Local outputs changed during timing")
            resources = {arm: [{key: kernel.get(key) for key in
                               ("symbol", "registers", "shared_bytes", "spills", "grid", "binary_sha256")}
                              for kernel in site["kernel_resources"][arm]]
                         for arm in ("baseline", "candidate")}
            rows.append({"family": family, "schedule": schedule, "site": site["label"],
                         "receipt": str(path), "receipt_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "baseline_mean_ms": means["baseline"], "candidate_mean_ms": means["candidate"],
                         "saved_ms": means["baseline"] - means["candidate"],
                         "baseline_median_ms": statistics.median(arms["baseline"]),
                         "candidate_median_ms": statistics.median(arms["candidate"]),
                         "baseline_P95_ms": sorted(arms["baseline"])[56],
                         "candidate_P95_ms": sorted(arms["candidate"])[56],
                         "recorded_bitwise_equal": comparison["bitwise_equal"],
                         "resources": resources})
    result = {"status": "recorded_local_timings_recomputed", "rows": rows,
              "scope": "One owned offline site of each family; Event intervals may include submission gaps, not isolated ALU time",
              "quality_accepted": False, "game_performance_evidence": False, "GPU_executed_by_main": False}
    output = DATA / "MAIN_MLP_REVIEW.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
