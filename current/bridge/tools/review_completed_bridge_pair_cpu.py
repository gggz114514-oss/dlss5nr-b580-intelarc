"""Independently reduce four raw completed-frame arms; no runtime or network.

Timing statistics alone do not attest that an experimental switch was applied.
Keep the tester's health, activation and source evidence as separate gates.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


METRICS = (
    "record_last_ms", "resources_last_ms", "hdr_initialize_last_ms",
    "xess_record_last_ms", "record_to_submit_gap_last_ms",
    "record_to_retire_span_last_ms",
)


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def physical(path):
    for part in (path, *path.parents):
        if part.is_symlink() or part.is_junction():
            raise ValueError("Redirected result path: " + str(part))


def recording(row):
    if "last_nr_frame_id" in row:
        return row
    state = row.get("state", row)
    return state["processing"]["stages"]["recording"]


def reduce_arm(path):
    data = read(path)
    samples = data.get("recording_samples", data.get("samples"))
    if not isinstance(samples, list) or len(samples) < 30:
        raise ValueError("At least 30 raw completed frames required: " + str(path))
    rows = [recording(row) for row in samples]
    ids = [row["last_nr_frame_id"] for row in rows]
    if any(type(value) is not int or value < 1 for value in ids):
        raise ValueError("Missing or invalid completed-frame identity")
    if any(b <= a for a, b in zip(ids, ids[1:])):
        raise ValueError("Duplicate or regressed frames must not be measured again")
    metrics = {}
    for key in METRICS:
        values = [row.get(key) for row in rows]
        if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in values):
            raise ValueError("Missing or invalid timing; do not substitute zero: " + key)
        ordered = sorted(values)
        metrics[key] = {"n": len(values), "mean_ms": statistics.fmean(values),
                        "p50_ms": statistics.median(values),
                        "p95_ms": ordered[math.ceil(.95 * len(values)) - 1]}
    return {"input": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "first_frame_id": ids[0], "last_frame_id": ids[-1], "metrics": metrics,
            "warmup_evidence": data.get("warmup")}, set(ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs=4, type=Path, required=True,
                        help="Four raw arms in actual OFF/ON/ON/OFF order")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    arms, seen = [], set()
    for path in args.arms:
        physical(path)
        path = path.resolve(strict=True)
        if path.drive.casefold() != "d:":
            raise ValueError("Read the task's D experiment results only")
        arm, ids = reduce_arm(path)
        if seen.intersection(ids):
            raise ValueError("The same completed frame occurs in more than one arm")
        seen.update(ids)
        arms.append(arm)
    paired = {}
    for key in METRICS:
        off_values = [arms[i]["metrics"][key]["mean_ms"] for i in (0, 3)]
        on_values = [arms[i]["metrics"][key]["mean_ms"] for i in (1, 2)]
        off, on = statistics.fmean(off_values), statistics.fmean(on_values)
        paired[key] = {"off_ms": off, "on_ms": on, "saved_ms": off - on,
                       "both_on_below_both_off": max(on_values) < min(off_values),
                       "off_arm_drift_ms": off_values[1] - off_values[0]}
    physical(args.output)
    output = args.output.resolve()
    if output.drive.casefold() != "d:" or not output.parent.is_dir():
        raise ValueError("Use an existing D experiment output directory")
    result = {"status": "main_raw_statistics_reviewed", "arms": arms, "paired": paired,
              "network_or_GPU_executed": False, "runtime_or_visual_acceptance": False,
              "scope": "Unique completed-frame CPU recording and span; span includes game and waits, not Present/FPS.",
              "missing_metrics_substituted": False,
              "p95_policy": "Nearest rank ceil(0.95*n) on each raw arm"}
    with output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"status": result["status"], "record": paired[METRICS[0]],
                      "span": paired[METRICS[-1]], "receipt": str(output)}))


if __name__ == "__main__":
    main()
