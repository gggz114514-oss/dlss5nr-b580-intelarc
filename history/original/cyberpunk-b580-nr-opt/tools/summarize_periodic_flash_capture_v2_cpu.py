"""Summarize a completed passive capture; stdlib only, no live API or GPU access."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def summarize(path):
    raw = path.read_bytes()
    records = [json.loads(line) for line in raw.decode("utf-8-sig").splitlines() if line.strip()]
    if not records or records[0].get("kind") != "start":
        raise ValueError("capture has no start record")
    footer = records[-1]
    if footer.get("kind") != "end" or footer.get("stopped") != "duration":
        raise ValueError("read only a capture with a completed duration footer")
    first = records[0]["periodic_flash_v2"]
    last = footer["last_observation"]["periodic_flash_v2"]
    if first.get("status") != "ok" or last.get("status") != "ok":
        raise ValueError("first/end snapshots do not have native v2 evidence")
    low, high = first["counters"]["seen"], last["counters"]["seen"]
    if high < low:
        raise ValueError("native counter restarted during capture")
    deltas = {key: last["counters"][key] - value for key, value in first["counters"].items()}
    loss = {key: last["native_header"][key] - first["native_header"][key]
            for key in ("dropped", "frames_dropped")}
    events = {}
    unknown_records = []
    for record in records:
        observation = record.get("last_observation") or record
        diagnostic = observation.get("periodic_flash_v2", {})
        if diagnostic.get("status") != "ok":
            unknown_records.append({"utc": record.get("capture_utc"),
                                    "status": diagnostic.get("status"),
                                    "reason": diagnostic.get("reason")})
        for window in diagnostic.get("windows", ()):
            sr = window["sr_sequence"]
            # Initial ring contents can belong to earlier parameter switches.
            if not low < sr <= high:
                continue
            for trigger in window.get("triggers", ()):
                if trigger.get("kind") != "fallback":
                    continue
                key = (sr, trigger.get("eval_id"), trigger.get("reason"))
                event = events.setdefault(key, {"sr_sequence": sr, "trigger": trigger, "frames": {}})
                for frame in window.get("frames", ()):
                    event["frames"][frame["sr_sequence"]] = frame
    rows, intervals, previous = [], [], None
    fields = ("sr_sequence", "nr_frame_id", "nr_composed", "raw_fallback", "nr_skipped",
              "native_reason", "color_age", "source_mask", "game_reset", "reset_causes", "observations")
    for event in sorted(events.values(), key=lambda value: value["sr_sequence"]):
        sr, trigger, frames = event["sr_sequence"], event["trigger"], event["frames"]
        steady = trigger["native_steady_us"]
        interval = None if previous is None else (steady - previous) / 1e6
        if interval is not None:
            intervals.append(interval)
        previous = steady
        before, current, after = (frames.get(index, {}) for index in (sr - 1, sr, sr + 1))
        rows.append({"sr_sequence": sr, "reason": trigger.get("reason"),
                     "color_age": trigger.get("color_age"), "source_mask": trigger.get("source_mask"),
                     "interval_seconds": interval,
                     "immediate_composed_neighbors": (before.get("nr_composed") is True
                                                       and after.get("nr_composed") is True),
                     "event_raw_and_skipped": (current.get("raw_fallback") is True
                                                and current.get("nr_skipped") is True),
                     "frames": [{key: frame.get(key) for key in fields}
                                for _, frame in sorted(frames.items())]})
    return {"capture": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest(),
            "start_utc": records[0]["capture_utc"], "end_utc": footer["capture_utc"],
            "samples": footer["samples"], "sr_range": [low, high], "counter_deltas": deltas,
            "diagnostic_drop_deltas": loss, "unknown_records": unknown_records,
            "unique_fallback_windows": len(rows),
            "fallback_windows_match_counter": len(rows) == deltas["original_fallback"],
            "interval_seconds": ({"min": min(intervals), "median": statistics.median(intervals),
                                  "max": max(intervals)} if intervals else None),
            "events": rows, "pixel_or_present_evidence": False,
            "semantics": "completed SR/NR CPU event evidence; initial windows excluded; no live monitoring"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize(args.capture)
    encoded = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
        print(json.dumps({"output": str(args.output.resolve()),
                          "counter_deltas": result["counter_deltas"],
                          "unique_fallback_windows": result["unique_fallback_windows"],
                          "fallback_windows_match_counter": result["fallback_windows_match_counter"],
                          "interval_seconds": result["interval_seconds"]}, ensure_ascii=False))
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
