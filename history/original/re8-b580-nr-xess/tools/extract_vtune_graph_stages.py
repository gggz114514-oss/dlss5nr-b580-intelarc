"""Read per-task GPU timestamps around named graph markers from VTune SQLite.

VTune's ordinary computing-task report aggregates identical kernel names and
loses their order. This reads the same result's task instances without changing
the capture. Timings are exploratory: reject incomplete/out-of-order sequences,
compare VTune's elapsed-time report with the assumed TSC frequency, and retain
the uninstrumented whole-frame control separately.
"""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter
import json
from pathlib import Path
import sqlite3
import statistics

from summarize_vtune_tasks import classify_task


DEFAULT_GROUPS = (
    "pre", "encoder C32", "encoder C64", "encoder C128", "encoder C256",
    "encoder C512", "ViT", "decoder C512", "decoder C256",
    "decoder C128", "decoder C64", "decoder C32", "RGB/post",
)


def read_tasks(db: Path):
    connection = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True)
    try:
        rows = connection.execute("""
            SELECT g.start_tsc, g.end_tsc, t.name, c.queue,
                   c.type, t.global_dim_str
            FROM gpu_compute_task_data AS g
            JOIN dd_compute_task AS c ON g.attr = c.rowid
            JOIN dd_compute_task_type AS t ON c.type = t.rowid
            ORDER BY g.start_tsc
        """).fetchall()
        elapsed = connection.execute("""
            SELECT MAX(end_tsc) - MIN(start_tsc) FROM global_elapsed_time_data
        """).fetchone()[0]
    finally:
        connection.close()
    return rows, elapsed


def summarize(db: Path, markers: tuple[str, ...], groups: tuple[str, ...],
              drop_first: int, drop_last: int, ticks_per_second: float,
              expected_collection_seconds: float | None) -> dict:
    if len(markers) != len(groups) + 1:
        raise ValueError("Need one more marker than groups")
    if drop_first < 0 or drop_last < 0:
        raise ValueError("Warm-up and trailing sequence exclusions must be nonnegative")
    rows, collection_ticks = read_tasks(db)
    bad_timestamps = [row for row in rows if row[1] <= row[0]]
    if any(row[2] in markers for row in bad_timestamps):
        raise ValueError("A graph stage marker has an invalid GPU timestamp")
    rows = [row for row in rows if row[1] > row[0]]
    collection_seconds = collection_ticks / ticks_per_second
    if (expected_collection_seconds is not None and
            abs(collection_seconds - expected_collection_seconds) >
            max(0.02, expected_collection_seconds * 0.005)):
        raise ValueError("VTune collection TSC frequency calibration failed: "
                         f"{collection_seconds} vs {expected_collection_seconds}")

    sequences = []
    partial = []
    incomplete = 0
    for row in rows:
        start, end, name, queue = row[:4]
        if name == markers[0]:
            if partial:
                incomplete += 1
            partial = [(start, end, name, queue)]
        elif partial and name in markers:
            if name != markers[len(partial)]:
                incomplete += 1
                partial = []
                continue
            partial.append((start, end, name, queue))
            if len(partial) == len(markers):
                if len({part[3] for part in partial}) == 1:
                    sequences.append(tuple(partial))
                else:
                    incomplete += 1
                partial = []
    if partial:
        incomplete += 1
    usable = sequences[drop_first:len(sequences) - drop_last if drop_last else None]
    starts = [row[0] for row in rows]
    tick_per_ms = ticks_per_second / 1000.0
    frames = []
    invalid = []
    for sequence_index, sequence in enumerate(usable, start=drop_first):
        groups_out = []
        for index, group in enumerate(groups):
            begin = sequence[index][1]
            end = sequence[index + 1][0]
            if begin >= end:
                invalid.append({"sequence": sequence_index, "group": group,
                                "reason": "nonpositive marker interval"})
                break
            first = bisect_left(starts, begin)
            last = bisect_left(starts, end)
            body = [row for row in rows[first:last]
                    if row[2] not in markers and row[3] == sequence[0][3]]
            crossing = sum(row[1] > end for row in body)
            durations = Counter()
            signatures = Counter()
            category_durations = Counter()
            category_counts = Counter()
            for task_start, task_end, name, _, type_id, global_dim in body:
                duration_ms = (task_end - task_start) / tick_per_ms
                durations[name] += duration_ms
                signatures[f"{name} [type={type_id}, grid={global_dim}]"] += duration_ms
                category = classify_task(name)
                category_durations[category] += duration_ms
                category_counts[category] += 1
            groups_out.append({
                "name": group,
                "elapsed_ms": (end - begin) / tick_per_ms,
                "task_duration_sum_ms": sum(row[1] - row[0]
                                              for row in body) / tick_per_ms,
                "task_count": len(body),
                "crossing_tasks": crossing,
                "excluded_other_queue_tasks": (last - first) - len(body),
                "task_duration_by_category_ms": dict(category_durations),
                "task_count_by_category": dict(category_counts),
                "task_duration_by_name_ms": dict(durations),
                "task_duration_by_signature_ms": dict(signatures),
            })
        else:
            frames.append({
                "sequence": sequence_index,
                "marker_queue": sequence[0][3],
                "marker_span_ms": (sequence[-1][1] - sequence[0][0]) / tick_per_ms,
                "groups": groups_out,
            })
    if not frames:
        raise RuntimeError("No valid marker sequences remain")
    summary = {}
    for index, group in enumerate(groups):
        values = [frame["groups"][index]["elapsed_ms"] for frame in frames]
        tasks = [frame["groups"][index]["task_count"] for frame in frames]
        crossing = [frame["groups"][index]["crossing_tasks"] for frame in frames]
        category_durations = Counter()
        category_counts = Counter()
        names = Counter()
        signatures = Counter()
        for frame in frames:
            part = frame["groups"][index]
            category_durations.update(part["task_duration_by_category_ms"])
            category_counts.update(part["task_count_by_category"])
            names.update(part["task_duration_by_name_ms"])
            signatures.update(part["task_duration_by_signature_ms"])
        summary[group] = {"median_interval_ms": statistics.median(values),
                          "min_interval_ms": min(values),
                          "max_interval_ms": max(values),
                          "median_gpu_task_count": statistics.median(tasks),
                          "median_crossing_tasks": statistics.median(crossing),
                          "max_crossing_tasks": max(crossing),
                          "category_task_duration_sum_ms": dict(category_durations),
                          "category_task_count": dict(category_counts),
                          "top_task_names_by_duration_ms": names.most_common(10)}
        summary[group]["top_task_signatures_by_duration_ms"] = signatures.most_common(10)
    return {
        "source_db": str(db.resolve()),
        "ticks_per_second_assumed": ticks_per_second,
        "collection_seconds_from_tsc": collection_seconds,
        "expected_collection_seconds": expected_collection_seconds,
        "discarded_tasks_with_invalid_timestamps": len(bad_timestamps),
        "complete_marker_sequences": len(sequences),
        "dropped_initial_sequences": drop_first,
        "dropped_trailing_sequences": drop_last,
        "incomplete_sequences": incomplete,
        "invalid_sequences": invalid,
        "usable_sequences": len(frames),
        "marker_queues": dict(Counter(str(frame["marker_queue"])
                                     for frame in frames)),
        "median_marker_span_ms": statistics.median(
            frame["marker_span_ms"] for frame in frames),
        "median_intervals_sum_ms": statistics.median(
            sum(part["elapsed_ms"] for part in frame["groups"])
            for frame in frames),
        "group_summary": summary,
        "frames": frames,
        "limits": ("Marker kernels add GPU scheduling and are absent from production; "
                   "VTune reported some unreliable GPU scheduler timestamps. "
                   "These intervals are exploratory until sequence consistency, "
                   "frozen output hash and whole-frame marker overhead are checked."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("db", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markers", help="comma-separated marker kernel names")
    parser.add_argument("--groups", help="comma-separated group labels")
    parser.add_argument("--drop-first", type=int, default=0)
    parser.add_argument("--drop-last", type=int, default=0)
    parser.add_argument("--ticks-per-second", type=float, default=10_000_000_000)
    parser.add_argument("--expected-collection-seconds", type=float)
    args = parser.parse_args()
    groups = tuple(args.groups.split(",")) if args.groups else DEFAULT_GROUPS
    markers = (tuple(args.markers.split(",")) if args.markers else
               tuple(f"nr_stage_{i:02d}" for i in range(len(groups) + 1)))
    report = summarize(args.db, markers, groups, args.drop_first, args.drop_last,
                       args.ticks_per_second, args.expected_collection_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "frames"},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
