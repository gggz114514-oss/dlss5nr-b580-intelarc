"""Full-chain byte and ABBA timing gate for one C128 QKV producer.

The installed 480/540 structure combo remains active. Prepare the D: cache,
then benchmark in a fresh process with DiskOnly; no game files are changed.
"""
from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import traceback


def run(runtime: Path, cache: Path, report: Path, height: int,
        phase: str, candidate_kind: str) -> int:
    runtime, cache, report = (p.resolve() for p in (runtime, cache, report))
    if cache.drive.casefold() != "d:" or report.drive.casefold() != "d:":
        raise ValueError("Cache and report must stay on D:")
    sys.dont_write_bytecode = True
    project = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(runtime / "game"), str(runtime), str(runtime / "modules"),
                    str(runtime / "fast/backend"), str(project / "game")]
    from runtime_environment import isolate
    isolate()
    from fast_cached_runtime_v1 import bootstrap, DiskOnly
    bootstrap()
    from precompile_game_modes import fill_missing_cache
    import torch
    from triton import knobs
    from nr_backend.controlled_temporal import NRControls
    from nr_game_fullsize import FullsizeGameModes
    from check_c64_qkv_direct_pack_v1 import cache_directory
    if candidate_kind == "one":
        import c128_qkv_direct_pack_one_v1 as candidate
    else:
        import c128_qkv_direct_pack_all_v1 as candidate

    config = json.loads((runtime / "data/product-v1/local-runtime-v1.json")
                        .read_text(encoding="utf-8-sig"))
    combo_modes = tuple(config.get("structure_combo_modes", ()))
    if combo_modes != (480, 540) or height not in combo_modes:
        raise RuntimeError("Expected installed 480/540 structure combo")
    torch.manual_seed(3330 + height)
    scene = torch.rand((540, 960, 3), device="xpu", dtype=torch.float32)
    frames = [(scene + i * .0003).clamp(0, 1).contiguous() for i in range(8)]
    motion = torch.empty((540, 960, 2), device="xpu", dtype=torch.float32)
    motion[..., 0], motion[..., 1] = .375, -.25
    torch.xpu.synchronize()
    result = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(),
              "height": height, "phase": phase, "candidate_kind": candidate_kind,
              "combo_modes": combo_modes,
              "cache": str(cache), "segments": []}
    arms = ("baseline", "candidate") if phase == "prepare" else (
        "baseline", "candidate", "candidate", "baseline")
    try:
        for order, arm in enumerate(arms):
            modes = FullsizeGameModes(runtime / "exact",
                                      runtime / "data/product-v1/profile-v1.json",
                                      config["profile_sha256"], controlled=True,
                                      graph_capture_policy="all", combo_modes=combo_modes)
            row = {"order": order, "arm": arm, "rgb_hashes": [], "routes": [],
                   "combo_capture_gates": [], "times_ms": []}
            try:
                cache_scope = DiskOnly() if phase == "benchmark" else fill_missing_cache()
                with cache_directory(cache, knobs), cache_scope as cache_state:
                    modes.select(height, (540, 960))
                    stack = modes.session._stack
                    scope = (candidate.installed(stack.window_blocks, stack.model,
                                                 height=height)
                             if arm == "candidate" else nullcontext(None))
                    with scope as calls:
                        total = 3 if phase == "prepare" else 143
                        for i in range(total):
                            start = time.perf_counter_ns()
                            rgb = modes.process(frames[i % 8], motion, height=height,
                                                reset=i == 0, history_warp="fused",
                                                graph_replay=True, controls=NRControls())
                            torch.xpu.synchronize()
                            route = modes.last_combo_frame_route
                            if i < 3:
                                row["routes"].append(route)
                                row["rgb_hashes"].append(hashlib.sha256(
                                    rgb.detach().contiguous().cpu().numpy().tobytes()).hexdigest())
                                if not bool(torch.isfinite(rgb).all().item()):
                                    raise RuntimeError("Nonfinite RGB frame")
                            if route == "capture":
                                gate = modes.last_combo_capture_gate
                                if gate is None or not all(gate.values()):
                                    raise RuntimeError("Installed structure combo capture gate failed")
                                row["combo_capture_gates"].append(dict(gate))
                            if phase == "benchmark" and i >= 23:
                                if route != "replay" or not modes.last_graph_used:
                                    raise RuntimeError("Timed frame did not replay graph")
                                row["times_ms"].append((time.perf_counter_ns() - start) * 1e-6)
                        row["candidate_capture_calls"] = (
                            None if calls is None else calls["direct_pack"])
                        row["candidate_by_block"] = (
                            None if calls is None else dict(calls.get("by_block", {})))
                        if arm == "candidate" and row["candidate_capture_calls"] < 2:
                            raise RuntimeError("C128 target missed eager/capture frames")
                        if (arm == "candidate" and candidate_kind == "all" and
                                (len(row["candidate_by_block"]) != 12 or
                                 not all(n > 0 for n in row["candidate_by_block"].values()))):
                            raise RuntimeError("A C128 block missed eager/capture frames")
                        if row["routes"][-1] != "replay" or len(row["combo_capture_gates"]) != 2:
                            raise RuntimeError("Expected two graph captures followed by replay")
                    row["disk_only_hits"] = getattr(cache_state, "hits", None)
                    row["compilation"] = (dict(cache_state) if phase == "prepare" else None)
                if row["times_ms"]:
                    ordered = sorted(row["times_ms"])
                    row["median_ms"] = statistics.median(ordered)
                    row["p95_ms"] = ordered[113]
                result["segments"].append(row)
                print(json.dumps({"order": order, "arm": arm,
                                  "median_ms": row.get("median_ms"),
                                  "capture_calls": row["candidate_capture_calls"]}),
                      flush=True)
            finally:
                modes.close()
        if len({tuple(row["rgb_hashes"]) for row in result["segments"]}) != 1:
            raise RuntimeError("Continuous-history RGB bytes differ across arms")
        result["passed"] = True
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-12000:]
    finally:
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "error": result.get("error")},
                     ensure_ascii=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--height", type=int, choices=(480, 540), required=True)
    parser.add_argument("--phase", choices=("prepare", "benchmark"), required=True)
    parser.add_argument("--candidate", choices=("one", "all"), default="one")
    args = parser.parse_args()
    raise SystemExit(run(args.runtime, args.cache, args.report, args.height,
                         args.phase, args.candidate))
