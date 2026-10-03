"""Prepare and pair-test the fast-only ordinary-C32 independent E4M3 ablation.

Both arms use the installed G runtime and its current 480/540 structure combo.
Run prepare first, then benchmark in a fresh process with the same D cache.
RGB hashes may differ; finite outputs, graph routes and candidate capture are
the correctness gates. No game or exact files are modified.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (540, 960)
ARMS = ("baseline", "candidate", "candidate", "baseline")


def on_d(path: Path) -> Path:
    resolved = path.resolve()
    base = Path("D:/Codex-NR-Experiments/nr-b580").resolve()
    if resolved.drive.casefold() != "d:" or not resolved.is_relative_to(base):
        raise ValueError(f"Experiment output must remain under {base}: {resolved}")
    return resolved


@contextmanager
def cache_directory(path, knobs):
    previous = (os.environ.get("TRITON_CACHE_DIR"), knobs.cache.dir)
    os.environ["TRITON_CACHE_DIR"] = str(path)
    knobs.cache.dir = str(path)
    try:
        yield
    finally:
        knobs.cache.dir = previous[1]
        if previous[0] is None:
            os.environ.pop("TRITON_CACHE_DIR", None)
        else:
            os.environ["TRITON_CACHE_DIR"] = previous[0]


def summary(values):
    ordered = sorted(values)
    mean = statistics.mean(ordered)
    return {"count": len(ordered), "median_ms": statistics.median(ordered),
            "p95_ms": ordered[math.ceil(len(ordered) * .95) - 1],
            "cv_percent": statistics.pstdev(ordered) / mean * 100 if mean else 0,
            "min_ms": ordered[0], "max_ms": ordered[-1]}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    runtime = args.runtime.resolve()
    cache, report = on_d(args.cache), on_d(args.report)
    if not (runtime / "fast/backend/nr_backend/c32_block.py").is_file():
        raise FileNotFoundError("Installed RE8 backend not found")
    if args.phase == "benchmark" and not cache.is_dir():
        raise FileNotFoundError("Prepare phase has not created the D cache")
    if args.phase == "prepare":
        cache.mkdir(parents=True, exist_ok=True)
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(runtime / "game"), str(runtime),
                    str(runtime / "modules"), str(runtime / "fast/backend")]
    sys.path.append(str(ROOT / "game"))
    from runtime_environment import isolate
    isolate()
    from fast_cached_runtime_v1 import DiskOnly, bootstrap
    bootstrap()
    from precompile_game_modes import fill_missing_cache
    import torch
    from triton import knobs
    from nr_backend.controlled_temporal import NRControls
    from nr_game_fullsize import FullsizeGameModes
    from c32_activation_unrounded_v1 import installed as unrounded_scope

    config = json.loads((runtime / "data/product-v1/local-runtime-v1.json")
                        .read_text(encoding="utf-8-sig"))
    combo_modes = tuple(config.get("structure_combo_modes", ()))
    if combo_modes != (480, 540):
        raise RuntimeError("Expected currently installed 480/540 structure combo")
    torch.manual_seed(1257 + args.height)
    scene = torch.rand((*SOURCE, 3), device="xpu", dtype=torch.float32)
    frames = [(scene + (i - 3.5) * .0003).clamp(0, 1).contiguous()
              for i in range(8)]
    motion = torch.empty((*SOURCE, 2), device="xpu", dtype=torch.float32)
    motion[..., 0], motion[..., 1] = .375, -.25
    torch.xpu.synchronize()
    result = {"passed": False, "phase": args.phase, "height": args.height,
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "runtime": str(runtime), "cache": str(cache),
              "runtime_c32_sha256": sha(runtime / "fast/backend/nr_backend/c32_block.py"),
              "candidate_sha256": sha(ROOT / "game/c32_activation_unrounded_v1.py"),
              "combo_modes": combo_modes, "pid": os.getpid(), "segments": []}
    arms = ARMS if args.phase == "benchmark" else ("baseline", "candidate")
    try:
        for order, arm in enumerate(arms):
            modes = FullsizeGameModes(runtime / "exact",
                                      runtime / "data/product-v1/profile-v1.json",
                                      config["profile_sha256"], controlled=True,
                                      graph_capture_policy="all",
                                      combo_modes=combo_modes)
            row = {"arm": arm, "order": order, "times_ms": [], "routes": [],
                   "candidate_calls": 0, "rgb_sha256": [], "finite": None,
                   "disk_only_hits": None}
            try:
                guard = DiskOnly() if args.phase == "benchmark" else fill_missing_cache()
                with cache_directory(cache, knobs), guard as cache_state:
                    modes.select(args.height, SOURCE)
                    scope = (unrounded_scope(modes.session._stack, height=args.height,
                                             runtime=runtime)
                             if arm == "candidate" else nullcontext(None))
                    with scope as calls:
                        total = (3 if args.phase == "prepare" else
                                 args.warmup + args.measured)
                        for i in range(total):
                            start = time.perf_counter_ns()
                            output = modes.process(frames[i % 8], motion,
                                                   height=args.height, reset=i == 0,
                                                   history_warp="fused", graph_replay=True,
                                                   controls=NRControls())
                            torch.xpu.synchronize()
                            route = modes.last_combo_frame_route
                            row["routes"].append(route)
                            if args.phase == "benchmark" and i >= args.warmup:
                                if route != "replay" or not modes.last_graph_used:
                                    raise RuntimeError("Timed frame did not replay captured graph")
                                row["times_ms"].append((time.perf_counter_ns() - start) * 1e-6)
                        if row["routes"][-1] != "replay":
                            raise RuntimeError("Mode did not reach graph replay")
                        row["candidate_calls"] = 0 if calls is None else calls["fusion_input"]
                        row["candidate_c32_blocks"] = None if calls is None else calls["by_block"]
                        if arm == "candidate" and (row["candidate_calls"] < 7 or
                                any(sum(v.values()) < 1 for v in calls["by_block"].values())):
                            raise RuntimeError("C32 family ablation did not hit all seven blocks in capture")
                        # Readback and checks are outside all timed frames.
                        row["finite"] = bool(torch.isfinite(output).all().item())
                        if not row["finite"]:
                            raise RuntimeError("Nonfinite fast NR output")
                        row["rgb_sha256"].append(hashlib.sha256(
                            output.detach().contiguous().cpu().numpy().tobytes()).hexdigest())
                    row["disk_only_hits"] = getattr(cache_state, "hits", None)
                if row["times_ms"]:
                    row["summary"] = summary(row["times_ms"])
                result["segments"].append(row)
            finally:
                modes.close()
        if args.phase == "benchmark":
            b = [v for row in result["segments"] if row["arm"] == "baseline"
                 for v in row["times_ms"]]
            c = [v for row in result["segments"] if row["arm"] == "candidate"
                 for v in row["times_ms"]]
            bs, cs = summary(b), summary(c)
            rows = result["segments"]
            result["aggregate"] = {"baseline": bs, "candidate": cs,
                                   "delta_median_ms": cs["median_ms"] - bs["median_ms"],
                                   "paired_deltas_ms": [
                                       rows[1]["summary"]["median_ms"] - rows[0]["summary"]["median_ms"],
                                       rows[2]["summary"]["median_ms"] - rows[3]["summary"]["median_ms"]]}
        result["passed"] = True
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-12000:]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "phase": args.phase,
                      "error": result.get("error"), "aggregate": result.get("aggregate")},
                     ensure_ascii=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "benchmark"))
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--height", type=int, choices=(480, 540), required=True)
    parser.add_argument("--warmup", type=int, default=16)
    parser.add_argument("--measured", type=int, default=120)
    args = parser.parse_args()
    if args.warmup < 4 or args.measured < 20:
        parser.error("Need at least 4 warmup and 20 measured frames")
    raise SystemExit(run(args))
