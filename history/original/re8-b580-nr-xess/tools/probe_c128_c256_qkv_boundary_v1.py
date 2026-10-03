"""Measure the installed C128/C256 QKV matrix-to-window boundary on B580.

Uses the game's actual padded shapes, weights and cached kernels. This is a
local graph-replay stage cost, not an estimate of a future fusion's gain.
It neither changes the game nor compiles kernels during the timing phase.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import traceback


def run(runtime: Path, cache: Path, report: Path, height: int) -> int:
    runtime, cache, report = (path.resolve() for path in (runtime, cache, report))
    if cache.drive.casefold() != "d:" or report.drive.casefold() != "d:":
        raise ValueError("Probe cache and report must stay on D:")
    if not runtime.is_dir() or not cache.is_dir():
        raise FileNotFoundError("Installed game runtime or existing cache missing")
    sys.dont_write_bytecode = True
    sys.path[:0] = [str(runtime / "game"), str(runtime), str(runtime / "modules"),
                    str(runtime / "fast/backend")]
    from runtime_environment import isolate
    isolate()
    from fast_cached_runtime_v1 import bootstrap, DiskOnly
    bootstrap()
    import torch
    from triton import knobs
    import nr_backend.multihead_block as blocks
    from nr_backend.controlled_temporal import NRControls
    from nr_backend.execution import use_arithmetic_backend
    from fused_qkv_pack_native_half_v1 import forward as pack
    from nr_game_fullsize import FullsizeGameModes
    from check_c64_qkv_direct_pack_v1 import cache_directory

    config = json.loads((runtime / "data/product-v1/local-runtime-v1.json")
                        .read_text(encoding="utf-8"))
    output = {"passed": False, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "height": height, "cache": str(cache), "families": {}}
    modes = FullsizeGameModes(runtime / "exact",
                              runtime / "data/product-v1/profile-v1.json",
                              config["profile_sha256"], controlled=True,
                              graph_capture_policy="all")
    try:
        with cache_directory(cache, knobs), DiskOnly() as disk:
            modes.select(height, (540, 960))
            # One real chain capture provides the actual padded per-block shapes.
            torch.manual_seed(20260927)
            frame = torch.rand((540, 960, 3), device="xpu", dtype=torch.float32)
            motion = torch.zeros((540, 960, 2), device="xpu", dtype=torch.float32)
            modes.process(frame, motion, height=height, reset=True,
                          history_warp="fused", graph_replay=True,
                          controls=NRControls())
            torch.xpu.synchronize()
            stack = modes.session._stack
            named = dict(stack.model.named_modules())
            rows = defaultdict(dict)
            for call in stack.window_blocks.calls:
                shape = tuple(call["padded_shape"])
                channels = shape[-1]
                if channels not in (128, 256):
                    continue
                name = call["module"]
                block = named[name]
                rows[channels][name] = (name, shape, block.attention)
            output["shape_inventory"] = {str(c): [
                {"module": name, "padded_shape": list(shape),
                 "heads": att.heads, "weight_shape": list(att.qkv.shape)}
                for name, shape, att in group.values()]
                for c, group in rows.items()}
            if len(rows[128]) != 12 or len(rows[256]) != 16:
                raise RuntimeError(f"Unexpected C128/C256 block inventory: "
                                   f"{len(rows[128])}/{len(rows[256])}")

            for channels in (128, 256):
                cases = []
                for name, shape, att in rows[channels].values():
                    features = torch.randn(shape, device="xpu", dtype=torch.float16) * .125
                    cases.append((name, shape, att, blocks.quantize_fp8(features)))
                torch.xpu.synchronize()

                def compute(kind: str):
                    retained = []
                    for _name, shape, att, features in cases:
                        h, w, _ = shape
                        if kind == "pack":
                            z = precomputed[_name]
                        else:
                            z = blocks.sm89_f16_dot(features, att.qkv, chunk_k=16)
                        if kind != "matmul":
                            qkv, _ = pack(z.reshape(h, w, att.heads, 3, 32),
                                          att.scale, att.pixel_order,
                                          rows=stack.window_blocks.layout.rows)
                            retained.extend(qkv)
                        retained.append(z)
                    return retained

                precomputed = {}
                with modes.session._installed(), torch.inference_mode(), \
                        use_arithmetic_backend("triton"):
                    for name, shape, att, features in cases:
                        precomputed[name] = blocks.sm89_f16_dot(
                            features, att.qkv, chunk_k=16)
                    torch.xpu.synchronize()
                    family = {}
                    for kind in ("pair", "matmul", "pack"):
                        stream = torch.xpu.Stream()
                        graph = torch.xpu.XPUGraph()
                        with stream:
                            retained = compute(kind)
                        torch.xpu.synchronize()
                        with torch.xpu.graph(graph, stream=stream):
                            retained = compute(kind)
                        torch.xpu.synchronize()
                        for _ in range(20):
                            with stream:
                                graph.replay()
                            torch.xpu.synchronize()
                        times = []
                        for _ in range(120):
                            started = time.perf_counter_ns()
                            with stream:
                                graph.replay()
                            torch.xpu.synchronize()
                            times.append((time.perf_counter_ns() - started) * 1e-6)
                        ordered = sorted(times)
                        family[kind] = {"median_ms": statistics.median(times),
                                        "p95_ms": ordered[113], "min_ms": ordered[0],
                                        "max_ms": ordered[-1], "samples": len(times)}
                        del retained, graph, stream
                output["families"][str(channels)] = family
                print(json.dumps({"channels": channels, "stages": family}), flush=True)
            output["disk_only_hits"] = disk.hits
        output["passed"] = True
    except BaseException as exc:
        output["error"] = f"{type(exc).__name__}: {exc}"
        output["traceback"] = traceback.format_exc()[-12000:]
    finally:
        modes.close()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": output["passed"], "error": output.get("error")},
                     ensure_ascii=False), flush=True)
    return 0 if output["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--height", type=int, choices=(360, 480, 540), required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.runtime, args.cache, args.report, args.height))
