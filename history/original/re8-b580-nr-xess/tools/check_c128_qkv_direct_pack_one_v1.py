"""Byte-gate and pair-time one C128 QKV direct-write candidate.

Run --phase prepare once, then --phase benchmark in a fresh process. Only the
test cache/report on D: are writable; the game runtime is read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import traceback


def run(runtime: Path, cache: Path, report: Path, height: int, phase: str) -> int:
    runtime, cache, report = (path.resolve() for path in (runtime, cache, report))
    if cache.drive.casefold() != "d:" or report.drive.casefold() != "d:":
        raise ValueError("Cache and reports must stay on D:")
    sys.dont_write_bytecode = True
    project = Path(__file__).resolve().parents[1]
    sys.path[:0] = [str(runtime / "game"), str(runtime), str(runtime / "modules"),
                    str(runtime / "fast/backend"), str(project / "game")]
    from runtime_environment import isolate
    isolate()
    from fast_cached_runtime_v1 import bootstrap, DiskOnly
    bootstrap()
    import torch
    from triton import knobs
    import nr_backend.multihead_block as blocks
    from nr_backend.execution import use_arithmetic_backend
    from fused_qkv_pack_native_half_v1 import forward as prepare
    from nr_game_fullsize import FullsizeGameModes
    from check_c64_qkv_direct_pack_v1 import cache_directory
    import c128_qkv_direct_pack_one_v1 as candidate

    config = json.loads((runtime / "data/product-v1/local-runtime-v1.json")
                        .read_text(encoding="utf-8"))
    result = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(),
              "height": height, "phase": phase, "cache": str(cache), "segments": []}
    modes = FullsizeGameModes(runtime / "exact",
                              runtime / "data/product-v1/profile-v1.json",
                              config["profile_sha256"], controlled=True,
                              graph_capture_policy="all")
    try:
        with cache_directory(cache, knobs):
            modes.select(height, (540, 960))
            stack = modes.session._stack
            module = stack.model.encoder[2][1].attention
            h, w = candidate._SHAPES[height]
            torch.manual_seed(8400 + height)
            features = (torch.randn((h, w, 128), device="xpu",
                                    dtype=torch.float16) * .125).half()

            def digest(tensors):
                return [hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes())
                        .hexdigest() for t in tensors]

            def compute(arm, operand):
                if arm == "candidate":
                    return candidate.direct_pack(operand, module, height=height)
                z = blocks.sm89_f16_dot(operand, module.qkv, chunk_k=16)
                return prepare(z.reshape(h, w, module.heads, 3, 32),
                               module.scale, module.pixel_order,
                               rows=stack.window_blocks.layout.rows)[0]

            if phase == "benchmark":
                cache_scope = DiskOnly()
            else:
                # The installed runtime normalizes compiler options before its
                # DiskOnly lookup; prepare must use the identical cache key.
                from precompile_game_modes import fill_missing_cache
                cache_scope = fill_missing_cache()
            with cache_scope as disk, modes.session._installed(), \
                    torch.inference_mode(), use_arithmetic_backend("triton"):
                operand = blocks.quantize_fp8(features)
                torch.xpu.synchronize()
                reference = compute("baseline", operand)
                proposed = compute("candidate", operand)
                torch.xpu.synchronize()
                result["qkv_hashes"] = {"baseline": digest(reference),
                                        "candidate": digest(proposed)}
                result["finite"] = all(bool(torch.isfinite(x).all().item())
                                       for x in (*reference, *proposed))
                if (not result["finite"] or
                        result["qkv_hashes"]["baseline"] != result["qkv_hashes"]["candidate"]):
                    raise RuntimeError("C128 Q/K/V bytes or finite check failed")
                torch.xpu.synchronize()
                if phase == "benchmark":
                    for arm in ("baseline", "candidate", "candidate", "baseline"):
                        stream = torch.xpu.Stream()
                        graph = torch.xpu.XPUGraph()
                        with torch.xpu.graph(graph, stream=stream):
                            tensors = compute(arm, operand)
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
                        result["segments"].append({"arm": arm,
                                                   "median_ms": statistics.median(times),
                                                   "p95_ms": ordered[113],
                                                   "hashes": digest(tensors)})
                        del tensors, graph, stream
                    if len({tuple(row["hashes"]) for row in result["segments"]}) != 1:
                        raise RuntimeError("Q/K/V graph replay bytes changed")
                    result["disk_only_hits"] = disk.hits
                else:
                    result["compilation"] = dict(disk)
        result["passed"] = True
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-12000:]
    finally:
        modes.close()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": result["passed"], "error": result.get("error"),
                      "segments": result["segments"]}, ensure_ascii=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--height", type=int, choices=(360, 480, 540), required=True)
    parser.add_argument("--phase", choices=("prepare", "benchmark"), required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.runtime, args.cache, args.report, args.height, args.phase))
