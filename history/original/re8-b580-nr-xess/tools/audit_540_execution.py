"""Record the installed 540p model's capture-time dispatches without timing them.

The observer never changes a kernel argument or model operation. A graph replay
does not execute Python, so the capture body is inventoried alongside the first
replay. This is a dispatch/shape audit, not a GPU utilisation benchmark.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback


def stage_from_stack():
    frame = sys._getframe(1)
    while frame is not None:
        if frame.f_code.co_name == "forward_front" and frame.f_code.co_filename.endswith("capture_body_v1.py"):
            line = frame.f_lineno
            if line <= 53:
                return "pre"
            if line <= 57:
                return f"encoder_C{frame.f_locals.get('c', '?')}"
            if line <= 61:
                return "encoder_C512"
            if line == 62:
                return "ViT"
            if line == 63:
                return "decoder_input"
            if line == 64:
                return "decoder_C512"
            if line <= 68:
                return f"decoder_C{frame.f_locals.get('c', '?')}"
            return "post"
        frame = frame.f_back
    return "outside_body"


def run(root: Path, report: Path) -> int:
    root = root.resolve()
    report = report.resolve()
    if report.drive.casefold() != "d:":
        raise ValueError("Audit output must be on D:")
    if not (root / "python/python313.dll").is_file():
        raise FileNotFoundError("Expected the portable RE8 NR runtime")
    sys.path[:0] = [str(root / "game"), str(root), str(root / "modules"),
                    str(root / "fast/backend")]
    from runtime_environment import isolate
    hold = isolate()
    from fast_cached_runtime_v1 import bootstrap, DiskOnly
    bootstrap()
    import torch
    from triton.compiler.compiler import CompiledKernel
    import capture_body_v1 as body
    from current_dense_tiled_provider_v1 import CurrentDenseTiledMatrices
    from strided_batched_v2 import StridedMatrices
    from nr_backend.controlled_temporal import NRControls
    from nr_game_fullsize import FullsizeGameModes

    config = json.loads((root / "data/product-v1/local-runtime-v1.json").read_text(encoding="utf-8"))
    original_body = body.forward_front
    original_launch = CompiledKernel.launch_metadata
    original_dense = CurrentDenseTiledMatrices.dense
    original_batched = StridedMatrices.batched
    active_body = 0
    body_calls = 0
    launches = Counter()
    providers = Counter()
    frames = []
    modes = None

    def tagged_body(*args, **kwargs):
        nonlocal active_body, body_calls
        body_calls += 1
        previous, active_body = active_body, body_calls
        try:
            return original_body(*args, **kwargs)
        finally:
            active_body = previous

    def launch(kernel, grid, stream, *args):
        fn = getattr(getattr(kernel.src, "fn", None), "fn", None)
        name = f"{getattr(fn, '__module__', 'unknown')}.{getattr(fn, '__name__', kernel.name)}"
        launches[(active_body, stage_from_stack(), name)] += 1
        return original_launch(kernel, grid, stream, *args)

    def provider_event(kind, provider, a, w, initial, before):
        changed = tuple(sorted((key, value - before.get(key, 0))
                               for key, value in provider.calls.items()
                               if value > before.get(key, 0)))
        providers[(active_body, stage_from_stack(), kind, tuple(a.shape),
                   tuple(w.shape), initial is not None, changed)] += 1

    def dense(provider, a, w, *, chunk_k, initial=None, **kwargs):
        before = dict(provider.calls)
        result = original_dense(provider, a, w, chunk_k=chunk_k, initial=initial, **kwargs)
        provider_event(f"dense_K{chunk_k}", provider, a, w, initial, before)
        return result

    def batched(provider, a, w, *, initial=None, **kwargs):
        before = dict(provider.calls)
        result = original_batched(provider, a, w, initial=initial, **kwargs)
        provider_event("batched", provider, a, w, initial, before)
        return result

    body.forward_front = tagged_body
    CompiledKernel.launch_metadata = launch
    CurrentDenseTiledMatrices.dense = dense
    StridedMatrices.batched = batched
    result = {"passed": False, "started_utc": datetime.now(timezone.utc).isoformat(),
              "runtime": str(root), "resolution": [960, 540],
              "scope": "capture-time dispatch inventory; no timing"}
    try:
        with DiskOnly() as disk:
            color = torch.full((540, 960, 3), 0.5, dtype=torch.float32, device="xpu")
            motion = torch.full((540, 960, 2), 0.375, dtype=torch.float32, device="xpu")
            modes = FullsizeGameModes(root / "exact", root / "data/product-v1/profile-v1.json",
                                      config["profile_sha256"], controlled=True,
                                      graph_capture_policy="all")
            try:
                for frame_no in range(3):
                    output = modes.process(color, motion, height=540, reset=frame_no == 0,
                                           history_warp="fused", graph_replay=True,
                                           controls=NRControls())
                    torch.xpu.synchronize()
                    if output.shape != color.shape or not bool(torch.isfinite(output).all().item()):
                        raise RuntimeError("540p model returned an invalid output")
                    frames.append({"frame": frame_no + 1, "replayed": modes.last_graph_used,
                                   "sha256": hashlib.sha256(output.cpu().numpy().tobytes()).hexdigest()})
                graph = modes.session._stack.graph
                result["graph_entries"] = len(graph.entries)
                result["graph_replays"] = graph.replays
                result["provider_mode"] = modes.session._stack.provider.mode
            finally:
                modes.close()
                modes = None
            result["disk_only_hits"] = disk.hits
        result["frames"] = frames
        result["body_calls"] = body_calls
        result["launches"] = [dict(body_call=body_id, stage=stage, kernel=name, count=count)
                              for (body_id, stage, name), count in sorted(launches.items())]
        result["provider_calls"] = [dict(body_call=body_id, stage=stage, kind=kind,
                                         activation_shape=list(a), weight_shape=list(w),
                                         has_initial=initial, dispatch_delta=dict(changed), count=count)
                                    for (body_id, stage, kind, a, w, initial, changed), count
                                    in sorted(providers.items())]
        result["passed"] = bool(frames[-1]["replayed"] and body_calls >= 3)
        if not result["passed"]:
            result["error"] = "Capture and replay were not both observed"
    except BaseException as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()[-10000:]
    finally:
        body.forward_front = original_body
        CompiledKernel.launch_metadata = original_launch
        CurrentDenseTiledMatrices.dense = original_dense
        StridedMatrices.batched = original_batched
        if modes is not None:
            modes.close()
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: result.get(key) for key in ("passed", "body_calls", "graph_entries",
                         "graph_replays", "disk_only_hits", "error")}, ensure_ascii=False), flush=True)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.runtime, args.report))
