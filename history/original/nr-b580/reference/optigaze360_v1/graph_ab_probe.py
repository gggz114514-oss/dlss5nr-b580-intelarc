"""隔离定位：同一低分辨率几何，只切换 NR 主体的 XPU 图重放。"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec240 = importlib.util.spec_from_file_location(
    "isolated_run240", HERE.parent / "optigaze240_v1" / "run_video.py")
base240 = importlib.util.module_from_spec(spec240)
spec240.loader.exec_module(base240)

DATA = Path("D:/Codex-NR-Experiments/nr-b580/optigaze-geometry-graph-ab-v1")


def run_one(geometry, graph, frames):
    if geometry == 240:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "isolated_adapter240", HERE.parent / "optigaze240_v1" / "adapter.py")
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
        arm = "low240"
        initialize = base240.initialize
    else:
        import adapter
        arm = "low360"
        import importlib.util
        spec = importlib.util.spec_from_file_location("isolated_run360", HERE / "run_video.py")
        base360 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(base360)
        initialize = base360.initialize

    torch, session = initialize(arm)
    measurements, hashes = [], []
    try:
        warmup = session.warmup(frames=3)
        manifest = base240.inputs()
        for item in manifest["frames"][:frames]:
            rgb_np, motion_np = base240.capture_frame(item)
            rgb = torch.from_numpy(rgb_np).to("xpu")
            motion = torch.from_numpy(motion_np).to("xpu")
            low_rgb, low_motion = adapter.prepare(rgb, motion)
            torch.xpu.synchronize()
            t0 = time.perf_counter()
            with session._installed(), torch.inference_mode(), session._use_arithmetic_backend("triton"):
                if graph:
                    with session._stack.graph.installed():
                        value = session._stack.model(low_rgb, low_motion,
                                                     reset=bool(item["reset"])).float()
                else:
                    value = session._stack.model(low_rgb, low_motion,
                                                 reset=bool(item["reset"])).float()
                torch.xpu.synchronize()
            measurements.append((time.perf_counter() - t0) * 1000)
            hashes.append(hashlib.sha256(value.contiguous().cpu().numpy().tobytes()).hexdigest())
        stable = measurements[4:] if len(measurements) > 4 else measurements
        return {"geometry": geometry, "graph": graph, "frames": frames,
                "warmup": warmup, "median_ms": statistics.median(stable),
                "mean_ms": statistics.mean(stable), "samples_ms": measurements,
                "output_sha256": hashes,
                "graph_entries": session._stack.graph.metadata() if graph else []}
    finally:
        session.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("geometry", type=int, choices=(240, 360))
    parser.add_argument("--frames", type=int, default=12)
    args = parser.parse_args()
    if not 2 <= args.frames <= 243:
        parser.error("frames must be 2..243")
    DATA.mkdir(parents=True, exist_ok=True)
    eager = run_one(args.geometry, False, args.frames)
    captured = run_one(args.geometry, True, args.frames)
    equal = eager["output_sha256"] == captured["output_sha256"]
    result = {"passed": equal, "scope": "same geometry, eager vs graph model body",
              "eager": eager, "graph": captured,
              "speed_ratio": eager["median_ms"] / captured["median_ms"]}
    target = DATA / f"geometry{args.geometry}-frames{args.frames}.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": equal, "eager_ms": eager["median_ms"],
                      "graph_ms": captured["median_ms"], "report": str(target)},
                     ensure_ascii=False), flush=True)
    if not equal:
        raise RuntimeError("Graph changed NR output; reject speed conclusion")


if __name__ == "__main__":
    main()
