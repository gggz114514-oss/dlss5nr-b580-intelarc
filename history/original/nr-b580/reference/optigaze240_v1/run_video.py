"""Isolated 243-frame face-video comparison; no product or exact-line changes.

Run each arm in a separate process because the full-size geometry adapter uses a
process-global shape.  All source frames come from the same frozen capture.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path("E:/ComfyUI-aki-v3-IntelArc_20260722")
REF = ROOT / "nr-b580/reference"
HERE = Path(__file__).resolve().parent
DATA = Path("D:/Codex-NR-Experiments/nr-b580/optigaze240-v1")
CACHE = Path("D:/Codex-NR-Experiments/nr-b580/optigaze240-v1-smoke/cache/triton-cache-c32-triton38-v1")
CAPTURE = Path("D:/Codex-NR-Experiments/nr-b580/fourway-face-review-v1/exact/validation.json")
CONFIG = Path("D:/Codex-NR-Experiments/nr-b580/product-v1/local-runtime-v1.json")
VIDEO = REF / "inputs/visual-qa-01/clip480.mp4"
LABELS = {
    "source": "Input 480p (frozen RGB)",
    "full": "Full-size fast NR 480p",
    "nr256": "Existing NR256 (active 256x142)",
    "plain240": "NR 240p + plain residual upscale",
    "guide240": "NR 240p + source-guided reconstruction",
    "temporal240": "NR 240p + guided + temporal reprojection",
}
ARMS = list(LABELS)
WIDTH, HEIGHT, FPS = 864, 480, 24
_DLL_HANDLES = []


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def inputs():
    cap = json.loads(CAPTURE.read_text(encoding="utf-8-sig"))
    if not cap.get("passed") or len(cap["frames"]) != 243:
        raise RuntimeError("The frozen 243-frame capture is not available")
    if digest(VIDEO) != cap["source_sha256"]:
        raise RuntimeError("Face-video source hash differs from frozen capture")
    for item in cap["frames"]:
        if digest(item["file"]) != item["sha256"]:
            raise RuntimeError(f"Capture frame changed: {item['file']}")
    return cap


def ffmpeg():
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def encoder(path, width=WIDTH, height=HEIGHT, *, review=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
               "-framerate", str(FPS), "-i", "pipe:0"]
    if review:
        command += ["-i", str(VIDEO), "-map", "0:v:0", "-map", "1:a:0",
                    "-threads", "4", "-c:v", "libx264", "-preset", "fast", "-crf", "13",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart"]
    else:
        command += ["-an", "-threads", "4", "-c:v", "libx264rgb", "-preset", "ultrafast", "-qp", "0"]
    command.append(str(path))
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def to_rgb8(array):
    import numpy as np
    if not np.isfinite(array).all():
        raise RuntimeError("Non-finite output pixels")
    return np.clip(np.rint(np.clip(array, 0, 1) * 255), 0, 255).astype(np.uint8)


def capture_frame(item):
    import numpy as np
    with np.load(item["file"], allow_pickle=False) as z:
        rgb = np.ascontiguousarray(z["rgb"].astype(np.float32))
        motion = np.ascontiguousarray(z["motion"].astype(np.float32))
    if rgb.shape != (HEIGHT, WIDTH, 3) or motion.shape != (HEIGHT, WIDTH, 2):
        raise RuntimeError(f"Wrong source geometry: {rgb.shape}, {motion.shape}")
    return rgb, motion


def initialize(arm):
    # Use the pinned runtime, independently from ComfyUI's loaded Python state.
    rows = REF / "fullsize_rows_v1"
    sys.path[:0] = [str(rows), str(HERE)]
    import rows_fscache_guard_v1
    rows_fscache_guard_v1.install()
    sys.path.insert(0, str(ROOT / "nr-b580-int8/product/comfy"))
    from runtime_environment import isolate
    _DLL_HANDLES.extend(isolate()[:2])
    for directory in (ROOT / "xess-tools/XeSS-R4-Offline/runtime/bin",
                      ROOT / "xess-tools/venv/Lib/site-packages/openvino/libs",
                      ROOT / "ComfyUI-aki-v3-IntelArc/python/Library/bin",
                      Path("D:/Codex-NR-Experiments/nr-b580/product-worker/native-v4")):
        if not directory.is_dir():
            raise FileNotFoundError(f"Required isolated runtime directory missing: {directory}")
        _DLL_HANDLES.append(os.add_dll_directory(str(directory)))
    sys.path.insert(0, str(ROOT / "nr-b580-int8/product/precompile"))
    from fast_cached_runtime_v1 import bootstrap
    bootstrap()
    # Cached artifacts live on D:. New 240p shapes compile once here.
    if not CACHE.is_dir():
        raise FileNotFoundError(f"Isolated verified Triton cache is missing: {CACHE}")
    os.environ["TRITON_CACHE_DIR"] = str(CACHE)
    if arm == "full":
        sys.path.insert(0, str(REF / "fullchain_timing_v1"))
        from fullsize_session_v1 import FullsizeSession
        cls = FullsizeSession
    elif arm == "nr256":
        from nr_runtime_v1 import Session
        cls = Session
    else:
        sys.path.insert(0, str(REF / "nr_geom_re4_v1"))
        from re4_session_v1 import open_session
        cls = open_session
    import torch
    config = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    args = (ROOT / "nr-b580", config["profile_path"], config["profile_sha256"])
    if arm == "nr256":
        session = cls.create(*args)
    elif arm == "full":
        session = cls(*args)
    else:
        session = cls(*args, geometry=(240, 432), padding=(256, 512))
    return torch, session


def summarize(samples):
    if not samples:
        return None
    return {"count": len(samples), "median_ms": round(statistics.median(samples), 3),
            "mean_ms": round(statistics.mean(samples), 3),
            "p10_ms": round(sorted(samples)[int(0.1 * (len(samples) - 1))], 3),
            "p90_ms": round(sorted(samples)[int(0.9 * (len(samples) - 1))], 3)}


def run_arm(arm):
    import numpy as np
    manifest = inputs()
    arm_dir = DATA / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    outputs = [arm] if arm in ("source", "full", "nr256") else ["plain240", "guide240", "temporal240"]
    writers = {name: encoder(DATA / "video" / f"{name}.mkv") for name in outputs}
    report = {"arm": arm, "source_sha256": manifest["source_sha256"], "frames": 0,
              "derived_geometry": arm == "low240", "timings": {}, "videos": {}}
    torch = session = None
    times = {name: [] for name in ("prepare", "model", "reconstruct", "total")}
    prev_base = prev_edit = None
    try:
        if arm != "source":
            torch, session = initialize(arm)
            if arm == "low240":
                import adapter
                import torch.nn.functional as F
            if arm == "nr256":
                t_warm = time.perf_counter()
                zero_rgb = torch.zeros((HEIGHT, WIDTH, 3), device="xpu", dtype=torch.float32)
                zero_motion = torch.zeros((HEIGHT, WIDTH, 2), device="xpu", dtype=torch.float32)
                session.process(zero_rgb, zero_motion, reset=True)
                session.process(zero_rgb, zero_motion, reset=False)
                session.reset()
                report["warmup"] = {"frames": 2, "seconds": round(time.perf_counter() - t_warm, 3)}
            else:
                report["warmup"] = session.warmup(frames=3)
        for item in manifest["frames"]:
            rgb_np, motion_np = capture_frame(item)
            if arm == "source":
                frames = {arm: to_rgb8(rgb_np)}
            else:
                rgb = torch.from_numpy(rgb_np).to("xpu")
                motion = torch.from_numpy(motion_np).to("xpu")
                torch.xpu.synchronize()
                t0 = time.perf_counter()
                with torch.inference_mode():
                    if arm == "low240":
                        low_rgb, low_motion = adapter.prepare(rgb, motion)
                        torch.xpu.synchronize()
                        t1 = time.perf_counter()
                        low_model = session.process(low_rgb, low_motion, reset=bool(item["reset"])).color.contiguous()
                        t2 = time.perf_counter()
                        residual = adapter.residual(low_rgb, low_model)
                        scaled = F.interpolate(residual.permute(2, 0, 1)[None], size=(HEIGHT, WIDTH),
                                               mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
                        plain = (rgb + scaled).clamp(0, 1)
                        guided = adapter.guided(rgb, low_rgb, residual)
                        edited = (residual if prev_edit is None else
                                  adapter.temporal(low_rgb, residual, low_motion, prev_base, prev_edit))
                        guided_t = adapter.guided(rgb, low_rgb, edited)
                        prev_base, prev_edit = low_rgb, edited
                        torch.xpu.synchronize()
                        t3 = time.perf_counter()
                        frames = {"plain240": to_rgb8(plain.cpu().numpy()),
                                  "guide240": to_rgb8(guided.cpu().numpy()),
                                  "temporal240": to_rgb8(guided_t.cpu().numpy())}
                        times["prepare"].append((t1 - t0) * 1000)
                        times["model"].append((t2 - t1) * 1000)
                        times["reconstruct"].append((t3 - t2) * 1000)
                        times["total"].append((t3 - t0) * 1000)
                    else:
                        out = session.process(rgb, motion, reset=bool(item["reset"])).color
                        torch.xpu.synchronize()
                        t1 = time.perf_counter()
                        frames = {arm: to_rgb8(out.cpu().numpy())}
                        times["model"].append((t1 - t0) * 1000)
                        times["total"].append((t1 - t0) * 1000)
            for name, frame in frames.items():
                writers[name].stdin.write(np.ascontiguousarray(frame).tobytes())
            report["frames"] += 1
            if report["frames"] in (1, 60, 120, 180, 243):
                print(f"{arm}: {report['frames']}/243", flush=True)
        for name, writer in writers.items():
            writer.stdin.close()
            if writer.wait() != 0:
                raise RuntimeError(f"ffmpeg encode failed: {name}")
            report["videos"][name] = str(DATA / "video" / f"{name}.mkv")
        report["timings"] = {key: summarize(values[4:]) for key, values in times.items()}
        report["passed"] = report["frames"] == 243
        (arm_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        if session is not None:
            session.close()
        for writer in writers.values():
            if writer.poll() is None:
                writer.kill()
                writer.wait()


def render():
    from PIL import Image, ImageDraw, ImageFont
    import numpy as np
    inputs()
    for arm in ("source", "full", "nr256", "low240"):
        report = json.loads((DATA / arm / "report.json").read_text(encoding="utf-8"))
        if not report.get("passed") or report["frames"] != 243:
            raise RuntimeError(f"Incomplete arm: {arm}")
    sources = {}
    for name in ARMS:
        path = DATA / "video" / f"{name}.mkv"
        command = [ffmpeg(), "-hide_banner", "-loglevel", "error", "-i", str(path),
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
        sources[name] = subprocess.Popen(command, stdout=subprocess.PIPE)
    strip = 36
    canvas_w, canvas_h = WIDTH * 3, (HEIGHT + strip) * 2
    out = DATA / "review_480p_240p_6way.mp4"
    writer = encoder(out, canvas_w, canvas_h, review=True)
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 22)
    try:
        for i in range(243):
            canvas = Image.new("RGB", (canvas_w, canvas_h), (16, 16, 16))
            draw = ImageDraw.Draw(canvas)
            for slot, name in enumerate(ARMS):
                raw = sources[name].stdout.read(WIDTH * HEIGHT * 3)
                if len(raw) != WIDTH * HEIGHT * 3:
                    raise RuntimeError(f"Short decode: {name} frame {i}")
                x = (slot % 3) * WIDTH
                y = (slot // 3) * (HEIGHT + strip)
                draw.text((x + 12, y + 5), LABELS[name], font=font, fill="white")
                frame = Image.fromarray(np.frombuffer(raw, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3))
                canvas.paste(frame, (x, y + strip))
            writer.stdin.write(canvas.tobytes())
        writer.stdin.close()
        if writer.wait() != 0:
            raise RuntimeError("Review encode failed")
        for name, source in sources.items():
            if source.wait() != 0:
                raise RuntimeError(f"Review decode failed: {name}")
        print(f"Review video: {out}", flush=True)
    finally:
        if writer.poll() is None:
            writer.kill()
        for source in sources.values():
            if source.poll() is None:
                source.kill()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("arm", choices=("source", "full", "nr256", "low240", "render"))
    args = parser.parse_args()
    DATA.mkdir(parents=True, exist_ok=True)
    if args.arm == "render":
        render()
    else:
        run_arm(args.arm)


if __name__ == "__main__":
    main()
