"""Compare aligned, decoded C32 ablation videos without loading all frames."""

import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(r"D:\Codex-NR-Experiments\nr-b580\c32-activation-unrounded-v1-20260927\face540-review")
FFMPEG = Path(r"E:\ComfyUI-aki-v3-IntelArc_20260722\xess-tools\work\d3d12-media-pipeline\deps\ffmpeg-lgpl-shared-9.0\ffmpeg-n9.0-latest-win64-lgpl-shared-9.0\bin\ffmpeg.exe")
WIDTH, HEIGHT, FPS = 960, 540, 24
FRAME_BYTES = WIDTH * HEIGHT * 3
REGIONS = {"full": (slice(None), slice(None)), "face": (slice(18, 450), slice(284, 716))}


def decoder(path):
    return subprocess.Popen(
        [str(FFMPEG), "-v", "error", "-i", str(path), "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def read_exact(stream, count):
    data = bytearray()
    while len(data) < count:
        chunk = stream.read(count - len(data))
        if not chunk:
            break
        data.extend(chunk)
    return data


def main():
    procs = [decoder(ROOT / "baseline.mp4"), decoder(ROOT / "candidate.mp4")]
    result = {key: {"absolute_sum": 0, "squared_sum": 0, "channel_count": 0,
                    "above_1": 0, "above_2": 0, "above_4": 0, "above_8": 0,
                    "per_frame_mae": [], "temporal_delta_mae": [],
                    "max_channel_error": 0} for key in REGIONS}
    prev = None
    frames = 0
    try:
        while True:
            chunks = [read_exact(proc.stdout, FRAME_BYTES) for proc in procs]
            if not chunks[0] and not chunks[1]:
                break
            if any(len(chunk) != FRAME_BYTES for chunk in chunks):
                raise RuntimeError(f"Unmatched or partial decoded frame at {frames}: {[len(x) for x in chunks]}")
            arrays = [np.frombuffer(chunk, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3).astype(np.int16)
                      for chunk in chunks]
            for name, (ys, xs) in REGIONS.items():
                a, b = arrays[0][ys, xs], arrays[1][ys, xs]
                diff = np.abs(a - b)
                item = result[name]
                item["absolute_sum"] += int(diff.sum(dtype=np.int64))
                item["squared_sum"] += int(np.square(diff.astype(np.int32)).sum(dtype=np.int64))
                item["channel_count"] += diff.size
                item["max_channel_error"] = max(item["max_channel_error"], int(diff.max()))
                for threshold in (1, 2, 4, 8):
                    item[f"above_{threshold}"] += int(np.count_nonzero(diff > threshold))
                item["per_frame_mae"].append(float(diff.mean()))
                if prev is not None:
                    temporal = np.abs((a - prev[0][ys, xs]) - (b - prev[1][ys, xs]))
                    item["temporal_delta_mae"].append(float(temporal.mean()))
            prev = arrays
            frames += 1
    finally:
        for proc in procs:
            if proc.stdout:
                proc.stdout.close()
            err = proc.stderr.read().decode("utf-8", "replace") if proc.stderr else ""
            code = proc.wait()
            if code:
                raise RuntimeError(f"ffmpeg exited {code}: {err}")

    for item in result.values():
        n = item.pop("channel_count")
        abs_sum = item.pop("absolute_sum")
        squared_sum = item.pop("squared_sum")
        item["mae_levels"] = abs_sum / n
        item["rmse_levels"] = (squared_sum / n) ** 0.5
        item["psnr_db"] = 20 * np.log10(255 / item["rmse_levels"])
        for threshold in (1, 2, 4, 8):
            item[f"channels_above_{threshold}_percent"] = 100 * item.pop(f"above_{threshold}") / n
        per_frame = np.asarray(item.pop("per_frame_mae"))
        temporal = np.asarray(item.pop("temporal_delta_mae"))
        item["per_frame_mae_median"] = float(np.median(per_frame))
        item["per_frame_mae_p95"] = float(np.percentile(per_frame, 95))
        item["per_frame_mae_max"] = float(per_frame.max())
        item["per_frame_mae_worst_index"] = int(np.argmax(per_frame))
        item["per_frame_mae_worst_second"] = int(np.argmax(per_frame)) / FPS
        item["temporal_delta_mae_median"] = float(np.median(temporal))
        item["temporal_delta_mae_p95"] = float(np.percentile(temporal, 95))
        item["temporal_delta_mae_max"] = float(temporal.max())
        item["temporal_delta_mae_worst_frame"] = int(np.argmax(temporal)) + 1
    output = {"basis": "decoded H.264 RGB24; separate lossy encodes, not raw model tensors",
              "frames": frames, "size": [WIDTH, HEIGHT], "fps": FPS, "regions": result}
    path = ROOT / "pixel-diff-decoded-v1.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
