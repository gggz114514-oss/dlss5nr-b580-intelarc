"""Offline visual smoke for a single captured pre-SR R11G11B10 game frame.

This does not claim temporal or in-game acceptance. It keeps scene-linear
highlights above 1 unchanged while the existing fast NR receives SDR RGB.
"""
import ctypes
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image, ImageDraw


CAPTURE = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\dlss720-color-20396.bin")
OUT = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\dlss720-fast-nr540-preview.png")
RUNTIME = Path(r"G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime")
WIDTH, HEIGHT = 1280, 720


def decode_channel(packed, shift, mantissa_bits):
    bits = (packed >> shift) & ((1 << (mantissa_bits + 5)) - 1)
    exponent = bits >> mantissa_bits
    mantissa = bits & ((1 << mantissa_bits) - 1)
    value = np.ldexp(np.ones(bits.shape, dtype=np.float32),
                     np.maximum(exponent.astype(np.int32), 1) - 15 - mantissa_bits)
    value *= mantissa + np.where(exponent == 0, 0, 1 << mantissa_bits)
    return value.astype(np.float32)


def to_preview(rgb):
    mapped = np.maximum(rgb, 0) / (1 + np.maximum(rgb, 0))
    encoded = np.where(mapped <= 0.0031308, mapped * 12.92,
                       1.055 * np.power(mapped, 1 / 2.4) - 0.055)
    return Image.fromarray(np.uint8(np.rint(np.clip(encoded, 0, 1) * 255)), "RGB")


def main():
    if CAPTURE.stat().st_size != WIDTH * HEIGHT * 4:
        raise ValueError("Capture size/row pitch changed; stop instead of guessing")
    packed = np.fromfile(CAPTURE, dtype="<u4").reshape(HEIGHT, WIDTH)
    original = np.stack((decode_channel(packed, 0, 6),
                         decode_channel(packed, 11, 6),
                         decode_channel(packed, 22, 5)), axis=-1)
    if not np.isfinite(original).all():
        raise ValueError("Nonfinite captured color")
    proxy = np.clip(original, 0, 1).astype(np.float32)

    runtime_bin = RUNTIME / "python" / "Library" / "bin"
    dll_dirs = [os.add_dll_directory(str(runtime_bin)),
                os.add_dll_directory(str(RUNTIME / "openvino-libs"))]
    os.environ["PATH"] = str(runtime_bin) + os.pathsep + os.environ.get("PATH", "")
    for key in ("ONEAPI_DEVICE_SELECTOR", "ONEAPI_ROOT", "TRITON_INTEL_SYCL_COMPILER",
                "TRITON_INTEL_DEVICE_EXTENSIONS"):
        os.environ.pop(key, None)
    sycl = ctypes.WinDLL(str(runtime_bin / "sycl9.dll"))
    sys.path[:0] = [str(RUNTIME), str(RUNTIME / "game"), str(RUNTIME / "modules"),
                    str(RUNTIME / "fast" / "backend")]
    from fast_cached_runtime_v1 import DiskOnly, bootstrap
    bootstrap()
    import rows_fscache_guard_v1 as guard
    guard.install()
    import torch
    from nr_backend.controlled_temporal import NRControls
    from nr_game_fullsize import FullsizeGameModes

    config = json.loads((RUNTIME / "data" / "product-v1" /
                         "local-runtime-v1.json").read_text(encoding="utf-8-sig"))
    modes = FullsizeGameModes(RUNTIME / "exact", RUNTIME / "data" /
                              "product-v1" / "profile-v1.json",
                              config["profile_sha256"], controlled=True,
                              graph_capture_policy="initial-360")
    try:
        with DiskOnly():
            color = torch.from_numpy(proxy.copy()).to("xpu")
            motion = torch.zeros((HEIGHT, WIDTH, 2), dtype=torch.float32, device="xpu")
            result = modes.process(color, motion, height=540, reset=True,
                                   history_warp="reference", graph_replay=False,
                                   controls=NRControls()).float().cpu().numpy()
    finally:
        modes.close()
    if result.shape != original.shape or not np.isfinite(result).all():
        raise ValueError("NR returned invalid RGB")

    # The model cannot accept HDR. This diagnostic candidate only applies its
    # SDR delta to midtones; high scene-linear values stay exactly original.
    peak = original.max(axis=-1, keepdims=True)
    weight = np.clip((1.2 - peak) / 0.4, 0, 1)
    composite = np.maximum(original + weight * (result - proxy), 0)
    if not np.array_equal(composite[peak[..., 0] >= 1.2],
                          original[peak[..., 0] >= 1.2]):
        raise AssertionError("HDR highlights changed")

    before, after = to_preview(original), to_preview(composite)
    comparison = Image.new("RGB", (WIDTH * 2, HEIGHT + 40), "#101010")
    comparison.paste(before, (0, 40))
    comparison.paste(after, (WIDTH, 40))
    draw = ImageDraw.Draw(comparison)
    draw.text((12, 12), "Game input / preview tone map", fill="white")
    draw.text((WIDTH + 12, 12), "NR 540p / highlights preserved", fill="white")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    comparison.save(OUT)
    changed = np.max(np.abs(np.asarray(after).astype("int16") -
                            np.asarray(before).astype("int16")), axis=2)
    print(json.dumps({"output": str(OUT), "pixels_changed": int(np.count_nonzero(changed)),
                      "max_preview_channel_difference": int(changed.max()),
                      "hdr_pixels_preserved": int(np.count_nonzero(peak[..., 0] >= 1.2))}))


if __name__ == "__main__":
    main()
