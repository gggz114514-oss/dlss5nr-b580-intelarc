"""CPU-only analysis of a completed R11G11B10_FLOAT readback buffer.

The capture producer must independently prove its copy's resource state and
fence completion. This tool never touches the game, GPU, or D3D12 resources.
"""

import argparse
import json
import math
import mmap
import struct
from pathlib import Path


def _ufloat(value: int, mantissa_bits: int) -> float:
    exponent = value >> mantissa_bits
    mantissa = value & ((1 << mantissa_bits) - 1)
    if exponent == 31:
        return math.inf if mantissa == 0 else math.nan
    if exponent == 0:
        return math.ldexp(mantissa, 1 - 15 - mantissa_bits)
    return math.ldexp((1 << mantissa_bits) | mantissa,
                      exponent - 15 - mantissa_bits)


def decode_pixel(packed: int) -> tuple[float, float, float]:
    return (_ufloat(packed & 0x7FF, 6),
            _ufloat((packed >> 11) & 0x7FF, 6),
            _ufloat((packed >> 22) & 0x3FF, 5))


def analyze(path: Path, width: int, height: int, row_pitch: int,
            max_pixels: int) -> dict:
    if width <= 0 or height <= 0 or width > 16384 or height > 16384:
        raise ValueError("Invalid image extent")
    if row_pitch < width * 4 or row_pitch % 4:
        raise ValueError("Invalid row pitch")
    total = width * height
    if max_pixels <= 0 or max_pixels > 4_000_000:
        raise ValueError("max_pixels must be 1..4,000,000")
    expected = (height - 1) * row_pitch + width * 4
    if path.stat().st_size < expected:
        raise ValueError("Readback file is shorter than the declared image")
    indices = range(total) if total <= max_pixels else (
        i * total // max_pixels for i in range(max_pixels))
    count = above_one = nonfinite = 0
    maxima = [0.0, 0.0, 0.0]
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0,
                                               access=mmap.ACCESS_READ) as data:
        for pixel_index in indices:
            y, x = divmod(pixel_index, width)
            packed = struct.unpack_from("<I", data, y * row_pitch + x * 4)[0]
            rgb = decode_pixel(packed)
            count += 1
            if not all(math.isfinite(v) for v in rgb):
                nonfinite += 1
                continue
            above_one += any(v > 1.0 for v in rgb)
            for channel in range(3):
                maxima[channel] = max(maxima[channel], rgb[channel])
    return {"format": "DXGI_FORMAT_R11G11B10_FLOAT", "width": width,
            "height": height, "pixels_examined": count,
            "complete_frame": count == total, "pixels_above_one": above_one,
            "nonfinite_pixels": nonfinite, "channel_max": maxima,
            "gpu_capture_provenance_verified_by_this_tool": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--row-pitch", type=int, required=True)
    parser.add_argument("--max-pixels", type=int, default=4_000_000)
    args = parser.parse_args()
    print(json.dumps(analyze(args.input, args.width, args.height,
                             args.row_pitch, args.max_pixels), ensure_ascii=False))


if __name__ == "__main__":
    main()
