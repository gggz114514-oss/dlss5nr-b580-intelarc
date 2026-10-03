import importlib.util
import math
import struct
import sys
import tempfile
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / "tools" / "analyze_r11g11b10.py"
spec = importlib.util.spec_from_file_location("pixel_probe", module_path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)

one = (15 << 6) | ((15 << 6) << 11) | ((15 << 5) << 22)
assert probe.decode_pixel(one) == (1.0, 1.0, 1.0)
bright = (15 << 6 | 32) | ((15 << 6) << 11) | ((15 << 5) << 22)
assert probe.decode_pixel(bright) == (1.5, 1.0, 1.0)
half = (14 << 6) | ((14 << 6) << 11) | ((14 << 5) << 22)
assert probe.decode_pixel(half) == (0.5, 0.5, 0.5)
assert math.isinf(probe.decode_pixel(31 << 6)[0])

with tempfile.TemporaryDirectory(dir=sys.argv[1]) as directory:
    image = Path(directory) / "two_rows.bin"
    image.write_bytes(struct.pack("<II", one, bright) + bytes(248) +
                      struct.pack("<II", half, one))
    report = probe.analyze(image, 2, 2, 256, 4)
    assert report["complete_frame"] is True
    assert report["pixels_above_one"] == 1
    assert report["channel_max"] == [1.5, 1.0, 1.0]
