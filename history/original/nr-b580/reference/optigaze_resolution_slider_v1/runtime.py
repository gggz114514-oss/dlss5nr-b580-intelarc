"""Open one isolated B580 geometry session per process; no product edits."""

import json
import os
from pathlib import Path
import sys

from geometry import Geometry

ROOT = Path("E:/ComfyUI-aki-v3-IntelArc_20260722")
REF = ROOT / "nr-b580/reference"
CONFIG = Path("D:/Codex-NR-Experiments/nr-b580/product-v1/local-runtime-v1.json")
CACHE = Path("D:/Codex-NR-Experiments/nr-b580/optigaze240-v1-smoke/cache/triton-cache-c32-triton38-v1")
_DLL_HANDLES = []


def initialize(geometry: Geometry):
    rows = REF / "fullsize_rows_v1"
    sys.path[:0] = [str(rows), str(REF / "optigaze360_v1")]
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
    if not CACHE.is_dir():
        raise FileNotFoundError(f"Triton cache missing: {CACHE}")
    os.environ["TRITON_CACHE_DIR"] = str(CACHE)
    sys.path.insert(0, str(REF / "nr_geom_re4_v1"))
    from re4_session_v1 import open_session
    import torch
    config = json.loads(CONFIG.read_text(encoding="utf-8-sig"))
    args = (ROOT / "nr-b580", config["profile_path"], config["profile_sha256"])
    session = open_session(*args, geometry=geometry.low, padding=geometry.padded)
    return torch, session
