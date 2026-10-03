"""Launch one arm with the reviewed MSVC/Intel host compiler environment."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path("E:/ComfyUI-aki-v3-IntelArc_20260722")
ROWS = ROOT / "nr-b580/reference/fullsize_rows_v1"
DATA = Path("D:/Codex-NR-Experiments/nr-b580/optigaze240-v1")
CACHE = Path("D:/Codex-NR-Experiments/nr-b580/optigaze240-v1-smoke/cache/triton-cache-c32-triton38-v1")
sys.path.insert(0, str(ROWS))
import rows_toolchain_v1


def main():
    arm = sys.argv[1]
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if arm in ("full", "nr256", "low240"):
        env.update(rows_toolchain_v1.environment(DATA / "tmp"))
        env["CODEBUDDY_SAFE_DELETE_ENABLED"] = "0"
        env["MAX_JOBS"] = "14"
        env["TRITON_CACHE_DIR"] = str(CACHE)
    cmd = [str(ROOT / "ComfyUI-aki-v3-IntelArc/python/python.exe"),
           str(Path(__file__).resolve().parent / "run_video.py"), arm]
    raise SystemExit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
