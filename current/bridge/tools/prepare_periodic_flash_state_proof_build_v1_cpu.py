"""Prepare the exact installed native-v2 build base without touching the game."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
BASE = PROJECT / "artifacts/periodic-flash-v2-source-20261001"
OUTPUT = PROJECT / "artifacts/periodic-flash-state-proof-v1-build-source-20261001"
BUILD = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001\periodic-flash-state-proof-v1-native-build")


def physical(path):
    for item in (path, *path.parents):
        if item.is_symlink() or item.is_junction():
            raise RuntimeError("redirected path: " + str(item))


def main():
    physical(BASE)
    physical(OUTPUT)
    physical(BUILD)
    if OUTPUT.exists():
        raise RuntimeError("fresh build source stage required")
    receipt = json.loads((BASE / "source-receipt.json").read_text(encoding="utf-8"))
    files = {}
    for name, digest in receipt["files"].items():
        path = BASE / name
        physical(path)
        if not path.resolve().is_relative_to(BASE.resolve()):
            raise RuntimeError("base path escaped")
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest() != digest:
            raise RuntimeError("frozen installed source base changed: " + name)
        files[name] = data
    OUTPUT.mkdir()
    for name, data in files.items():
        path = OUTPUT / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    BUILD.mkdir(parents=True, exist_ok=True)
    report = {"status": "base_staged", "base": str(BASE), "output": str(OUTPUT),
              "base_files": receipt["files"], "GPU_executed": False, "G_writes": False}
    with (BUILD / "base-stage-receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "files": len(files), "source": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
