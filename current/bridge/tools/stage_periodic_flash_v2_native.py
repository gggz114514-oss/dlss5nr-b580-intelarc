"""Stage the passed CPU-only diagnostic patch without changing live sources."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

PROJECT = Path(__file__).resolve().parents[1]
PERF = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001")
DEST = PROJECT / "artifacts/periodic-flash-v2-source-20261001"
G_HOST = Path(r"G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime\game\nr_game_pre_xess_host.py")
G_HOST_SHA = "c76bfecfdb1a417d214b5c9132402aa4aae4848e967715f93936408dae0c30af"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical(path):
    for item in (path, *path.parents):
        if item.is_symlink() or item.is_junction():
            raise RuntimeError("refuse redirected source: " + str(item))


if __name__ == "__main__":
    report = json.loads((PERF / "periodic-flash-sol61-cpu-v2.json").read_text(encoding="utf-8-sig"))
    if report.get("passed") is not True or report.get("unexpected_gpu_modules"):
        raise RuntimeError("diagnostic CPU checks have not passed")
    for name, value in report["sources"].items():
        if sha(Path(name)) != value:
            raise RuntimeError("diagnostic changed after CPU receipt: " + name)
    if sha(G_HOST) != G_HOST_SHA:
        raise RuntimeError("installed host differs from the compared baseline")
    if DEST.exists() or not DEST.is_relative_to(PROJECT / "artifacts"):
        raise ValueError("use the fresh explicit diagnostic stage")
    physical(DEST)
    spec = importlib.util.spec_from_file_location("flashdiag_stage_builder", PROJECT / "tools/build_periodic_flash_diag_v2_cpu.py")
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    patched = builder.patched_sources()
    DEST.mkdir()
    shutil.copy2(PROJECT / "CMakeLists.txt", DEST / "CMakeLists.txt")
    for name in ("src", "include", "game", "tests", "shaders"):
        source = PROJECT / name
        for item in (source, *source.rglob("*")):
            physical(item)
        shutil.copytree(source, DEST / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    host = None
    for name, content in patched.items():
        relative = Path(name)
        if relative.parts[0] == PROJECT.name:
            relative = Path(*relative.parts[1:])
            if not relative.is_relative_to("src"):
                raise RuntimeError("unexpected native patch target: " + name)
            (DEST / relative).write_text(content, encoding="utf-8")
        elif relative.name == "nr_game_pre_xess_host.py":
            host = content
        else:
            raise RuntimeError("unexpected host patch target: " + name)
    if host is None:
        raise RuntimeError("host diagnostics missing from patch")
    # G/E baseline diff was reviewed: only v1 passive hooks are absent on G.
    (DEST / "game/nr_game_pre_xess_host.py").write_text(host, encoding="utf-8")
    shutil.copy2(PROJECT.parent / ".codex-worktrees/re8-fp8-unround-fast-20260928/game/nr_temporal_diagnostics_v1.py",
                 DEST / "game/nr_temporal_diagnostics_v1.py")
    receipt = {"status": "staged", "source": str(DEST), "GPU_touched": False,
               "G_writes": False, "base_host_sha256": G_HOST_SHA,
               "files": {str(path.relative_to(DEST)): sha(path) for path in DEST.rglob("*") if path.is_file()}}
    (DEST / "source-receipt.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "files"}, ensure_ascii=False))
