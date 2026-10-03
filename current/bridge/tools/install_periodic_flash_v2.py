"""Install only the frozen, CPU-checked flash diagnostics after normal exit.

Uses the physical common runtime. Does not install CMake's copied adapter/web,
change kernels, or replace the signed OptiScaler loader.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
STAGE = PROJECT / "artifacts/periodic-flash-v2-source-20261001"
PERF = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001")
BACKUP = PERF / "periodic-flash-v2-installed-20261001"
RUNTIME = Path(r"G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime")
PLUGINS = Path(r"G:\epic\Cyberpunk2077\bin\x64\plugins")
ASI = PERF / "flash-v2-native-build/Release/CyberpunkNRBridge.asi"
ASI_SHA = "2f8c9e8450f32da31efeb01f2e8697d2cb6e7a5e66a12a23d593620635e7be9d"
FILES = (
    ("plugin/CyberpunkNRBridge.asi", ASI, PLUGINS / "CyberpunkNRBridge.asi",
     "246af3af01d84b8e468761875e7c03d5060ed6267aca7a3ff1a09f20d1f779f8"),
    ("game/nr_game_pre_xess_host.py", STAGE / "game/nr_game_pre_xess_host.py",
     RUNTIME / "game/nr_game_pre_xess_host.py",
     "c76bfecfdb1a417d214b5c9132402aa4aae4848e967715f93936408dae0c30af"),
    ("game/periodic_flash_snapshot_v2.py", STAGE / "game/periodic_flash_snapshot_v2.py",
     RUNTIME / "game/periodic_flash_snapshot_v2.py", None),
    ("game/nr_temporal_diagnostics_v1.py", STAGE / "game/nr_temporal_diagnostics_v1.py",
     RUNTIME / "game/nr_temporal_diagnostics_v1.py", None),
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def physical(path):
    for item in (path, *path.parents):
        if item.is_symlink() or item.is_junction():
            raise RuntimeError("refuse redirected path: " + str(item))


def idle():
    result = subprocess.check_output(["tasklist.exe", "/FO", "CSV", "/NH"], text=True, errors="replace")
    names = {row[0].casefold() for row in csv.reader(io.StringIO(result)) if row}
    if names & {"cyberpunk2077.exe", "re8.exe", "zenlesszonezero.exe"}:
        raise RuntimeError("install only after normal game exit")


def inspect():
    receipt = json.loads((STAGE / "source-receipt.json").read_text(encoding="utf-8"))
    for name, digest in receipt["files"].items():
        path = STAGE / name
        physical(path)
        if not path.is_relative_to(STAGE) or sha(path.read_bytes()) != digest:
            raise RuntimeError("frozen source changed: " + name)
    if sha(ASI.read_bytes()) != ASI_SHA:
        raise RuntimeError("CPU-built ASI changed")
    spec = importlib.util.spec_from_file_location("flash_host_delta_proof", PROJECT / "tools/prepare_periodic_flash_diag_v2_cpu.py")
    proof_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(proof_module)
    final, proof = proof_module.host_from_actual(
        FILES[1][2].read_bytes(),
        (PROJECT.parent / ".codex-worktrees/re8-fp8-unround-fast-20260928/game/nr_game_pre_xess_host.py").read_bytes(),
        FILES[1][1].read_text(encoding="utf-8"),
    )
    if final != FILES[1][1].read_text(encoding="utf-8"):
        raise RuntimeError("diagnostic host delta changed")
    entries, payloads = [], {}
    for label, source, target, expected in FILES:
        physical(source)
        physical(target)
        if not (target.is_relative_to(RUNTIME / "game") or target == PLUGINS / "CyberpunkNRBridge.asi"):
            raise RuntimeError("target escaped explicit scope")
        data = source.read_bytes()
        if source.suffix == ".py":
            compile(data, str(source), "exec", dont_inherit=True)
        before = target.read_bytes() if target.exists() else None
        if (before is None) != (expected is None) or (before is not None and sha(before) != expected):
            raise RuntimeError("actual original changed: " + str(target))
        entries.append({"label": label, "source": str(source), "target": str(target),
                        "before_sha256": sha(before) if before is not None else None,
                        "installed_sha256": sha(data)})
        payloads[label] = (before, data)
    return entries, payloads, proof


def main():
    idle()
    physical(BACKUP)
    if BACKUP.exists() or BACKUP.resolve().parent != PERF.resolve():
        raise RuntimeError("use the fresh task-owned backup directory")
    entries, payloads, proof = inspect()
    BACKUP.mkdir()
    report = {"status": "prepared", "files": entries, "host_delta_proof": proof,
              "GPU_executed": False, "signed_loader_changed": False,
              "scope": "four diagnostic files; adapter/controls/kernels/cache/model unchanged",
              "child_environment": {"NR_DIAG_PERIODIC_FLASH_V2": "1"}}
    for entry in entries:
        before, _ = payloads[entry["label"]]
        if before is not None:
            path = BACKUP / entry["label"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(before)
    (BACKUP / "before.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    idle()
    checked, _, _ = inspect()
    if checked != entries:
        raise RuntimeError("source/target changed while backing up")
    written = []
    try:
        for entry in entries:
            path = Path(entry["target"])
            before, data = payloads[entry["label"]]
            written.append((path, before))
            path.write_bytes(data)
            if path.read_bytes() != data:
                raise RuntimeError("installed bytes differ: " + str(path))
    except BaseException:
        for path, before in reversed(written):
            if before is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(before)
        report["status"] = "rolled_back"
        (BACKUP / "rollback.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    report["status"] = "installed"
    (BACKUP / "installed.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "backup": str(BACKUP), "files": entries}, ensure_ascii=False))


if __name__ == "__main__":
    main()
