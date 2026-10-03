"""Promote the frozen native source after live acceptance; no runtime writes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PERF = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001")
SOURCE = PROJECT / "artifacts/periodic-flash-cp-default-v5-source-20261001"
BUILD = PERF / "periodic-flash-cp-default-v5-native-build"
PLAN = BUILD / "canonical-promotion-plan.json"
BACKUP = PERF / "periodic-flash-cp-default-v5-canonical-backup-20261001"
FILES = ("CMakeLists.txt", "include/nr_sync_probe.h", "src/asi.cpp",
         "src/deferred_identity.cpp", "src/deferred_identity.h", "src/nr_sync_probe.cpp",
         "src/periodic_flash_native_diag.h", "tests/state_proof_cpu.cpp")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    options = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "native_installer", PROJECT / "tools/install_periodic_flash_state_proof_v1_cpu.py")
    native = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(native)
    receipt_path = BUILD / "build-receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("status") != "cpu_build_passed" or Path(receipt["source_root"]) != SOURCE:
        raise RuntimeError("frozen completed CPU build required")
    files = []
    staged = {}
    for name, digest in receipt["source_files"].items():
        source = SOURCE / name
        target = PROJECT / name
        native.physical(source)
        native.physical(target)
        data = source.read_bytes()
        if sha(data) != digest:
            raise RuntimeError("staged source changed: " + name)
        if name.startswith("game/"):
            continue  # passive host source remains in its own shared runtime tree
        before = target.read_bytes() if target.is_file() else None
        if name not in FILES and (before is None or sha(before) != digest):
            raise RuntimeError("unreviewed canonical difference: " + name)
        if name in FILES:
            staged[name] = (target, before, data)
            files.append({"file": name, "before_sha256": sha(before) if before is not None else None,
                          "after_sha256": digest})
    plan = {"status": "checked", "build_receipt_sha256": sha(receipt_path.read_bytes()),
            "files": files, "scope": "eight E native source files only; no game/runtime/cache writes"}
    if not options.apply:
        with PLAN.open("x", encoding="utf-8") as handle:
            json.dump(plan, handle, indent=2)
        print(json.dumps({"status": "checked", "files": len(files), "plan": str(PLAN)}))
        return
    expected = json.loads(PLAN.read_text(encoding="utf-8"))
    if expected != plan or len(files) != len(FILES):
        raise RuntimeError("canonical source changed since review")
    native.physical(BACKUP)
    if BACKUP.exists() or BACKUP.resolve().parent != PERF.resolve():
        raise RuntimeError("fresh task-owned backup required")
    BACKUP.mkdir()
    for name, (_, before, _) in staged.items():
        if before is not None:
            saved = BACKUP / name
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(before)
    written = []
    try:
        for name, (target, before, data) in staged.items():
            if (target.read_bytes() if target.is_file() else None) != before:
                raise RuntimeError("canonical source changed during promotion: " + name)
            written.append(name)
            target.write_bytes(data)
            if target.read_bytes() != data:
                raise RuntimeError("canonical write differs: " + name)
    except BaseException:
        for name in reversed(written):
            target, before, _ = staged[name]
            if before is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(before)
        raise
    plan["status"] = "promoted"
    (BACKUP / "promoted.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "promoted", "files": len(files), "backup": str(BACKUP)}))


if __name__ == "__main__":
    main()
