"""Install the CPU-verified native flash fix after normal game exit.

Only CyberpunkNRBridge.asi is replaced. The signed loader, Python runtime,
model, kernels, cache and current user controls are not written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess


PROJECT = Path(__file__).resolve().parents[1]
PERF = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001")
BUILD = PERF / "periodic-flash-state-proof-v1-native-build"
SOURCE = PROJECT / "artifacts/periodic-flash-state-proof-v1-build-source-20261001"
TARGET = Path(r"G:\epic\Cyberpunk2077\bin\x64\plugins\CyberpunkNRBridge.asi")
EXPECTED_BEFORE = "2f8c9e8450f32da31efeb01f2e8697d2cb6e7a5e66a12a23d593620635e7be9d"
BACKUP = PERF / "periodic-flash-state-proof-v1-installed-20261001"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def physical(path):
    if not path.is_absolute():
        raise RuntimeError("absolute path required")
    for item in (path, *path.parents):
        if item.is_symlink() or item.is_junction():
            raise RuntimeError("refuse redirected path: " + str(item))


def require_idle():
    output = subprocess.check_output(["tasklist.exe", "/FO", "CSV", "/NH"],
                                     text=True, errors="replace")
    names = {row[0].casefold() for row in csv.reader(io.StringIO(output)) if row}
    if names & {"cyberpunk2077.exe", "re8.exe", "zenlesszonezero.exe"}:
        raise RuntimeError("install requires normal game exit")


def inspect():
    receipt_path = BUILD / "build-receipt.json"
    physical(receipt_path)
    report = json.loads(receipt_path.read_text(encoding="utf-8"))
    if (report.get("status") != "cpu_build_passed"
            or Path(report["source_root"]) != SOURCE
            or report.get("GPU_executed") is not False):
        raise RuntimeError("missing completed CPU-only build verification")
    binary = Path(report["asi_path"])
    if binary != BUILD / "Release/CyberpunkNRBridge.asi":
        raise RuntimeError("unexpected built binary")
    for relative, digest in report["source_files"].items():
        path = SOURCE / relative
        physical(path)
        if not path.resolve().is_relative_to(SOURCE.resolve()) or sha(path.read_bytes()) != digest:
            raise RuntimeError("reviewed build source changed: " + relative)
    physical(binary)
    physical(TARGET)
    data, before = binary.read_bytes(), TARGET.read_bytes()
    if sha(data) != report["asi_sha256"] or sha(before) != EXPECTED_BEFORE:
        raise RuntimeError("built binary or installed original changed")
    return before, data, {
        "status": "prepared", "source": str(binary), "target": str(TARGET),
        "before_sha256": sha(before), "installed_sha256": sha(data),
        "build_receipt_sha256": sha(receipt_path.read_bytes()),
        "GPU_executed": False, "scope": "one native ASI only",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    before, data, report = inspect()
    if args.check_only:
        print(json.dumps({**report, "status": "checked"}, ensure_ascii=False))
        return
    require_idle()
    physical(BACKUP)
    if BACKUP.exists() or BACKUP.resolve().parent != PERF.resolve():
        raise RuntimeError("fresh task-owned backup directory required")
    BACKUP.mkdir()
    (BACKUP / "CyberpunkNRBridge.asi").write_bytes(before)
    (BACKUP / "before.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    require_idle()
    checked_before, checked_data, checked_report = inspect()
    if (checked_before, checked_data, checked_report) != (before, data, report):
        raise RuntimeError("files changed during backup")
    try:
        TARGET.write_bytes(data)
        if TARGET.read_bytes() != data:
            raise RuntimeError("installed binary bytes differ")
    except BaseException:
        TARGET.write_bytes(before)
        report["status"] = "rolled_back"
        (BACKUP / "rollback.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    report["status"] = "installed"
    (BACKUP / "installed.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "backup": str(BACKUP)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
