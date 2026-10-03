"""Install two opt-in Python diagnostic files, preserving actual game originals."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
TARGET = Path(r"G:\epic\Cyberpunk2077\bin\x64\plugins")
PERF = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001")
NAMES = ("cyberpunk_nr_adapter.py", "nr_numeric_cost_meter_v1.py")
ORIGINAL_ADAPTER_SHA256 = "760d74d7db4f9922be2428a12e5bb6519d59a1cfb93afae09cf5f385ba57d200"

def digest(data):
    return hashlib.sha256(data).hexdigest()

def inspect():
    for path in (TARGET, *TARGET.parents):
        if path.is_symlink() or (hasattr(path, "is_junction") and path.is_junction()):
            raise RuntimeError("refuse redirected game plugin path: " + str(path))
    sources = {name: (PROJECT / "game" / name).read_bytes() for name in NAMES}
    installed = (TARGET / NAMES[0]).read_bytes()
    if digest(installed) != ORIGINAL_ADAPTER_SHA256:
        raise RuntimeError("installed adapter differs from the original reviewed adapter; preserve it")
    if (TARGET / NAMES[1]).exists():
        raise RuntimeError("preserve existing cost meter; use its installation receipt")
    return sources, installed

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "install"))
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()
    sources, installed = inspect()
    report = {"status": "ready", "source_sha256": {n: digest(d) for n, d in sources.items()},
              "original_adapter_sha256": digest(installed), "target": str(TARGET),
              "activation": "child-process environment CYBERPUNK_NR_COST_METER=1; default off"}
    if args.command == "install":
        processes = subprocess.check_output(["tasklist.exe", "/FO", "CSV", "/NH"],
                                             text=True, errors="replace")
        names = {r[0].casefold() for r in csv.reader(io.StringIO(processes)) if r}
        if names & {"cyberpunk2077.exe", "re8.exe", "zenlesszonezero.exe"}:
            raise RuntimeError("game still running; preserve originals until normal exit")
        if args.backup is None:
            raise ValueError("provide new task-owned D --backup")
        backup = args.backup.resolve()
        if not backup.is_relative_to(PERF.resolve()) or backup.exists():
            raise ValueError("backup must be a new PERF child directory")
        backup.mkdir(parents=True)
        (backup / NAMES[0]).write_bytes(installed)
        (backup / "before.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        # Recheck the original after writing its recoverable backup.
        if (TARGET / NAMES[0]).read_bytes() != installed:
            raise RuntimeError("game adapter changed during backup")
        try:
            for name in NAMES:
                (TARGET / name).write_bytes(sources[name])
                if (TARGET / name).read_bytes() != sources[name]:
                    raise RuntimeError("installed diagnostic SHA mismatch: " + name)
        except BaseException:
            (TARGET / NAMES[0]).write_bytes(installed)
            raise
        report.update(status="installed", backup=str(backup))
        (backup / "installed-receipt.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))

if __name__ == "__main__":
    main()
