"""Back up and install the reviewed CPU validation/control files only.

Uses the physical common runtime, never Cyberpunk's nr-runtime junction.
No kernel, cache, model, DLL or persistent product configuration is changed.
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
ROOT = PROJECT.parent / ".codex-worktrees/re8-fp8-unround-fast-20260928"
RUNTIME = Path(r"G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime")
PLUGINS = Path(r"G:\epic\Cyberpunk2077\bin\x64\plugins")
PERF = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001")
FILES = (
    ("common/numeric_frame_validation_720_v1.py", ROOT / "game/numeric_frame_validation_720_v1.py",
     RUNTIME / "game/numeric_frame_validation_720_v1.py", None),
    ("common/numeric_cleanup_suite_720_v1.py", ROOT / "game/numeric_cleanup_suite_720_v1.py",
     RUNTIME / "game/numeric_cleanup_suite_720_v1.py", "a8df25a32f1aa01340949b46def1ce5d99c57d085486c437a9fd4436655bab83"),
    ("common/decoder_input_full_k_720_v1.py", ROOT / "game/decoder_input_full_k_720_v1.py",
     RUNTIME / "game/decoder_input_full_k_720_v1.py", "8c6ce4a1fd6372cc4a5ecf4355725e4ff437eb9c3c8a00ec965f7e11102a5d28"),
    ("common/nr_game_controls.py", ROOT / "game/nr_game_controls.py",
     RUNTIME / "game/nr_game_controls.py", "503aa8a0376de46c9e391c8a3dd702823f3699b803b10ca292d53544c089d04e"),
    ("plugins/cyberpunk_nr_adapter.py", PROJECT / "game/cyberpunk_nr_adapter.py",
     PLUGINS / "cyberpunk_nr_adapter.py", "7ce77bd3e8d651801e81db7bdfb4178b66dc4ca3491a9d00b2067362a2ec3d36"),
    ("plugins/nr_numeric_cost_meter_v1.py", PROJECT / "game/nr_numeric_cost_meter_v1.py",
     PLUGINS / "nr_numeric_cost_meter_v1.py", "bcf77da5b07063b9211e0f1b69e9697a5b1328def13f4fa5ef3f0f4ba1048f1b"),
)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def physical(path):
    for part in (path, *path.parents):
        if part.is_symlink() or part.is_junction():
            raise RuntimeError("refuse redirected path: " + str(part))


def game_idle():
    out = subprocess.check_output(["tasklist.exe", "/FO", "CSV", "/NH"], text=True, errors="replace")
    names = {row[0].casefold() for row in csv.reader(io.StringIO(out)) if row}
    if names & {"cyberpunk2077.exe", "re8.exe", "zenlesszonezero.exe"}:
        raise RuntimeError("game is running; install only after normal exit")


def inspect():
    entries, payloads = [], {}
    for label, source, target, expected in FILES:
        physical(source)
        physical(target)
        if not (target.is_relative_to(RUNTIME / "game") or target.is_relative_to(PLUGINS)):
            raise RuntimeError("target escaped the explicit Python directories")
        data = source.read_bytes()
        compile(data, str(source), "exec", dont_inherit=True)
        before = target.read_bytes() if target.exists() else None
        if (before is None) != (expected is None) or (before is not None and sha(before) != expected):
            raise RuntimeError("installed file differs from reviewed original: " + str(target))
        entries.append({"label": label, "source": str(source), "target": str(target),
                        "before_sha256": sha(before) if before is not None else None,
                        "after_sha256": sha(data)})
        payloads[label] = (before, data)
    return entries, payloads


def install(backup):
    game_idle()
    physical(backup)
    if not backup.is_relative_to(PERF.resolve()) or backup.exists():
        raise ValueError("backup must be a fresh task-owned PERF child directory")
    entries, payloads = inspect()
    backup.mkdir(parents=True)
    report = {"status": "ready", "files": entries, "backup": str(backup),
              "scope": "CPU frame validation and web controls; no kernel/cache/model/DLL changes"}
    for entry in entries:
        before, _ = payloads[entry["label"]]
        if before is not None:
            path = backup / entry["label"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(before)
    (backup / "before.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Recheck game/source/targets after recoverable originals are on disk.
    game_idle()
    checked, _ = inspect()
    if checked != entries:
        raise RuntimeError("source or target changed during backup")
    written = []
    try:
        for entry in entries:
            target = Path(entry["target"])
            before, data = payloads[entry["label"]]
            written.append((target, before))
            target.write_bytes(data)
            if target.read_bytes() != data:
                raise RuntimeError("installed source mismatch: " + str(target))
    except BaseException:
        for target, before in reversed(written):
            if before is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(before)
        report["status"] = "rolled_back"
        (backup / "rollback.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    report["status"] = "installed"
    (backup / "installed-receipt.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "install"))
    parser.add_argument("--backup", type=Path)
    args = parser.parse_args()
    if args.command == "install":
        if args.backup is None:
            raise ValueError("provide --backup")
        result = install(args.backup.resolve())
    else:
        entries, _ = inspect()
        result = {"status": "ready", "files": entries, "writes": False}
    print(json.dumps(result, ensure_ascii=False))
