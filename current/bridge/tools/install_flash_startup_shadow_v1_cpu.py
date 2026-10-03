"""Install only the prepared ASI and passive host observer after game exit."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PERF = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001")
BUILD = PERF / "periodic-flash-startup-shadow-v1-native-build"
SOURCE = PROJECT / "artifacts/periodic-flash-startup-shadow-v1-source-20261001"
BACKUP = PERF / "periodic-flash-startup-shadow-v1-installed-20261001"
HOST = Path(r"G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime\game\nr_game_pre_xess_host.py")
OLD_HOST_SHA = "0ab98f559e0b81cb4c92113dc7e9e4763931c1657b94de049ee9748e730fbc75"

spec = importlib.util.spec_from_file_location("flash_native_installer", PROJECT / "tools/install_periodic_flash_state_proof_v1_cpu.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
native.SOURCE, native.BUILD = SOURCE, BUILD
native.EXPECTED_BEFORE = "c628d341b2c0d25087c926228b85d5506abc5b29d1fffc1df8d85b6a3955f3da"


def inspect():
    original, prepared, report = native.inspect()
    native.physical(HOST)
    old_host, new_host = HOST.read_bytes(), (SOURCE / "game/nr_game_pre_xess_host.py").read_bytes()
    if native.sha(old_host) != OLD_HOST_SHA:
        raise RuntimeError("installed host changed")
    result = [(native.TARGET, original, prepared), (HOST, old_host, new_host)]
    report.update(runtime_scope="legacy-direct-shadow-v1", GPU_executed=False,
                  scope="ASI plus passive startup observer only",
                  flash_fix_accepted=False,
                  files=[{"target": str(target), "before_sha256": native.sha(before),
                          "installed_sha256": native.sha(after)} for target, before, after in result])
    return result, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    files, report = inspect()
    if args.check_only:
        print(json.dumps({**report, "status": "checked"}, ensure_ascii=False))
        return
    native.require_idle()
    native.physical(BACKUP)
    if BACKUP.exists() or BACKUP.resolve().parent != PERF.resolve():
        raise RuntimeError("fresh backup directory required")
    BACKUP.mkdir()
    for target, before, after in files:
        (BACKUP / target.name).write_bytes(before)
    (BACKUP / "before.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    native.require_idle()
    checked_files, checked_report = inspect()
    if (files, report) != (checked_files, checked_report):
        raise RuntimeError("source or installed files changed during backup")
    written = []
    try:
        for target, before, after in files:
            written.append((target, before))
            target.write_bytes(after)
            if target.read_bytes() != after:
                raise RuntimeError("installed bytes differ: " + str(target))
    except BaseException:
        for target, before in reversed(written):
            target.write_bytes(before)
        report["status"] = "rolled_back"
        (BACKUP / "rollback.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        raise
    report["status"] = "installed"
    (BACKUP / "installed.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "backup": str(BACKUP)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
