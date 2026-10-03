"""Promote accepted bridge cache/timing source only; never write a G runtime.

Require the frozen CPU build, GPU equality and raw live review receipts. Back
up only changed E sources on D; reject unrelated canonical changes and redirects.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SHARED = PROJECT.parent / ".codex-worktrees/re8-fp8-unround-fast-20260928/game"
STAGE = PROJECT / "artifacts/bridge-cache-integrated-v1-20261001/payload"
PERF = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001")
BUILD = PERF / "bridge-cache-integrated-v1-build/build-receipt.json"
OUTER = PROJECT / "artifacts/bridge-outer-timing-v1-20261001/source-manifest.json"
FILES = ("CMakeLists.txt", "include/nr_bridge.h", "src/asi.cpp",
         "src/deferred_identity.cpp", "src/deferred_identity.h",
         "src/nr_hdr_proxy.cpp", "src/nr_hdr_proxy.h", "src/nr_record_timing.h",
         "game/cyberpunk_nr_adapter.py", "game/cyberpunk_nr_web.py",
         "game/nr_game_controls.py", "tests/record_timing_cpu.cpp")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def physical(path):
    for part in (path, *path.parents):
        if part.is_symlink() or part.is_junction():
            raise RuntimeError("Redirected source/backup: " + str(part))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "promote"))
    parser.add_argument("--live-review", type=Path, required=True)
    args = parser.parse_args()
    physical(args.live_review)
    review_path = args.live_review.resolve(strict=True)
    if not review_path.is_relative_to(PERF):
        raise ValueError("Use the task's reviewed live receipt")
    live, build, original = read(review_path), read(BUILD), read(OUTER)
    if (live["status"] != "main_raw_review_passed" or not live["cache_restored_ON"] or
            not live["paired"]["record_last_ms"]["both_on_below_both_off"] or
            live["paired"]["record_last_ms"]["saved_ms"] <= 0 or
            build["status"] != "cpu_build_passed"):
        raise RuntimeError("Accepted CPU build and repeatable live gain are required")
    proof_path = PERF.parent / "bridge-cache-gpu-probe-v1-retry3-20261001/GPU-RECEIPT.json"
    proof = read(proof_path)
    installer_receipt = read(PERF / "bridge-cache-trial-v1/20261001T150507112380Z/installed.json")
    if (proof["status"] != "passed" or proof["outputs_identical"] is not True or
            proof["stable_cache_counters"] is not True or
            installer_receipt["status"] != "installed_trial"):
        raise RuntimeError("Reviewed equality-gated trial install receipt required")
    changes = []
    for name in FILES:
        source = STAGE / name
        target = SHARED / "nr_game_controls.py" if name == "game/nr_game_controls.py" else PROJECT / name
        physical(source); physical(target)
        data = source.read_bytes()
        if sha(data) != build["source_pins"][name]:
            raise RuntimeError("Built source changed: " + name)
        before = target.read_bytes() if target.exists() else None
        if before == data:
            continue
        label = "shared/game/nr_game_controls.py" if name == "game/nr_game_controls.py" else name
        expected = original["base_pins"].get(label)
        if (None if before is None else sha(before)) != expected:
            raise RuntimeError("Canonical source has unrelated edits: " + str(target))
        if source.suffix == ".py":
            compile(data, str(target), "exec", dont_inherit=True)
        changes.append((label, target, before, data))
    if args.command == "check":
        print(json.dumps({"status": "ready", "E_changes": len(changes), "G_writes": False}))
        return
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup = PERF / "bridge-cache-source-merged-v1" / tag
    physical(backup)
    backup.mkdir(parents=True, exist_ok=False)
    rows = []
    for label, target, before, data in changes:
        if before is not None:
            saved = backup / label
            saved.parent.mkdir(parents=True, exist_ok=True)
            saved.write_bytes(before)
        rows.append({"label": label, "target": str(target),
                     "before_sha256": None if before is None else sha(before), "after_sha256": sha(data)})
    receipt = {"status": "backed_up", "changes": rows, "live_review": str(review_path),
               "live_review_sha256": sha(review_path.read_bytes()), "build_receipt": str(BUILD),
               "GPU_receipt": str(proof_path), "GPU_receipt_sha256": sha(proof_path.read_bytes()),
               "G_writes": False, "GPU_executed": False}
    (backup / "before.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    written = []
    try:
        for label, target, before, data in changes:
            if (target.read_bytes() if target.exists() else None) != before:
                raise RuntimeError("Canonical changed after backup")
            written.append((target, before))
            target.write_bytes(data)
            if target.read_bytes() != data:
                raise RuntimeError("Canonical write mismatch")
    except BaseException:
        for target, before in reversed(written):
            if before is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(before)
        receipt["status"] = "rolled_back"
        (backup / "rollback.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        raise
    receipt["status"] = "source_merged"
    (backup / "promoted.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "E_changes": len(changes),
                      "G_writes": False, "receipt": str(backup / "promoted.json")}))


if __name__ == "__main__":
    main()
