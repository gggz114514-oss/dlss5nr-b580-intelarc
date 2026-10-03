"""Install only the six authenticated missing front cache groups, CPU only.

No existing cache file is overwritten. The group marker is published last.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import uuid

PERF = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001")
PLAN = PERF / "front-cache-missing-cpu-20261001.json"
SOURCE = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\c32-window-mlp-chain-20260929\fullframe-selected-diskonly\triton-cache")
TARGET = Path(r"G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime\data\fast-cache")
NAMES = {"__grp__front.json", "front.json", "front.source", "front.spv", "front.ttgir", "front.ttir", "front.llir"}


def digest(data):
    return hashlib.sha256(data).hexdigest()


def bounded(path, root):
    if not path.is_absolute() or not path.is_relative_to(root):
        raise ValueError("Path outside explicit cache root: " + str(path))
    for ancestor in (path, *path.parents):
        if ancestor.exists() and (ancestor.is_symlink() or
                (hasattr(ancestor, "is_junction") and ancestor.is_junction())):
            raise ValueError("Link in cache path: " + str(ancestor))
        if ancestor == root:
            break


def encoded_payload(value, expected):
    # The audit records JSON objects and their exact intended byte hashes.
    # Accept only a representation reproducing that hash; never amend it.
    for indent in (2, None):
        for sorted_keys in (False, True):
            for newline in ("", "\n"):
                data = (json.dumps(value, indent=indent, sort_keys=sorted_keys) + newline).encode("utf-8")
                if digest(data) == expected:
                    return data
    raise ValueError("Cannot reproduce audited JSON payload SHA")


def prepare(output):
    plan_bytes = PLAN.read_bytes()
    plan = json.loads(plan_bytes.decode("utf-8-sig"))
    variants = [v for v in plan["variants"] if v["seed_abi"] == "i64"]
    if len(variants) != 6 or len({v["cache_key"] for v in variants}) != 6:
        raise ValueError("Expected exactly six missing high-seed groups")
    rows = []
    for v in variants:
        if digest(Path(v["receipt_path"]).read_bytes()) != v["receipt_sha256"]:
            raise ValueError("Completed source receipt changed")
        if not v["all_recorded_source_file_SHAs_match"] or not v["binary_SHA_matches_preflight"]:
            raise ValueError("Group lacks completed source/binary evidence")
        source_dir, target_dir = Path(v["source_group_dir"]), Path(v["target_group_dir"])
        bounded(source_dir, SOURCE); bounded(target_dir, TARGET)
        if source_dir != SOURCE / v["cache_key"] or target_dir != TARGET / v["cache_key"]:
            raise ValueError("Unexpected group directory")
        if {p["relative_path"] for p in v["file_plan"]} != NAMES:
            raise ValueError("Unexpected front cache files")
        if v["planned_target_metadata"]["hash"] != v["expected_hash_from_actual_completed_receipt"]:
            raise ValueError("Metadata hash disagrees with actual completed receipt")
        for p in v["file_plan"]:
            name = p["relative_path"]
            source, target = Path(p["source_path"]), Path(p["target_path"])
            if source != source_dir / name or target != target_dir / name:
                raise ValueError("Unexpected child path")
            bounded(source, SOURCE); bounded(target, TARGET)
            data = source.read_bytes()
            if digest(data) != p["source_sha256"]:
                raise ValueError("Source cache SHA changed: " + str(source))
            if p["rewrite_for_target"]:
                key = "planned_target_group_marker" if name == "__grp__front.json" else "planned_target_metadata"
                if name not in {"front.json", "__grp__front.json"}:
                    raise ValueError("Only cache metadata may be rewritten")
                data = encoded_payload(v[key], p["planned_target_sha256"])
            if digest(data) != p["planned_target_sha256"]:
                raise ValueError("Planned target SHA mismatch")
            exists = target.exists()
            if exists and digest(target.read_bytes()) != p["planned_target_sha256"]:
                raise ValueError("Refusing to overwrite existing cache file: " + str(target))
            staged = output / "staged" / v["cache_key"] / name
            staged.parent.mkdir(parents=True, exist_ok=True)
            staged.write_bytes(data)
            rows.append({**p, "staged": str(staged), "existed_before": exists,
                         "cache_key": v["cache_key"]})
    if len(rows) != 42:
        raise ValueError("Expected exactly 42 front files")
    return {"plan_sha256": digest(plan_bytes), "groups": 6, "files": rows}


def run(output, apply):
    bounded(output, PERF)
    if output.exists():
        raise ValueError("Use a new task-owned output directory")
    output.mkdir(parents=True)
    receipt = prepare(output)
    receipt.update(started_utc=datetime.now(timezone.utc).isoformat(), applied=False,
                   source_or_model_changed=False, gpu_touched=False)
    # This durable before-state is also the backup manifest: all additions are
    # recorded before any target mutation; matching existing bytes are skipped.
    before = output / "before-state.json"
    before.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    installed = []
    if apply:
        ordered = sorted(receipt["files"], key=lambda row: row["relative_path"] == "__grp__front.json")
        for row in ordered:
            target, staged = Path(row["target_path"]), Path(row["staged"])
            bounded(target, TARGET)
            if target.exists():
                if digest(target.read_bytes()) != row["planned_target_sha256"]:
                    raise ValueError("Target changed since before-state; refusing overwrite")
                continue
            data = staged.read_bytes()
            if digest(data) != row["planned_target_sha256"]:
                raise ValueError("Staging SHA changed")
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name("." + target.name + ".nrfront-" + uuid.uuid4().hex)
            with temp.open("xb") as stream:
                stream.write(data)
            # Windows rename refuses an existing destination, preventing a race
            # from replacing any game-owned cache bytes.
            os.rename(temp, target)
            installed.append(str(target))
        for row in receipt["files"]:
            if digest(Path(row["target_path"]).read_bytes()) != row["planned_target_sha256"]:
                raise ValueError("Installed SHA verification failed")
        receipt["applied"] = True
    receipt.update(status="completed", installed_paths=installed,
                   finished_utc=datetime.now(timezone.utc).isoformat())
    path = output / "completed-receipt.json"
    path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({"status": "completed", "applied": apply, "groups": 6,
                      "files": 42, "added": len(installed), "receipt": str(path)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    run(args.output, args.apply)
