"""Bounded stdlib-only checks for canonical installed cache JSON reuse.

Only the real PT7NM G/D groups are read. All staging and mutation fixtures are
fresh D-owned outputs; this script never exports a smoke or installs a cache.
Run with Python -I -B -S to avoid site/device imports and bytecode writes.
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import traceback


WORKTREE = Path(r"E:\ComfyUI-aki-v3-IntelArc_20260722\.codex-worktrees\re8-fp8-unround-fast-20260928")
EXPORTER = WORKTREE / "tools" / "export_numeric_game_cache_720_v1.py"
TASK = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930")
DEFAULT_OUTPUT = TASK / "live-web-perf-20261001" / "canonical-installed-reuse-cpu-v1-20261001"
REAL_KEY = "PT7NM73JSFCR5URMFAMCNI3VK3PZHUWVWHIJGYLSE6WX33FBTXFQ"
REAL_SOURCE = TASK / "cache-prepare-slow-v1-20261001" / "cache" / REAL_KEY
REAL_TARGET = Path(r"G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime\data\fast-cache") / REAL_KEY
METADATA = "_project.json"
MARKER = "__grp___project.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def fingerprint(module, group):
    module.reject_links(group)
    return {path.name: module.sha256_file(path)
            for path in module._safe_group_files(group, group.parent)}


def plan(module, source, target):
    metadata = json.loads((source / METADATA).read_bytes())
    marker = json.loads((source / MARKER).read_bytes())
    record = module.CompileRecord(
        candidate="cpu_check_only", result_path=Path(__file__).resolve(),
        record_path="CPU fixture or read-only PT7NM group; not a GPU RESULT",
        module="batched_branched_mlp_v1", qualname="_project",
        kernel_hash=metadata["hash"],
        cache_files={path: module.sha256_file(Path(path)) for path in marker["child_paths"].values()},
        reported_hashes=False,
        binary_hashes={module.sha256_file(source / "_project.spv")})
    return module.validate_and_plan_record(record, source.parent, target.parent,
                                           approved_roots=(source.parent,))


def stage(module, source, target, output):
    group = plan(module, source, target)
    staged = output / source.name
    staged.mkdir(parents=True)
    module.rewrite_target_jsons(group, target.parent, staged, published_dir=output)
    return group, staged


def assert_reused(module, group, staged, target):
    proof = group["canonical_installed_reuse"]
    assert proof["status"] == "verified"
    assert proof["target_written"] is False
    for entry in group["files"]:
        name = entry["relative_path"]
        assert (staged / name).read_bytes() == (target / name).read_bytes(), name
        assert entry["staged_sha256"] == entry["target_sha256"] == module.sha256_file(target / name)
        if name in (METADATA, MARKER):
            assert entry["canonical_installed_reuse"] is True
            assert entry["rewritten_for_target"] is False
    return proof


def fixture(module, root, *, change=None, incomplete=False):
    kernel_hash = sha(b"canonical-installed-reuse-cpu-fixture-only")
    key = base64.b32encode(bytes.fromhex(kernel_hash)).decode().rstrip("=")
    source, target = root / "source" / key, root / "target" / key
    source.mkdir(parents=True)
    target.mkdir(parents=True)
    artifacts = {
        "_project.source": b"CPU fixture source; never compiled\n",
        "_project.ttir": b"CPU fixture TTIR\n",
        "_project.ttgir": b"CPU fixture TTGIR\n",
        "_project.llir": b"CPU fixture LLIR\n",
        "_project.spv": b"CPU fixture binary; never loaded\n",
    }
    metadata = {"hash": kernel_hash, "name": "_project", "num_warps": 4,
                "num_stages": 1, "target": {"backend": "xpu", "arch": "cpu_fixture_only"},
                "debug": True, "cache_dir": str(source)}
    marker = {"child_paths": {name: str(source / name) for name in (*artifacts, METADATA)},
              "receipt": {"version": 1}}
    for name, data in artifacts.items():
        (source / name).write_bytes(data)
    (source / METADATA).write_text(json.dumps(metadata, separators=(",", ":")) + "\n", encoding="utf-8")
    (source / MARKER).write_text(json.dumps(marker, separators=(",", ":")) + "\n", encoding="utf-8")
    installed_metadata = {**metadata, "cache_dir": str(target)}
    installed_marker = {**marker, "child_paths": {name: str(target / name) for name in marker["child_paths"]}}
    if change == "metadata":
        installed_metadata["num_warps"] = 8
    elif change == "metadata_type":
        installed_metadata["debug"] = 1  # Python True == 1 must not establish JSON identity.
    elif change == "compiler_hash":
        installed_metadata["hash"] = sha(b"different compiler identity")
    elif change == "marker":
        installed_marker["receipt"] = {"version": 2}
    elif change == "child_path":
        installed_marker["child_paths"]["_project.spv"] = str(source / "_project.spv")
    for name, data in artifacts.items():
        (target / name).write_bytes(data + (b"changed" if change == "binary" and name.endswith(".spv") else b""))
    if not incomplete:
        (target / METADATA).write_text(json.dumps(installed_metadata, indent=2, sort_keys=True), encoding="utf-8")
        (target / MARKER).write_text(json.dumps(installed_marker, indent=2, sort_keys=True), encoding="utf-8")
    return source, target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    spec = importlib.util.spec_from_file_location("_canonical_installed_export_cpu", EXPORTER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    output = args.output_dir.absolute()
    module.reject_links(output)
    if output.drive.casefold() != "d:" or not output.resolve().is_relative_to(TASK.resolve()):
        raise ValueError("CPU fixtures/receipts must stay inside the task's D root")
    if output.exists():
        raise FileExistsError(f"Preserving existing CPU check output: {output}")
    output.mkdir(parents=True)
    checks = []
    before = {"G_PT7NM": fingerprint(module, REAL_TARGET), "D_PT7NM": fingerprint(module, REAL_SOURCE)}
    report = {"schema": "numeric-game-canonical-installed-reuse-cpu-check-v1",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "real PT7NM only plus small D fixtures; no smoke/export/install/GPU run",
              "exporter": str(EXPORTER), "exporter_sha256": module.sha256_file(EXPORTER),
              "check_script_sha256": module.sha256_file(Path(__file__)),
              "real_source": str(REAL_SOURCE), "real_target": str(REAL_TARGET),
              "real_cache_before": before, "checks": checks,
              "G_written": False, "real_cache_written": False, "GPU_executed": False}
    try:
        group, staged = stage(module, REAL_SOURCE, REAL_TARGET, output / "real-PT7NM-staging")
        proof = assert_reused(module, group, staged, REAL_TARGET)
        report["real_canonical_installed_reuse"] = proof
        checks.append({"name": "real_PT7NM_preserves_all_G_bytes_and_SHA", "passed": True})
        source, target = fixture(module, output / "fixtures" / "equivalent")
        target_before = fingerprint(module, target)
        group, staged = stage(module, source, target, output / "fixtures" / "equivalent-staging")
        assert_reused(module, group, staged, target)
        assert fingerprint(module, target) == target_before
        checks.append({"name": "formatted_equivalent_JSON_reuses_original_target_bytes", "passed": True})

        for change in ("binary", "metadata", "metadata_type", "compiler_hash", "marker", "child_path"):
            case = output / "fixtures" / change
            source, target = fixture(module, case, change=change)
            target_before = fingerprint(module, target)
            try:
                stage(module, source, target, case / "staging")
            except module.ExportError as exc:
                error = str(exc)
            else:
                raise AssertionError(f"Substantive {change} difference was not rejected")
            assert fingerprint(module, target) == target_before
            checks.append({"name": f"reject_{change}_change", "passed": True, "error": error})

        case = output / "fixtures" / "incomplete"
        source, target = fixture(module, case, incomplete=True)
        target_before = fingerprint(module, target)
        group, staged = stage(module, source, target, case / "staging")
        assert "canonical_installed_reuse" not in group
        assert fingerprint(module, target) == target_before
        assert json.loads((staged / METADATA).read_bytes())["cache_dir"] == str(target)
        assert all(not entry["canonical_installed_reuse"] for entry in group["files"])
        checks.append({"name": "incomplete_target_uses_normal_export_without_target_writes", "passed": True})

        case = output / "fixtures" / "incomplete-conflict"
        source, target = fixture(module, case, incomplete=True, change="binary")
        target_before = fingerprint(module, target)
        try:
            stage(module, source, target, case / "staging")
        except module.ExportError as exc:
            error = str(exc)
        else:
            raise AssertionError("Incomplete target's existing byte conflict was not rejected")
        assert fingerprint(module, target) == target_before
        checks.append({"name": "incomplete_target_still_rejects_existing_conflict", "passed": True, "error": error})
        assert not any(name.split(".", 1)[0] in ("torch", "triton") for name in sys.modules)
        report["torch_or_triton_imported"] = False
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    finally:
        after = {"G_PT7NM": fingerprint(module, REAL_TARGET), "D_PT7NM": fingerprint(module, REAL_SOURCE)}
        report["real_cache_after"] = after
        report["real_cache_unchanged"] = before == after
        if before != after:
            report["status"] = "failed"
            report["error"] = "Real PT7NM cache inputs changed during the CPU check"
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report["passed_checks"] = sum(row["passed"] is True for row in checks)
        receipt = output / "RESULT.json"
        receipt.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "passed_checks": report["passed_checks"],
                      "real_cache_unchanged": report["real_cache_unchanged"], "receipt": str(receipt)}, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
