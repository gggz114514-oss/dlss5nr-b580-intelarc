"""Prepare, disk-only verify, and additively install the frozen 720p slow profiles.

This wrapper reuses the frozen full-model smoke runner.  It only changes the
registry modes in memory so each named profile is tested as the four-flag
reference plus that one profile; no production source or runtime setting is
modified by this file.
"""
from __future__ import annotations

import argparse
import base64
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import uuid


PROJECT = Path(__file__).resolve().parents[1]
WORKTREE = Path(r"E:\ComfyUI-aki-v3-IntelArc_20260722\.codex-worktrees\re8-fp8-unround-fast-20260928")
RUNNER = WORKTREE / "tools" / "smoke_numeric_game_all_720_v1.py"
PROFILE_SOURCE = WORKTREE / "game" / "numeric_game_profiles_720_v1.py"
TASK = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930")
EXPERIMENTS = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt")
BASE_EXPORT = TASK / "cache-smoke-all-v5-20260930"
BASE_EXPORT_MANIFEST = BASE_EXPORT / "manifest.json"
RTZ_EXPORT = TASK / "cache-smoke-rtz-exact-v6-20261001"
RTZ_EXPORT_MANIFEST = RTZ_EXPORT / "manifest.json"
PREP_ROOT = TASK / "cache-prepare-slow-v1-20261001"
PREP_CACHE = PREP_ROOT / "cache"
PREP_MANIFEST = PREP_ROOT / "cache-manifest.json"
PREP_RECEIPT = PREP_ROOT / "cache-snapshot-receipt.json"
SOURCE_FREEZE = TASK / "source-freeze-validation-flash-v8-20261001.json"
SOURCE_INSTALL = TASK / "source-installed-validation-flash-v8-20261001.json"
CACHE_INSTALL = TASK / "rtz-cache-installed-exact-v7-20261001.json"
RTZ_RECEIPT = TASK / "native-rtz-v2-20260930" / "RTZ_REVIEWED_RECEIPT_v3.json"
GAME_CACHE = Path(r"G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime\data\fast-cache")
EXPORTER = WORKTREE / "tools" / "export_numeric_game_cache_720_v1.py"

REFERENCE = (
    "c512_k8_decoder",
    "c512_k8_c32_native",
    "num_history_fractional",
    "num_front_both",
)
SLOW = (
    "c512_k8_merge_fma", "c512_k8_post_fma16", "num_decoder_full_k",
    "num_branch_c64", "num_branch_c128", "num_branch_c256",
    "num_vit_denominator_ordered", "num_vit_denominator_fp32",
    "num_vit_norm_fma", "num_vit_exp_fma", "num_vit_norm_exp_fma",
    "num_vit_qkv_full_k", "num_vit_projection_full_k", "num_vit_exp_zero_constant",
    "num_history_dimension_rcp", "num_history_direct_pixel", "num_history_reciprocal",
    "num_post_sigmoid", "num_post_rne", "num_post_rtz",
)
HISTORY_BASE = ("c512_k8_decoder", "c512_k8_c32_native", "num_front_both")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def assert_no_reparse(path: Path, stop: Path | None = None) -> None:
    path = Path(path).absolute()
    stop = Path(stop).absolute() if stop is not None else None
    current = path
    while True:
        if current.exists() and (current.is_symlink() or
                                 bool(getattr(current, "is_junction", lambda: False)())):
            raise RuntimeError(f"Refusing linked/junction path: {current}")
        if stop is not None and current == stop:
            return
        parent = current.parent
        if parent == current:
            if stop is not None:
                raise RuntimeError(f"Path is not below required root {stop}: {path}")
            return
        current = parent


def load_registry():
    spec = importlib.util.spec_from_file_location("_slow_prepare_profile_registry", PROFILE_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(PROFILE_SOURCE)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolved_modes(registry) -> dict[str, dict]:
    if len(SLOW) != 20 or len(set(SLOW)) != 20 or len(set(REFERENCE)) != 4:
        raise RuntimeError("Unexpected profile list dimensions")
    unknown = (set(SLOW) | set(REFERENCE)) - set(registry.PROFILES)
    if unknown:
        raise RuntimeError(f"Profiles missing from frozen registry: {sorted(unknown)}")
    modes = {key: registry.combined_mode_options([*REFERENCE, key]) for key in SLOW}
    for key, mode in modes.items():
        numeric = mode.get("numeric_cleanup_720", {})
        required = (
            mode.get("c512_qkv_library_720") is True,
            mode.get("native_k8_720") is True,
            mode.get("decoder_gather_unround_720") is True,
            mode.get("c32_hidden_native_720") is True,
            numeric.get("history_value") == "fp32_fractional",
            numeric.get("front_noise") == "native_both",
        )
        if not all(required):
            raise RuntimeError(f"Reference flags did not survive combination for {key}: {mode}")
    return modes


def history_matrix(registry) -> list[dict]:
    values = (("reference", None), ("fp32_fractional", "num_history_fractional"),
              ("fp32_all_paths", "num_history_all_paths"))
    coordinates = (("reference", None), ("dimension_rcp", "num_history_dimension_rcp"),
                   ("direct_pixel", "num_history_direct_pixel"))
    reciprocals = (("table", None), ("native", "num_history_reciprocal"))
    rows = []
    for value_name, value_key in values:
        for coordinate_name, coordinate_key in coordinates:
            for reciprocal_name, reciprocal_key in reciprocals:
                group = [*HISTORY_BASE]
                group.extend(key for key in (value_key, coordinate_key, reciprocal_key) if key)
                mode = registry.combined_mode_options(group)
                numeric = mode.get("numeric_cleanup_720", {})
                expected_value = None if value_name == "reference" else value_name
                if numeric.get("history_value") != expected_value:
                    raise RuntimeError(f"History value mapping mismatch: {group} -> {numeric}")
                actual_coordinate = ("direct_pixel" if numeric.get("history_coord") == "direct_pixel"
                                     else "dimension_rcp" if numeric.get("history_dimension_rcp") == "native"
                                     else "reference")
                if actual_coordinate != coordinate_name:
                    raise RuntimeError(f"History coordinate mapping mismatch: {group} -> {numeric}")
                if numeric.get("history_reciprocal", "table") != reciprocal_name:
                    raise RuntimeError(f"History reciprocal mapping mismatch: {group} -> {numeric}")
                rows.append({"history_value": value_name, "history_coord": coordinate_name,
                             "history_reciprocal": reciprocal_name,
                             "profiles": group, "mode_options": mode})
    if len(rows) != 18 or len({tuple(row["profiles"]) for row in rows}) != 18:
        raise RuntimeError("History local matrix must contain 18 distinct compatible combinations")
    return rows


def self_test() -> dict:
    for path in (RUNNER, PROFILE_SOURCE, BASE_EXPORT_MANIFEST, RTZ_EXPORT_MANIFEST,
                 SOURCE_FREEZE, SOURCE_INSTALL, CACHE_INSTALL, RTZ_RECEIPT):
        if not path.is_file():
            raise FileNotFoundError(path)
    registry = load_registry()
    modes = resolved_modes(registry)
    history_rows = history_matrix(registry)
    base_manifest = read_json(BASE_EXPORT_MANIFEST)
    rtz_manifest = read_json(RTZ_EXPORT_MANIFEST)
    if base_manifest.get("selection") != "all-validated" or rtz_manifest.get("selection") != "all-validated":
        raise RuntimeError("Cache source manifests must both be all-validated exports")
    freeze = read_json(SOURCE_FREEZE)
    if (freeze.get("schema") != "numeric-game-source-freeze-v1" or
            freeze.get("all_candidates") is not True or freeze.get("gpu_run") is not False):
        raise RuntimeError("Unexpected source-freeze schema or scope")
    return {
        "status": "cpu-preflight-passed",
        "game_processes": "verified separately by tasklist and again before each GPU run",
        "profile_count": len(SLOW),
        "history_matrix_count": len(history_rows),
        "history_matrix": history_rows,
        "reference": list(REFERENCE),
        "profiles": list(SLOW),
        "resolved_mode_options": modes,
        "base_cache_groups": len(base_manifest.get("groups", [])),
        "rtz_cache_groups": len(rtz_manifest.get("groups", [])),
        "source_freeze_sha256": sha256_file(SOURCE_FREEZE),
        "runner_sha256": sha256_file(RUNNER),
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "gpu_started": False,
    }


def _remap_group_paths(groups: list[dict], old_root: Path, new_root: Path) -> None:
    for group in groups:
        for item in group.get("files", []):
            staged = Path(item["staged_path"]).resolve(strict=True)
            if not staged.is_relative_to(old_root):
                raise RuntimeError(f"Exported cache path escapes its staging root: {staged}")
            item["staged_path"] = str(new_root / staged.relative_to(old_root))


def materialize_cache_snapshot() -> dict:
    if PREP_ROOT.exists():
        raise FileExistsError(f"Preserving existing cache snapshot: {PREP_ROOT}")
    base = read_json(BASE_EXPORT_MANIFEST)
    extra = read_json(RTZ_EXPORT_MANIFEST)
    base_root = Path(base["staging_dir"]).resolve(strict=True)
    extra_root = Path(extra["staging_dir"]).resolve(strict=True)
    assert_no_reparse(base_root, TASK)
    assert_no_reparse(extra_root, TASK)
    if base.get("selection") != "all-validated" or extra.get("selection") != "all-validated":
        raise RuntimeError("Only exact all-validated export staging roots may seed preparation")

    PREP_ROOT.mkdir(parents=True)
    assert_no_reparse(PREP_ROOT, TASK)
    shutil.copytree(base_root, PREP_CACHE,
                    ignore=lambda directory, names: {"manifest.json"} if Path(directory).resolve() == base_root else set())
    overlay_rows = []
    for group in extra.get("groups", []):
        for item in group.get("files", []):
            source = Path(item["staged_path"]).resolve(strict=True)
            if not source.is_relative_to(extra_root):
                raise RuntimeError(f"RTZ artifact escapes its staging root: {source}")
            expected = item.get("staged_sha256")
            if not isinstance(expected, str) or sha256_file(source) != expected.lower():
                raise RuntimeError(f"RTZ staged artifact SHA mismatch: {source}")
            target = PREP_CACHE / source.relative_to(extra_root)
            assert_no_reparse(target, PREP_ROOT)
            if target.exists():
                if sha256_file(target) != expected.lower():
                    raise RuntimeError(f"Conflicting source cache bytes in isolated snapshot: {target}")
                action = "already-identical"
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                if target.name.startswith("__grp__"):
                    pass
                shutil.copy2(source, target)
                if sha256_file(target) != expected.lower():
                    raise RuntimeError(f"RTZ overlay copy SHA mismatch: {target}")
                action = "copied"
            overlay_rows.append({"source": str(source), "target": str(target),
                                 "sha256": expected.lower(), "action": action})

    derived = copy.deepcopy(base)
    derived["staging_dir"] = str(PREP_CACHE.resolve())
    _remap_group_paths(derived.get("groups", []), base_root, PREP_CACHE.resolve())
    derived["slow_prepare_overlay"] = {
        "source_manifest": str(RTZ_EXPORT_MANIFEST.resolve()),
        "source_manifest_sha256": sha256_file(RTZ_EXPORT_MANIFEST),
        "added_files": overlay_rows,
    }
    write_json(PREP_MANIFEST, derived)
    cache_before = 0
    for group in derived.get("groups", []):
        for item in group.get("files", []):
            staged = Path(item["staged_path"])
            expected = item.get("staged_sha256")
            if (not staged.is_relative_to(PREP_CACHE.resolve()) or
                    not expected or sha256_file(staged) != expected.lower()):
                raise RuntimeError(f"Isolated seed manifest validation failed: {staged}")
            cache_before += 1
    receipt = {
        "schema": "numeric-game-slow-cache-snapshot-v1",
        "status": "ready",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "base_manifest": str(BASE_EXPORT_MANIFEST.resolve()),
        "base_manifest_sha256": sha256_file(BASE_EXPORT_MANIFEST),
        "base_source_files_verified": cache_before,
        "rtz_overlay_files": len(overlay_rows),
        "rtz_overlay_added": sum(row["action"] == "copied" for row in overlay_rows),
        "cache_dir": str(PREP_CACHE.resolve()),
        "cache_manifest": str(PREP_MANIFEST.resolve()),
        "cache_manifest_sha256": sha256_file(PREP_MANIFEST),
        "cache_file_count": sum(1 for p in PREP_CACHE.rglob("*") if p.is_file()),
        "source_or_G_cache_changed": False,
    }
    write_json(PREP_RECEIPT, receipt)
    return receipt


class _ResolvedProfile:
    def __init__(self, original, mode):
        self._original = original
        self._mode = mode

    def __getattr__(self, name):
        return getattr(self._original, name)

    def mode_options(self):
        return copy.deepcopy(self._mode)


def assert_game_idle() -> str:
    result = subprocess.run(["tasklist", "/FI", "IMAGENAME eq Cyberpunk2077.exe"],
                            capture_output=True, text=True, check=True)
    output = (result.stdout or "") + (result.stderr or "")
    if "No tasks are running which match the specified criteria" not in output:
        raise RuntimeError("Cyberpunk is not idle; GPU preparation is forbidden:\n" + output)
    return output.strip()


def seed_exact_installed_group(key: str, group_dir: Path, metadata_name: str,
                               *, byte_budget: int = 512 * 1024 * 1024):
    """Reuse an exact installed artifact in isolated preparation, never in readonly.

    The compiler key already covers source, specialization and compiler options.
    This copies that same key; it does not compile a baseline or select another
    implementation. Existing different D files and all G files are preserved.
    """
    if len(key) != 64 or any(c not in '0123456789abcdef' for c in key):
        raise ValueError('Unexpected compiler cache key: ' + key)
    if Path(metadata_name).name != metadata_name or not metadata_name.endswith('.json'):
        raise ValueError('Unexpected kernel metadata name: ' + metadata_name)
    folder = base64.b32encode(bytes.fromhex(key)).decode().rstrip('=')
    expected_dir = (PREP_CACHE / folder).resolve()
    if group_dir.resolve() != expected_dir:
        raise RuntimeError('Installed seeding is limited to the isolated D cache')
    assert_no_reparse(expected_dir, TASK)
    source_dir = GAME_CACHE / folder
    assert_no_reparse(source_dir, GAME_CACHE)
    marker = source_dir / ('__grp__' + metadata_name)
    if not marker.is_file():
        return None
    marker_bytes = marker.read_bytes()
    children = json.loads(marker_bytes).get('child_paths')
    if not isinstance(children, dict) or not children or metadata_name not in children:
        raise RuntimeError('Incomplete installed cache group: ' + str(marker))
    plans, child_paths = [], {}
    for name, prior_path in children.items():
        if Path(name).name != name:
            raise RuntimeError('Installed child escapes its group: ' + name)
        source = source_dir / name
        assert_no_reparse(source, GAME_CACHE)
        data = source.read_bytes()
        before = hashlib.sha256(data).hexdigest()
        prior = Path(prior_path)
        if prior.resolve() != source.resolve():
            # Old installed markers can refer to a prior D export. Accept only
            # an exact byte mirror, rather than silently rebinding stale data.
            if prior.drive.casefold() != 'd:' or prior.name != name or prior.parent.name != folder:
                raise RuntimeError('Unexpected cross-root installed child: ' + str(prior))
            assert_no_reparse(prior)
            if not prior.is_file() or sha256_file(prior) != before:
                raise RuntimeError('Installed cross-root child is not an exact mirror: ' + str(prior))
        if name == metadata_name:
            metadata = json.loads(data)
            if metadata.get('hash') != key:
                raise RuntimeError('Installed kernel metadata has a different compiler key')
            if 'cache_dir' in metadata:
                metadata['cache_dir'] = str(expected_dir)
            data = (json.dumps(metadata, ensure_ascii=False, separators=(',', ':')) + '\n').encode()
        target = expected_dir / name
        plans.append((source, target, before, data))
        child_paths[name] = str(target)
    marker_data = (json.dumps({'child_paths': child_paths}, separators=(',', ':')) + '\n').encode()
    plans.append((marker, expected_dir / marker.name,
                  hashlib.sha256(marker_bytes).hexdigest(), marker_data))
    missing = []
    for source, target, before, data in plans:
        assert_no_reparse(target, TASK)
        if target.exists():
            if not target.is_file() or sha256_file(target) != hashlib.sha256(data).hexdigest():
                raise RuntimeError('Refusing to replace a different prepared artifact: ' + str(target))
        else:
            missing.append((source, target, before, data))
    total = sum(len(data) for _, _, _, data in missing)
    if total > byte_budget:
        raise RuntimeError('Exact installed cache seeding exceeds its remaining byte budget')
    expected_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    # The marker is last, so a partial copy cannot advertise a complete group.
    for source, target, before, data in missing:
        if sha256_file(source) != before:
            raise RuntimeError('Installed artifact changed during preparation: ' + str(source))
        with target.open('xb') as stream:
            stream.write(data)
        after = hashlib.sha256(data).hexdigest()
        if sha256_file(target) != after:
            raise RuntimeError('Prepared artifact copy failed SHA verification: ' + str(target))
        rows.append({'source': str(source), 'source_sha256': before,
                     'target': str(target), 'target_sha256': after})
    return {'key': key, 'metadata': metadata_name, 'group': str(expected_dir),
            'added_files': len(rows), 'added_bytes': total, 'files': rows,
            'G_written': False, 'kernel_or_options_changed': False}


def invoke_smoke(phase: str, output: Path) -> int:
    if phase not in {"prepare", "readonly", "verify-g"}:
        raise ValueError(phase)
    if not output.is_absolute() or not output.resolve().is_relative_to(TASK.resolve()):
        raise ValueError(f"Smoke output must be inside task D root: {output}")
    if phase == "prepare":
        if not PREP_CACHE.is_dir() or not PREP_MANIFEST.is_file():
            materialize_cache_snapshot()
        cache_dir = PREP_CACHE
    elif phase == "readonly":
        if not PREP_CACHE.is_dir() or not PREP_MANIFEST.is_file():
            raise FileNotFoundError("Run prepare phase before readonly verification")
        cache_dir = PREP_CACHE
    else:
        if not PREP_CACHE.is_dir() or not PREP_MANIFEST.is_file():
            raise FileNotFoundError("Prepared-cache manifest is required for G verification")
        cache_dir = GAME_CACHE

    game_idle = assert_game_idle()
    runner_tools = str((WORKTREE / "tools").resolve())
    if runner_tools not in sys.path:
        sys.path.insert(0, runner_tools)
    spec = importlib.util.spec_from_file_location("_slow_prepare_frozen_smoke_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise ImportError(RUNNER)
    smoke = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = smoke
    spec.loader.exec_module(smoke)
    modes = None
    original_loader = smoke._load_registry

    def load_patched_registry(source_hashes):
        registry = original_loader(source_hashes)
        nonlocal modes
        modes = resolved_modes(registry)
        profiles = dict(registry.PROFILES)
        for key, mode in modes.items():
            profiles[key] = _ResolvedProfile(profiles[key], mode)
        registry.PROFILES = profiles
        return registry

    smoke._load_registry = load_patched_registry
    def selected_history_combos(selected, run_rows, registry, *, prepare_cache):
        return [list(row["profiles"]) for row in history_matrix(registry)]
    smoke._eligible_combo_groups = selected_history_combos
    lookup_rows, installed_seed_rows = [], []
    original_cpu_loader = smoke._load_cpu_modules

    def observed_cpu_loader():
        support, suite, boundary = original_cpu_loader()
        original_runtime_loader = boundary.load_runtime

        def observed_runtime_loader(destination):
            env = original_runtime_loader(destination)
            # Task-process-only file-cache diagnostics. Do not alter the live
            # DiskOnly guard, missing-artifact policy, or any kernel source.
            from triton.runtime import cache as cache_module
            original_manager = cache_module.get_cache_manager

            def observed_manager(key):
                manager = original_manager(key)
                original_group = manager.get_group

                def observed_group(name):
                    result = original_group(name)
                    if not result and phase == 'prepare':
                        used = sum(row['added_bytes'] for row in installed_seed_rows)
                        seed = seed_exact_installed_group(
                            str(key), Path(manager.cache_dir), name,
                            byte_budget=512 * 1024 * 1024 - used)
                        if seed is not None:
                            installed_seed_rows.append(seed)
                            result = original_group(name)
                            if not result:
                                raise RuntimeError('Exact installed cache copy remains unreadable: ' + name)
                    if not result and len(lookup_rows) < 128:
                        lookup_rows.append({"hash": str(key), "kernel_metadata": name,
                                            "cache_dir": str(manager.cache_dir)})
                    return result

                manager.get_group = observed_group
                return manager

            cache_module.get_cache_manager = observed_manager
            return env

        boundary.load_runtime = observed_runtime_loader
        return support, suite, boundary

    smoke._load_cpu_modules = observed_cpu_loader
    args = [
        "--profiles", ",".join(SLOW),
        "--output", str(output.resolve()),
        "--source-freeze", str(SOURCE_FREEZE.resolve()),
        "--install-manifest", str(SOURCE_INSTALL.resolve()),
        "--cache-install-manifest", str(CACHE_INSTALL.resolve()),
        "--cache-manifest", str(PREP_MANIFEST.resolve()),
        "--cache-dir", str(cache_dir.resolve()),
        "--native-rtz-receipt", str(RTZ_RECEIPT.resolve()),
        "--gpu-ready",
    ]
    if phase == "prepare":
        args.append("--prepare-cache")
    invocation = {
        "schema": "numeric-game-slow-cache-invocation-v1",
        "phase": phase,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "output": str(output.resolve()),
        "cache_dir": str(cache_dir.resolve()),
        "cache_manifest_sha256": sha256_file(PREP_MANIFEST),
        "source_freeze_sha256": sha256_file(SOURCE_FREEZE),
        "runner_sha256": sha256_file(RUNNER),
        "driver_sha256": sha256_file(Path(__file__).resolve()),
        "game_tasklist": game_idle,
        "reference": list(REFERENCE),
        "profiles": list(SLOW),
        "resolved_mode_options": resolved_modes(load_registry()),
        "history_matrix": history_matrix(load_registry()),
        "threads": 14,
        "fresh_process_requirement": "one new GPU Python process per phase; caller launches this file anew",
    }
    invocation_path = output.with_name(output.name + ".driver-invocation.json")
    if invocation_path.exists():
        raise FileExistsError(invocation_path)
    write_json(invocation_path, invocation)
    try:
        code = smoke.main(args)
    finally:
        if output.is_dir():
            write_json(output / "CACHE_LOOKUP_DIAGNOSTICS.json", {
                "scope": "task-process cache lookup; prepare-only exact installed seeding; guard unchanged",
                "missing_groups": lookup_rows, "installed_seed_groups": installed_seed_rows})
    result_path = output.resolve() / "RESULT.json"
    report = read_json(result_path) if result_path.is_file() else {}
    write_json(output.resolve() / "DRIVER_RECEIPT.json", {
        **invocation,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "runner_return_code": code,
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path) if result_path.is_file() else None,
        "result_status": report.get("status"),
        "actual_gpu": report.get("actual_gpu"),
        "cache_fingerprint_equal": report.get("cache_fingerprint_equal"),
        "candidate_rows": len(report.get("profiles", [])),
        "passed_rows": sum(row.get("passed") is True for row in report.get("profiles", [])),
        "combo_rows": len(report.get("combos", [])),
    })
    return code


def export_cache(report_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Preserving existing cache export: {output_dir}")
    report = read_json(report_path)
    if report.get("status") != "passed" or report.get("cache_policy") != "readonly":
        raise RuntimeError("All 20 fresh-process readonly profile gates must pass before export")
    if (len(report.get("profiles", [])) != len(SLOW) or
            any(row.get("passed") is not True for row in report["profiles"]) or
            len(report.get("combos", [])) != 18 or
            any(row.get("passed") is not True for row in report["combos"])):
        raise RuntimeError("Readonly report does not contain 20 passed singles and 18 history combinations")
    command = [
        sys.executable, "-I", "-B", "-S", "-X", "utf8", str(EXPORTER),
        "--all-validated", "--smoke-report", str(report_path.resolve()),
        "--cache-source-root", str(PREP_CACHE.resolve()),
        "--cache-target-dir", str(GAME_CACHE),
        "--output-dir", str(output_dir.resolve()),
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    log_path = output_dir.with_name(output_dir.name + ".export-command.txt")
    write_json(log_path, {
        "command": command,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "report": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
    })
    if result.returncode != 0:
        raise RuntimeError(f"Cache exporter failed ({result.returncode}): {result.stdout}\n{result.stderr}")
    manifest_path = output_dir.resolve() / "manifest.json"
    manifest = read_json(manifest_path)
    totals = manifest.get("totals", {})
    candidates = manifest.get("candidate_results", {})
    expected_candidates = len(SLOW) + 18
    if (manifest.get("selection") != "all-validated" or len(candidates) != expected_candidates or
            totals.get("exported_candidate_count") != expected_candidates or
            totals.get("missing_candidates") or totals.get("missing_profiles")):
        raise RuntimeError(f"Export did not authenticate all 20 profiles: {totals}")
    return {"manifest": str(manifest_path), "manifest_sha256": sha256_file(manifest_path),
            "totals": totals, "candidate_results": candidates,
            "smoke_report": str(report_path.resolve()), "smoke_report_sha256": sha256_file(report_path)}


def install_additive(export_manifest_path: Path, smoke_report_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise FileExistsError(f"Preserving existing install receipt directory: {output_dir}")
    assert_no_reparse(GAME_CACHE)
    manifest_path = export_manifest_path.resolve(strict=True)
    manifest = read_json(manifest_path)
    smoke_report_path = smoke_report_path.resolve(strict=True)
    smoke = read_json(smoke_report_path)
    if (manifest.get("selection") != "all-validated" or
            manifest.get("cache_target_dir", "").casefold() != str(GAME_CACHE).casefold() or
            manifest.get("completed_smoke", {}).get("path", "").casefold() != str(smoke_report_path).casefold() or
            manifest.get("completed_smoke", {}).get("sha256") != sha256_file(smoke_report_path) or
            smoke.get("status") != "passed" or smoke.get("cache_policy") != "readonly"):
        raise RuntimeError("Export manifest is not tied to the exact completed readonly report and G target")
    totals = manifest.get("totals", {})
    candidates = manifest.get("candidate_results", {})
    if (len(candidates) != 38 or totals.get("exported_candidate_count") != 38 or
            totals.get("missing_candidates") or totals.get("missing_profiles")):
        raise RuntimeError("Refusing install unless all twenty profile cache receipts exported")
    export_root = Path(manifest.get("staging_dir", "")).resolve(strict=True)
    assert_no_reparse(export_root, TASK)
    marker_targets = set()
    plan_rows = {}
    for group in manifest.get("groups", []):
        marker = group.get("group_marker_relative_path")
        if isinstance(marker, str):
            marker_targets.add(str((GAME_CACHE / group["cache_key"] / marker).resolve(strict=False)).casefold())
        for item in group.get("files", []):
            staged = Path(item["staged_path"]).resolve(strict=True)
            target = Path(item["target_path"]).resolve(strict=False)
            expected = item.get("staged_sha256")
            target_expected = item.get("target_sha256")
            if (not staged.is_relative_to(export_root) or not target.is_relative_to(GAME_CACHE) or
                    not isinstance(expected, str) or expected != target_expected or
                    sha256_file(staged) != expected.lower()):
                raise RuntimeError(f"Export artifact identity/SHA/path check failed: {staged} -> {target}")
            key = str(target).casefold()
            prior = plan_rows.get(key)
            if prior and prior["sha256"] != expected.lower():
                raise RuntimeError(f"Two exported rows conflict at target: {target}")
            plan_rows[key] = {"staged": str(staged), "target": str(target), "sha256": expected.lower(),
                              "relative": str(target.relative_to(GAME_CACHE))}

    output_dir = output_dir.resolve()
    if not output_dir.is_relative_to(TASK.resolve()) or output_dir.drive.casefold() != "d:":
        raise ValueError(f"Install receipts must stay on task D root: {output_dir}")
    output_dir.mkdir(parents=True)
    before = []
    for row in plan_rows.values():
        target = Path(row["target"])
        assert_no_reparse(target, GAME_CACHE)
        if target.exists():
            current = sha256_file(target)
            if current != row["sha256"]:
                raise RuntimeError(f"Refusing to overwrite a different existing cache file: {target}")
            row["before"] = "identical"
            row["before_sha256"] = current
        else:
            row["before"] = "absent"
            row["before_sha256"] = None
        before.append(dict(row))
    plan = {
        "schema": "numeric-game-slow-cache-additive-install-v1",
        "status": "preflight-complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "export_manifest": str(manifest_path),
        "export_manifest_sha256": sha256_file(manifest_path),
        "smoke_report": str(smoke_report_path),
        "smoke_report_sha256": sha256_file(smoke_report_path),
        "target_root": str(GAME_CACHE),
        "candidate_count": len(candidates),
        "artifact_file_count": len(plan_rows),
        "new_file_count": sum(row["before"] == "absent" for row in before),
        "identical_existing_count": sum(row["before"] == "identical" for row in before),
        "source_or_kernel_changed": False,
        "overwrite_existing": False,
        "files": before,
    }
    write_json(output_dir / "install-plan.json", plan)
    markers = {value for value in marker_targets}
    ordered = sorted(plan_rows.values(), key=lambda row: (str(Path(row["target"])).casefold() in markers,
                                                           row["relative"].casefold()))
    installed = []
    try:
        for row in ordered:
            target, staged = Path(row["target"]), Path(row["staged"])
            if target.exists():
                if sha256_file(target) != row["sha256"]:
                    raise RuntimeError(f"Target changed after preflight; refusing overwrite: {target}")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            assert_no_reparse(target, GAME_CACHE)
            temp = target.with_name("." + target.name + ".luna-add-" + uuid.uuid4().hex)
            try:
                with temp.open("xb") as stream:
                    stream.write(staged.read_bytes())
                    stream.flush()
                    os.fsync(stream.fileno())
                if sha256_file(temp) != row["sha256"]:
                    raise RuntimeError(f"Temporary cache file SHA mismatch: {temp}")
                os.rename(temp, target)
            finally:
                if temp.exists():
                    temp.unlink()
            installed.append(row["relative"])
        for row in ordered:
            target = Path(row["target"])
            if not target.is_file() or sha256_file(target) != row["sha256"]:
                raise RuntimeError(f"Final installed cache SHA mismatch: {target}")
        receipt = {**plan, "status": "completed", "applied": True,
                   "installed_file_count": len(installed), "installed_paths": installed,
                   "finished_utc": datetime.now(timezone.utc).isoformat(),
                   "G_cache_changed_only_by_additions": True}
        write_json(output_dir / "completed-receipt.json", receipt)
        return receipt
    except BaseException as exc:
        write_json(output_dir / "failed-receipt.json", {
            **plan, "status": "incomplete_or_failed", "applied_paths": installed,
            "error": f"{type(exc).__name__}: {exc}",
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        })
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-test", help="CPU-only registry/manifest preflight")
    run = sub.add_parser("run", help="one fresh GPU Python process: prepare, readonly, or verify-g")
    run.add_argument("--phase", choices=("prepare", "readonly", "verify-g"), required=True)
    run.add_argument("--output", type=Path, required=True)
    export = sub.add_parser("export", help="CPU-only export of the completed 20-profile readonly smoke")
    export.add_argument("--report", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    install = sub.add_parser("install", help="SHA-verified additive install into the actual G cache")
    install.add_argument("--manifest", type=Path, required=True)
    install.add_argument("--report", type=Path, required=True)
    install.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "self-test":
            print(json.dumps(self_test(), ensure_ascii=False, indent=2))
            return 0
        if args.command == "run":
            if args.phase == "prepare":
                if PREP_ROOT.exists():
                    if not (PREP_CACHE.is_dir() and PREP_MANIFEST.is_file() and PREP_RECEIPT.is_file()):
                        raise RuntimeError("Preserving incomplete cache snapshot; inspect it before retrying")
                    receipt = read_json(PREP_RECEIPT)
                    if (receipt.get("cache_dir") != str(PREP_CACHE.resolve()) or
                            receipt.get("cache_manifest_sha256") != sha256_file(PREP_MANIFEST)):
                        raise RuntimeError("Existing prepared-cache manifest differs from its receipt")
                else:
                    receipt = materialize_cache_snapshot()
                print(json.dumps({"cache_snapshot": str(PREP_RECEIPT), **receipt}, ensure_ascii=False), flush=True)
            return invoke_smoke(args.phase, args.output)
        if args.command == "export":
            print(json.dumps(export_cache(args.report, args.output), ensure_ascii=False, indent=2))
            return 0
        if args.command == "install":
            receipt = install_additive(args.manifest, args.report, args.output)
            print(json.dumps({"status": receipt["status"],
                              "receipt": str(args.output.resolve() / "completed-receipt.json"),
                              "candidate_count": receipt["candidate_count"],
                              "artifact_file_count": receipt["artifact_file_count"],
                              "installed_file_count": receipt["installed_file_count"]}, ensure_ascii=False))
            return 0
        raise ValueError(args.command)
    except BaseException:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
