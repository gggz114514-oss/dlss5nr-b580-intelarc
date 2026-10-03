"""Review a pinned resource-pool trial and optionally install with rollback."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
DATA = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt")
WORKER = PROJECT / "artifacts/bridge-resource-pool-v1-20261001"
WEB = PROJECT / "artifacts/bridge-resource-pool-web-v1-20261002"
PLUGINS = Path(r"G:\epic\Cyberpunk2077\bin\x64\plugins")
RUNTIME = Path(r"G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime")
EXPECTED_ASI = "5a704ca4039e1a739eb0691fede12267d8cb39ad6b4485bf1067048b240ef6ff"
CURRENT_PINS = {
    "cyberpunk_nr_web.py": "8589fbb00318c7300f7edd5205de2647345d10356c841ed45024256fb6729093",
    "cyberpunk_nr_adapter.py": "e05c5749ff5c0978874ea3f2987469c70c0d3859650ac90012de2369f5209938",
    "nr_game_controls.py": "8e4a75d295eff4a184df539bda6c96f79415df275bc89c562deb0b25add38400",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def physical(path: Path, root: Path) -> None:
    if not path.resolve().is_relative_to(root.resolve()):
        raise RuntimeError("Path escaped target: " + str(path))
    for part in (path, *path.parents):
        if part.is_symlink() or part.is_junction():
            raise RuntimeError("Redirected path: " + str(part))


def closed() -> None:
    check = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "if (Get-Process -Name Cyberpunk2077,re8 -ErrorAction SilentlyContinue) { exit 11 } else { exit 0 }"],
        capture_output=True, text=True, check=False)
    if check.returncode:
        raise RuntimeError("Games must be closed before replacement")


def validate() -> tuple[list[tuple[Path, Path, Path]], dict]:
    manifest = read(WORKER / "source-manifest.json")
    build_path = DATA / "bridge-resource-pool-v1-20261001/checks/cpu-receipt.json"
    build = read(build_path)
    source = Path(build["source"])
    if source != WORKER / "payload" or build["source_freeze_sha256"] != manifest["source_freeze_sha256"]:
        raise RuntimeError("Different native build/source manifest")
    if (build["cpu_fake_lease_checks"], build["actual_deferred_control_checks"], build["targeted_ctest_passed"]) != (127, 8, 5):
        raise RuntimeError("Missing CPU lifetime/ABI checks")
    if not build["full_asi_link_verified"] or build["default_pool_enabled"] is not False:
        raise RuntimeError("Missing full build/default-off proof")
    for name, pin in manifest["source_pins"].items():
        physical(source / name, PROJECT)
        if sha(source / name) != pin:
            raise RuntimeError("Frozen worker source changed: " + name)
    binary = Path(build["cpu_binary_path"])
    physical(binary, DATA)
    if sha(binary) != build["cpu_binary_sha256"]:
        raise RuntimeError("Compiled ASI changed")
    probe_path = DATA / "bridge-resource-pool-byte-probe-v2-20261002/cpu-a39h3w6e/CPU_COMPILE_RECEIPT.json"
    probe = read(probe_path)
    for name, pin in probe["input_pins"].items():
        if sha(Path(name)) != pin:
            raise RuntimeError("Probe input changed: " + name)
    if sha(Path(probe["source"])) != probe["source_sha256"] or sha(Path(probe["exe"])) != probe["exe_sha256"]:
        raise RuntimeError("Compiled byte probe changed")
    gpu_path = DATA / "bridge-resource-pool-byte-probe-v2-20261002/gpu-byte-run01/gpu-byte-receipt.json"
    parent_path = DATA / "bridge-resource-pool-byte-probe-v2-20261002/supervisor-v1/gpu-byte-run01-parent-receipt.json"
    gpu, parent = read(gpu_path), read(parent_path)
    if not parent["completed"] or not parent["child_reaped"] or parent["watchdog_terminated"] or parent["exit_code"] != 0:
        raise RuntimeError("GPU child did not retire")
    if sha(gpu_path) != parent["probe_receipt_sha256"] or parent["input_pin_changes"]:
        raise RuntimeError("GPU receipt/source proof changed")
    for kind in ("source", "exe"):
        if parent[kind + "_sha256_before"] != probe[kind + "_sha256"] or parent[kind + "_sha256_after"] != probe[kind + "_sha256"]:
            raise RuntimeError("GPU-tested probe differs from CPU-built probe")
    expected = {"status": "passed_byte_fence_checks", "source_width": 1280, "source_height": 720,
                "byte_comparisons": 96, "pool_creates": 2, "pool_hits": 14,
                "same_canonical_resource_reuses": 14, "changed_binding_cases": 16,
                "off_on_off_cases": 16, "final_pool_enabled": False}
    for key, value in expected.items():
        if gpu.get(key) != value:
            raise RuntimeError("Missing byte/fence/counter gate: " + key)
    if gpu.get("debug_layer_available") is True:
        if gpu.get("debug_errors") != 0:
            raise RuntimeError("Debug layer reported errors")
    elif gpu.get("debug_layer_available") is not False or gpu.get("debug_errors") is not None:
        raise RuntimeError("Unavailable debug validation must remain explicitly null")
    samples = {(row["cache_on"], row["frame"], row["channel"]): row for row in gpu["samples"]}
    channels = {"proxy_rgba32f": 14745600, "motion_rg16f": 3686400, "composite_r11g11b10": 3686400}
    if len(samples) != 48:
        raise RuntimeError("Incomplete recorded byte hashes")
    for frame in range(8):
        for channel, count in channels.items():
            off, on = (samples[(cache, frame, channel)] for cache in (False, True))
            if off["bytes"] != count or on["bytes"] != count or off["sha256"] != on["sha256"]:
                raise RuntimeError("Independent recorded hash comparison failed")
    stage_path = WEB / "STAGE_RECEIPT.json"
    web = read(stage_path)
    web_cpu_path = DATA / "bridge-resource-pool-web-v1-20261002/cpu-01/CPU_CHECKS.json"
    checks = read(web_cpu_path)
    if sha(stage_path) != checks["stage_receipt_sha256"] or checks["status"] != "cpu_ABI_protocol_passed":
        raise RuntimeError("Web protocol CPU proof missing")
    if web["default_pool_enabled"] is not False or checks["model_native_apply_calls"] != 0:
        raise RuntimeError("Independent default-off control required")
    for name, pin in web["payload_pins"].items():
        if sha(WEB / "payload/game" / name) != pin:
            raise RuntimeError("Web payload changed: " + name)
    if sha(Path(web["native_worker_manifest"])) != web["native_worker_manifest_sha256"]:
        raise RuntimeError("Web/native pool ABI source differs")
    shader_pin = manifest["source_pins"]["shaders/nr_hdr_proxy.hlsl"]
    shader = PLUGINS / "nr_hdr_proxy.hlsl"
    physical(shader, PLUGINS)
    if sha(shader) != shader_pin or parent["shader_sha256"] != shader_pin:
        raise RuntimeError("Game shader differs from tested shader")
    targets = [(binary, PLUGINS / "CyberpunkNRBridge.asi", PLUGINS),
               (WEB / "payload/game/cyberpunk_nr_web.py", PLUGINS / "cyberpunk_nr_web.py", PLUGINS),
               (WEB / "payload/game/cyberpunk_nr_adapter.py", PLUGINS / "cyberpunk_nr_adapter.py", PLUGINS),
               (WEB / "payload/game/nr_game_controls.py", RUNTIME / "game/nr_game_controls.py", RUNTIME)]
    for _, destination, root in targets:
        physical(destination, root)
        if not destination.is_file():
            raise RuntimeError("Installed file missing: " + str(destination))
    if sha(targets[0][1]) != EXPECTED_ASI:
        raise RuntimeError("Current ASI differs from accepted bridge-cache base")
    for _, target, _ in targets[1:]:
        if sha(target) != CURRENT_PINS[target.name]:
            raise RuntimeError("Current Python base changed: " + str(target))
    return targets, {"native_CPU_receipt": str(build_path), "GPU_receipt": str(gpu_path),
                     "web_CPU_receipt": str(web_cpu_path), "default_pool_enabled": False,
                     "GPU_scope": gpu["scope"], "debug_layer_available": gpu["debug_layer_available"],
                     "debug_errors": gpu["debug_errors"], "performance_measured": False,
                     "real_NR_XPU_and_game_verified": False}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    targets, proof = validate()
    closed()
    plan = [{"source": str(src), "target": str(dst), "old_sha256": sha(dst), "new_sha256": sha(src)}
            for src, dst, _ in targets]
    if not args.install:
        print(json.dumps({"status": "reviewed_dry_run", "files": plan, **proof}))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output_root = DATA / "bridge-resource-pool-trial-v1-20261002"
    backup = output_root / stamp
    physical(backup, output_root)
    backup.mkdir(parents=True, exist_ok=False)
    for index, (_, target, _) in enumerate(targets):
        saved = backup / f"{index}-{target.name}"
        saved.write_bytes(target.read_bytes())
        if sha(saved) != plan[index]["old_sha256"]:
            raise RuntimeError("Backup verification failed")
        plan[index]["backup"] = str(saved)
    receipt = {"status": "backed_up", "files": plan, **proof}
    receipt_path = backup / "installed.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    replaced = []
    try:
        closed()
        for index, (src, target, root) in enumerate(targets):
            if sha(target) != plan[index]["old_sha256"]:
                raise RuntimeError("Target changed after review")
            temp = target.with_name(target.name + ".pool-trial-" + stamp + ".tmp")
            physical(temp, root)
            with temp.open("xb") as file:
                file.write(src.read_bytes())
            if sha(temp) != plan[index]["new_sha256"]:
                raise RuntimeError("Staged write verification failed")
            os.replace(temp, target)
            replaced.append(index)
        for index, (_, target, _) in enumerate(targets):
            if sha(target) != plan[index]["new_sha256"]:
                raise RuntimeError("Installed verification failed")
    except Exception:
        for index in reversed(replaced):
            target, root = targets[index][1:]
            temp = target.with_name(target.name + ".pool-rollback-" + stamp + ".tmp")
            physical(temp, root)
            with temp.open("xb") as file:
                file.write(Path(plan[index]["backup"]).read_bytes())
            os.replace(temp, target)
        receipt["status"] = "rolled_back"
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        raise
    receipt["status"] = "installed_trial_default_off"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path)}))


if __name__ == "__main__":
    main()
