"""Install the verified default-OFF native/ASI/web set with exact backups and rollback."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path

from install_bridge_resource_pool_trial_v1 import (
    DATA, PLUGINS, PROJECT, RUNTIME, closed, physical, sha, validate as review_pool,
)
from review_gpu_handoff_product_probe_cpu import review as review_native


PRODUCT = PROJECT / "artifacts/gpu-handoff-product-v1-20261002"


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(value, message):
    if not value:
        raise ValueError(message)


def validate(args):
    # Recheck the accepted installed cache baseline and the underlying pool byte
    # proof before adding its independent, still-default-OFF product control.
    _, pool = review_pool()
    native = review_native(args.cpu_receipt, args.runtime_receipt, args.native_stage)
    require(native["actual_cleanup_proven"] is True, "Native GPU/graph protocol not accepted")
    coupling = read(args.coupling)
    require(coupling["status"] == "isolated_native_ASI_web_ABI_coupling_passed" and
            Path(coupling["native_stage"]) == args.native_stage and
            coupling["native_build_review"]["cpu_receipt_sha256"] == sha(args.cpu_receipt),
            "Wrong native/product coupling receipt")
    manifest = read(PRODUCT / "SOURCE_PINS.json")
    require(sha(PRODUCT / "SOURCE_PINS.json") == coupling["product_source_manifest_sha256"] and
            sha(PRODUCT / "SOURCE_FROZEN.json") == coupling["product_frozen_receipt_sha256"],
            "Product freeze changed since main review")
    for name, digest in manifest.items():
        require(sha(PRODUCT / "payload" / name) == digest, "Product source changed: " + name)
    frozen = read(PRODUCT / "SOURCE_FROZEN.json")
    cpu = read(frozen["CPU_receipt"])
    require(sha(Path(cpu["asi"])) == coupling["product_ASI_sha256"] == frozen["ASI_sha256"],
            "Product ASI does not match reviewed build")
    require(sha(Path(frozen["CPU_receipt"]).parent / "CPU_GATE_RECEIPT.json") ==
            coupling["product_CPU_gate_sha256"], "CPU gate changed")
    native_cpu = read(args.cpu_receipt)
    changes = [
        (Path(cpu["asi"]), PLUGINS / "CyberpunkNRBridge.asi", PLUGINS),
        (PRODUCT / "payload/game/cyberpunk_nr_web.py", PLUGINS / "cyberpunk_nr_web.py", PLUGINS),
        (PRODUCT / "payload/game/cyberpunk_nr_adapter.py", PLUGINS / "cyberpunk_nr_adapter.py", PLUGINS),
    ]
    for name in ("nr_game_controls.py", "nr_game_pre_xess_host.py", "nr_gpu_handoff_host_v1.py"):
        changes.append((PRODUCT / "payload/game" / name, RUNTIME / "game" / name, RUNTIME))
    changes.extend([
        (args.native_stage / "payload/game/nr_texture_bridge_v1.py", RUNTIME / "game/nr_texture_bridge_v1.py", RUNTIME),
        (Path(native_cpu["candidate_DLL"]), RUNTIME / "native/nr_texture_bridge_re8_v1.dll", RUNTIME),
    ])
    for src, dst, root in changes:
        physical(src, DATA if src.drive.upper() == "D:" else PROJECT)
        physical(dst, root)
        require(src.is_file(), "Candidate file missing: " + str(src))
        if dst.name == "nr_gpu_handoff_host_v1.py":
            require(not dst.exists(), "Unreviewed controller already installed")
        else:
            require(dst.is_file(), "Installed file missing: " + str(dst))
    host = RUNTIME / "game/nr_game_pre_xess_host.py"
    expected_host = read(PRODUCT / "BASELINE_PINS.json")["installed_host_sha256"]
    require(sha(host) == expected_host, "Installed host changed")
    require(sha(RUNTIME / "game/nr_texture_bridge_v1.py") ==
            native_cpu["source_pins"]["baseline/game/nr_texture_bridge_v1.py"], "Installed wrapper changed")
    require(sha(PLUGINS / "nr_hdr_proxy.hlsl") == manifest["shaders/nr_hdr_proxy.hlsl"],
            "Installed pixel shader differs")
    closed()
    return changes, {
        "native_protocol_review": native, "coupling_receipt": str(args.coupling),
        "coupling_receipt_sha256": sha(args.coupling), "underlying_pool_proof": pool,
        "GPU_handoff_default_requested": False, "resource_pool_default_enabled": False,
        "real_game_and_performance_accepted": False, "model_source_changed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-stage", type=Path, required=True)
    parser.add_argument("--cpu-receipt", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path, required=True)
    parser.add_argument("--coupling", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    args = parser.parse_args()
    changes, proof = validate(args)
    plan = [{"source": str(src), "target": str(dst), "old_sha256": sha(dst) if dst.exists() else None,
             "new_sha256": sha(src)} for src, dst, _ in changes]
    if not args.install:
        print(json.dumps({"status": "reviewed_dry_run", "files": plan, **proof}, ensure_ascii=False))
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup_root = DATA / "gpu-handoff-product-trial-v1-20261002"
    backup = backup_root / stamp
    physical(backup, backup_root)
    backup.mkdir(parents=True, exist_ok=False)
    for index, (_, dst, _) in enumerate(changes):
        saved = backup / f"{index}-{dst.name}"
        if dst.exists():
            saved.write_bytes(dst.read_bytes())
            require(sha(saved) == plan[index]["old_sha256"], "Backup verification failed")
            plan[index]["backup"] = str(saved)
        else:
            plan[index]["backup"] = None
    receipt_path = backup / "installed.json"
    receipt = {"status": "backed_up", "files": plan, **proof}

    def save():
        receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    save()
    replaced, staged = [], []
    try:
        closed()
        for index, (src, dst, root) in enumerate(changes):
            require((sha(dst) if dst.exists() else None) == plan[index]["old_sha256"], "Target changed after review")
            temp = dst.with_name(dst.name + ".handoff-trial-" + stamp + ".tmp")
            physical(temp, root)
            with temp.open("xb") as stream:
                stream.write(src.read_bytes())
            staged.append(temp)
            require(sha(temp) == plan[index]["new_sha256"], "Staged write changed")
            os.replace(temp, dst)
            replaced.append(index)
        for index, (_, dst, _) in enumerate(changes):
            require(sha(dst) == plan[index]["new_sha256"], "Installed file hash mismatch")
    except BaseException:
        closed()
        for index in reversed(replaced):
            dst, root = changes[index][1:]
            require(sha(dst) == plan[index]["new_sha256"], "Rollback target changed externally")
            if plan[index]["backup"] is None:
                physical(dst, root)
                dst.unlink()  # exact task-created file; no recursive cleanup
            else:
                temp = dst.with_name(dst.name + ".handoff-rollback-" + stamp + ".tmp")
                physical(temp, root)
                with temp.open("xb") as stream:
                    stream.write(Path(plan[index]["backup"]).read_bytes())
                os.replace(temp, dst)
        for temp in staged:
            if temp.exists():
                physical(temp, changes[0][2] if temp.is_relative_to(PLUGINS) else RUNTIME)
                temp.unlink()
        receipt["status"] = "rolled_back"
        save()
        raise
    receipt["status"] = "installed_trial_default_off"
    save()
    print(json.dumps({"status": receipt["status"], "receipt": str(receipt_path), "files": len(plan)}))


if __name__ == "__main__":
    main()
