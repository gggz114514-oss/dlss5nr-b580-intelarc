"""Repair the actual fast-overlay bridge import, retaining the original modules.

The earlier trial updated game/nr_texture_bridge_v1.py, but activate() puts the
fast overlay's modules ahead of game/. Install the SAME protocol-tested wrapper
at that one shadowing path; do not reorder the rest of the fast backend.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import importlib.machinery
import json
from pathlib import Path
import subprocess


EXPECTED_SOURCE = "8a0cf609b9b7501145a9deac9b4c53dc5dda65ff08033304544ae132dc7e2e84"
EXPECTED_OLD = "a3c3269aceb9c6f47cd5cb5e2610c0521b2160951d02973fe18ec7680ef98c96"


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def physical(path):
    for part in (path, *path.parents):
        if part.is_symlink() or part.is_junction():
            raise ValueError("Redirected task path: " + str(part))


def tree(path):
    return ast.parse(path.read_text(encoding="utf-8-sig"))


def node(module, name):
    return next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == name)


def legacy_specs(module):
    bind = next(n for n in node(module, "TextureBridge").body
                if isinstance(n, ast.FunctionDef) and n.name == "_bind")
    return next(n.value for n in bind.body if isinstance(n, ast.Assign) and
                any(isinstance(t, ast.Name) and t.id == "specs" for t in n.targets))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("check", "install"))
    parser.add_argument("--installation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    physical(args.installation)
    installed = read(args.installation)
    if installed.get("status") != "installed_trial_default_off" or len(installed["files"]) != 8:
        raise ValueError("Repair the original reviewed eight-file installation only")
    for row in installed["files"]:
        physical(Path(row["target"]))
        if sha(Path(row["target"])) != row["new_sha256"]:
            raise ValueError("Installed file changed: " + row["target"])
    source = Path(installed["files"][6]["target"])
    runtime = source.parent.parent
    overlay = runtime / "experimental/fp8_unround_overlay"
    target = overlay / "modules/nr_texture_bridge_v1.py"
    physical(source); physical(target); physical(overlay / "bootstrap.py")
    paths = (overlay, overlay / "modules", runtime / "game", runtime / "toolchain",
             runtime / "fast/backend", runtime / "modules", runtime)
    spec = importlib.machinery.PathFinder.find_spec("nr_texture_bridge_v1", [str(p) for p in paths])
    if spec is None or Path(spec.origin).resolve() != target.resolve():
        raise ValueError("The observed fast-overlay import no longer resolves to the old wrapper")
    if sha(source) != EXPECTED_SOURCE or sha(target) != EXPECTED_OLD:
        raise ValueError("Protocol-tested source or observed old wrapper differs")
    current, candidate = tree(target), tree(source)
    for name in ("SourceFrame", "TextureOutput"):
        if ast.dump(node(current, name)) != ast.dump(node(candidate, name)):
            raise ValueError("Original frame/output data contract differs")
    if ast.dump(legacy_specs(current)) != ast.dump(legacy_specs(candidate)):
        raise ValueError("The original texture ABI differs")
    methods = {n.name for n in node(candidate, "TextureBridge").body if isinstance(n, ast.FunctionDef)}
    if not {"configure_gpu_handoff", "gpu_handoff_info", "poll_idle"} <= methods:
        raise ValueError("Actual wrapper must expose the owning-host optional seam")
    compile(source.read_bytes(), str(target), "exec", dont_inherit=True)
    output = args.output.resolve()
    physical(output)
    if output.drive.casefold() != "d:":
        raise ValueError("Keep task evidence and backups on D")
    output.mkdir(parents=True, exist_ok=False)
    plan = {"status": "actual_import_repair_checked", "source": str(source),
            "actual_module_target": str(target), "before_sha256": sha(target),
            "after_sha256": sha(source), "bootstrap_sha256": sha(overlay / "bootstrap.py"),
            "original_installation": str(args.installation),
            "original_installation_sha256": sha(args.installation),
            "SourceFrame_TextureOutput_and_legacy_ABI_unchanged": True,
            "global_import_order_changed": False, "original_modules_changed": False,
            "network_or_GPU_executed": False, "game_or_performance_accepted": False}
    (output / "PLAN.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    if args.mode == "check":
        print(json.dumps({"status": plan["status"], "target": str(target), "G_writes": False}))
        return
    process = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True,
                             text=True, timeout=5, check=True)
    if any(line.lower().startswith(('"cyberpunk2077.exe"', '"re8.exe"')) for line in process.stdout.splitlines()):
        raise ValueError("Exit both games before replacing a loaded wrapper")
    before = target.read_bytes()
    backup = output / "old-actual-overlay-wrapper.py"
    backup.write_bytes(before)
    if sha(target) != EXPECTED_OLD:
        raise ValueError("The actual wrapper changed after backup")
    try:
        target.write_bytes(source.read_bytes())
        if sha(target) != EXPECTED_SOURCE:
            raise ValueError("Actual-wrapper write mismatch")
    except BaseException:
        target.write_bytes(before)
        raise
    installed["files"].append({"source": str(source), "target": str(target),
        "old_sha256": EXPECTED_OLD, "new_sha256": EXPECTED_SOURCE, "backup": str(backup)})
    installed["actual_import_repair"] = plan
    installed["repair_installed_utc"] = datetime.now(timezone.utc).isoformat()
    receipt = output / "installed.json"
    receipt.write_text(json.dumps(installed, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "actual_import_repaired", "G_files_changed": 1,
                      "full_trial_files": 9, "receipt": str(receipt)}))


if __name__ == "__main__":
    main()
