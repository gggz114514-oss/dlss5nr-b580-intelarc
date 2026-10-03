"""Compose the two reviewed math payloads on E, without importing GPU code.

This stages default-off options; it does not publish availability or install.
Worker source inventories are checked before and after the composition.
"""
from __future__ import annotations
import ast
import dataclasses
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
MLP = PROJECT / "artifacts/whole-mlp-native-v1-20261001"
VIT = PROJECT / "artifacts/vit-attention-native-v1-20261001"
STAGE = PROJECT / "artifacts/native-math-composed-v1-20261001"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("Reviewed merge anchor changed: " + old[:100])
    return text.replace(old, new)


def physical(path):
    if not path.resolve().is_relative_to(PROJECT.resolve()):
        raise RuntimeError("Path outside source project: " + str(path))
    for item in (path, *path.parents):
        if item.is_junction() or item.is_symlink():
            raise RuntimeError("Redirected source/stage: " + str(item))


def worker_inventory():
    for root in (MLP, VIT):
        physical(root)
        if not (root / "REPORT.md").is_file():
            raise RuntimeError("Reviewed worker report required: " + str(root))
    mlp = json.loads((MLP / "source-pins.json").read_text(encoding="utf-8"))
    vit = json.loads((VIT / "SOURCE_PINS.json").read_text(encoding="utf-8"))
    inventory = {}
    for row in mlp["payload_sources"]:
        path = MLP / row["path"]
        physical(path)
        if sha(path) != row["sha256"]:
            raise RuntimeError("Whole MLP payload drift: " + row["path"])
        inventory[str(path)] = row["sha256"]
    for name, digest in vit["payload_sha256"].items():
        path = VIT / name
        physical(path)
        if sha(path) != digest:
            raise RuntimeError("ViT payload drift: " + name)
        inventory[str(path)] = digest
    for root, name in ((MLP, "source-pins.json"), (VIT, "SOURCE_PINS.json")):
        inventory[str(root / name)] = sha(root / name)
    return inventory


def cpu_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
        return module
    finally:
        sys.modules.pop(name, None)


def main():
    inventory = worker_inventory()
    physical(STAGE)
    if STAGE.exists():
        raise RuntimeError("Fresh isolated math stage required")
    a = MLP / "original/game/numeric_cleanup_options_720_v1.py"
    b = VIT / "original/game/numeric_cleanup_options_720_v1.py"
    if sha(a) != sha(b):
        raise RuntimeError("Workers did not start from the same options source")
    profile = MLP / "payload/game/numeric_game_profiles_720_v1.py"
    if not profile.is_file() or str(profile) not in inventory:
        raise RuntimeError("Final whole MLP registry integration not delivered")
    game = STAGE / "payload/game"
    game.mkdir(parents=True)
    originals = STAGE / "original/game"
    originals.mkdir(parents=True)
    copied = {}
    for root, names in (
        (MLP, ("whole_mlp_native_720_v1.py", "whole_mlp_native_720_spec_v1.py",
               "whole_mlp_native_720_kernel_v1.py", "numeric_cleanup_suite_720_v1.py")),
        (VIT, ("vit_attention_native_720_v1.py", "vit_attention_native_720_v1_kernels.py",
               "vit_numeric_suite_720_v1.py", "numeric_model_forward_720_v1.py")),
    ):
        for name in names:
            source = root / "payload/game" / name
            if str(source) not in inventory:
                raise RuntimeError("Unpinned worker payload: " + name)
            (game / name).write_bytes(source.read_bytes())
            copied[name] = str(source)
    options = (MLP / "payload/game/numeric_cleanup_options_720_v1.py").read_text(encoding="utf-8-sig")
    # Keep all old positional dataclass fields, then append new options.
    whole_fields = ('    whole_mlp_families: tuple[str, ...] = ()\n'
                    '    whole_mlp_schedule: str = "wide_parts"\n')
    options = one(options, whole_fields, "")
    options = one(options, '    front_noise: str = "table"\n',
                  '    front_noise: str = "table"\n' + whole_fields +
                  '    vit_attention_native: bool = False\n')
    options = one(options, '                       "vit_norm_fma", "vit_exp_fma")',
                  '                       "vit_norm_fma", "vit_exp_fma", "vit_attention_native")')
    options = one(options, '            "vit_denominator", "vit_norm_fma", "vit_exp_fma")}',
                  '            "vit_denominator", "vit_norm_fma", "vit_exp_fma", "vit_attention_native")}')
    options = one(options, 'INDEPENDENT_CANDIDATES = {\n',
                  'INDEPENDENT_CANDIDATES = {\n'
                  '    "vit_attention_native": {"vit_attention_native": True},\n')
    (game / "numeric_cleanup_options_720_v1.py").write_text(options, encoding="utf-8", newline="\n")
    (originals / a.name).write_bytes(a.read_bytes())
    # Registry entries are merely pending; availability remains unpublished.
    registry = profile.read_text(encoding="utf-8-sig")
    registry = one(registry, 'PROFILES = MappingProxyType({\n',
                   'PROFILES = MappingProxyType({\n'
                   '    "num_vit_attention_native": GameProfile("ViT 注意力原生 FP16 矩阵",\n'
                   '        numeric=(("vit_attention_native", True),), pending_validation=True),\n')
    (game / profile.name).write_text(registry, encoding="utf-8", newline="\n")
    checks = []
    for path in game.glob("*.py"):
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
        checks.append("syntax " + path.name)
    old = cpu_module("native_math_old_options", a)
    merged = cpu_module("native_math_merged_options", game / a.name)
    old_fields = [x.name for x in dataclasses.fields(old.NumericCleanupOptions)]
    new_fields = [x.name for x in dataclasses.fields(merged.NumericCleanupOptions)]
    if new_fields[:len(old_fields)] != old_fields:
        raise RuntimeError("Old positional option fields changed")
    if merged.NumericCleanupOptions().active or merged.NumericCleanupOptions().selected_scopes():
        raise RuntimeError("Default-off math stage unexpectedly enabled a scope")
    combinations = 0
    for whole, schedule, native, denominator, projection in itertools.product(
            ((), ("c128",), ("c256",), ("c128", "c256")),
            ("wide_parts", "fused"), (False, True),
            ("reference", "ordered_fused", "fp32_reduction"),
            ((), ("c128",), ("c128", "c256"))):
        value = merged.NumericCleanupOptions(
            whole_mlp_families=whole, whole_mlp_schedule=schedule,
            vit_attention_native=native, vit_denominator=denominator,
            branch_accum_families=projection,
            history_value="fp32_fractional", front_noise="native_both")
        scopes = value.selected_scopes()
        branch = [s for s in scopes if s[0] == "branch_accum"]
        if len(branch) != int(bool(whole or projection)):
            raise RuntimeError("Duplicate/lost branch owner during combination")
        if whole and branch[0][1] != "whole_mlp_native_720_v1":
            raise RuntimeError("Whole MLP union owner lost")
        if whole and branch[0][2]["branch_accum"] != {n:n in projection for n in ("c64", "c128", "c256")}:
            raise RuntimeError("Existing projection choices were cleared")
        vit_scope = [s for s in scopes if s[0] == "vit"]
        if native and (len(vit_scope) != 1 or vit_scope[0][2]["vit_attention_native"] is not True):
            raise RuntimeError("Native attention scope lost")
        combinations += 1
    if any(sha(Path(name)) != digest for name, digest in inventory.items()):
        raise RuntimeError("Worker inventory changed during composition")
    pins = {path.relative_to(STAGE).as_posix():sha(path) for path in STAGE.rglob("*.py")}
    result = {"status":"cpu_composed", "worker_pins":inventory, "source_pins":pins,
              "copied_from":copied, "CPU_checks":checks,
              "option_combinations":combinations, "positional_fields_preserved":True,
              "GPU_executed":False, "G_writes":False, "availability_published":False,
              "limits":"CPU metadata composition, not GPU compilation, numerical or speed acceptance."}
    (STAGE / "source-manifest.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k:result[k] for k in ("status", "option_combinations", "GPU_executed", "G_writes")}))


if __name__ == "__main__":
    main()
