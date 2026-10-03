"""Check the isolated ASI/Python/native interface without loading any DLL."""
import argparse
import ast
import ctypes as C
import hashlib
import json
from pathlib import Path
import re

from review_gpu_handoff_product_probe_cpu import review


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def require(value, message):
    if not value:
        raise ValueError(message)


def structure(path, name):
    text = Path(path).read_text(encoding="utf-8-sig")
    body = re.search(r"struct\s+" + re.escape(name) + r"\s*\{(.*?)\};", text, re.S)
    require(body is not None, "Missing C++ structure: " + name)
    return re.sub(r"\s+", "", re.sub(r"//[^\n]*", "", body.group(1)))


def python_structure(path, name):
    tree = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    namespace = {"C": C}
    code = compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec")
    exec(code, namespace)
    row = namespace[name]
    return C.sizeof(row), [(field, C.sizeof(kind), getattr(row, field).offset)
                          for field, kind in row._fields_]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-stage", type=Path, required=True)
    parser.add_argument("--native-cpu-receipt", type=Path, required=True)
    parser.add_argument("--product-stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    native_review = review(args.native_cpu_receipt, None, args.native_stage)
    root = args.product_stage
    frozen, pins = read(root / "SOURCE_FROZEN.json"), read(root / "SOURCE_PINS.json")
    require(sha(root / "SOURCE_PINS.json") == frozen["source_pins_sha256"], "Product manifest changed")
    require(len(pins) == frozen["payload_files"], "Incomplete product source manifest")
    for name, digest in pins.items():
        require(sha(root / "payload" / name) == digest, "Product source changed: " + name)
    cpu = read(frozen["CPU_receipt"])
    require(cpu["status"] == "cpu_build_and_fake_protocol_passed", "Product CPU checks failed")
    require(cpu["GPU_executed"] is False and cpu["ASI_loaded"] is False and cpu["G_writes"] is False,
            "Unexpected product CPU scope")
    require(cpu["asi_sha256"] == frozen["ASI_sha256"] == sha(cpu["asi"]), "Product ASI changed")
    checks = Path(frozen["CPU_receipt"]).parent
    compiled = read(checks / "COMPILED_SOURCE_PINS.json")
    for name, digest in compiled.items():
        require(pins.get(name) == digest, "Product compiled source pin differs: " + name)
    gate = read(checks / "CPU_GATE_RECEIPT.json")
    require(gate["status"] == "passed" and gate["ASI_sha256"] == cpu["asi_sha256"] and
            (gate["selected_ctests_passed"], gate["protocol_checks"], gate["host_web_checks"]) == (6, 76, 48),
            "Product source/ABI/controller CPU gate incomplete")
    expected = structure(args.native_stage / "payload/native/nr_texture_bridge_v1.h",
                         "NR_TextureGpuHandoffInfo")
    actual = structure(root / "payload/include/nr_gpu_handoff_api.h", "NRB_NativeGpuHandoffInfo")
    require(actual == expected, "Native/ASI capability information layouts differ")
    native_layout = python_structure(args.native_stage / "payload/game/nr_texture_bridge_v1.py", "HandoffInfo")
    web_layout = python_structure(root / "payload/game/cyberpunk_nr_web.py", "NativeHandoffInfo")
    require(native_layout == web_layout and native_layout[0] == 128,
            "Native/web ctypes field order or offsets differ")
    old_native = (args.native_stage / "baseline/native/nr_texture_bridge_v1.h").read_text(encoding="utf-8-sig")
    new_native = (args.native_stage / "payload/native/nr_texture_bridge_v1.h").read_text(encoding="utf-8-sig")
    require(new_native.startswith(old_native), "Old texture ABI declaration prefix changed")
    baseline_pins = read(root / "BASELINE_PINS.json")
    baseline_pins = baseline_pins["merged_baseline"]
    for name in ("include/nr_bridge.h", "include/nr_contract.h", "shaders/nr_hdr_proxy.hlsl"):
        require(pins[name] == baseline_pins[name], "Old Frame/Controls or pixel shader changed: " + name)
    model = read(root / "MODEL_SYNC_READ_ONLY_PIN.json")
    require(sha(model["path"]) == model["sha256"] and model["modified"] is False,
            "Model synchronization source changed")
    result = {
        "status": "isolated_native_ASI_web_ABI_coupling_passed",
        "native_build_review": native_review,
        "native_stage": str(args.native_stage), "product_stage": str(root),
        "product_source_manifest_sha256": sha(root / "SOURCE_PINS.json"),
        "product_frozen_receipt_sha256": sha(root / "SOURCE_FROZEN.json"),
        "product_CPU_receipt": frozen["CPU_receipt"], "product_ASI_sha256": cpu["asi_sha256"],
        "payload_pins_checked": len(pins), "capability_info_bytes": 128,
        "compiled_source_pins_checked": len(compiled),
        "product_CPU_gate_sha256": sha(checks / "CPU_GATE_RECEIPT.json"),
        "capability_field_offsets": web_layout[1], "Frame_Controls_and_HLSL_preserved": True,
        "GPU_executed": False, "G_writes": False,
        "whole_model_async": False, "GPU_and_game_performance_acceptance": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "capability_field_offsets"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
