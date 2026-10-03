"""Freeze the reviewed CPU-passed patch over the exact installed native base."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
PATCH = PROJECT / "artifacts/periodic-flash-state-proof-v1-20261001"
OUTPUT = PROJECT / "artifacts/periodic-flash-state-proof-v1-build-source-20261001"
PERF = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001")
BUILD = PERF / "periodic-flash-state-proof-v1-native-build"
CPU_RECEIPT = PERF / "periodic-flash-state-proof-v1-cpu-20261001/cpu-receipt.json"
CHANGED = {"include/nr_sync_probe.h", "src/nr_sync_probe.cpp", "src/asi.cpp"}
EXPECTED = {
    "include/nr_sync_probe.h": "af9727b58823f785fd0c33e31517e1b1f3777cb6c7c0e6f2c7cb579b0309d4f4",
    "src/nr_sync_probe.cpp": "c1db7fe75927cb919a20b4a12670aceb075ff23dbd89acee19f9621eed0f620e",
    "src/asi.cpp": "eee577ce542c5f3722d5ee6cd7ebec83cd82773291681fabc00a3b1ba8cf6555",
}
CMAKE_ADDITION = """
# Main integration: the scoped recorded-state fix has a separate CPU test.
option(NRB_DLSS_COLOR_STATE_PROOF_V1 "Scoped DLSS recorded-color state fix" OFF)
if(NRB_DLSS_COLOR_STATE_PROOF_V1)
  target_compile_definitions(CyberpunkNRBridge PRIVATE NRB_DLSS_COLOR_STATE_PROOF_V1=1)
endif()
add_executable(nr_color_state_proof_tests tests/state_proof_cpu.cpp)
target_link_libraries(nr_color_state_proof_tests PRIVATE nr_core)
target_compile_definitions(nr_color_state_proof_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
add_test(NAME nr_color_state_proof_tests COMMAND nr_color_state_proof_tests)
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def physical(path):
    for item in (path, *path.parents):
        if item.is_symlink() or item.is_junction():
            raise RuntimeError("redirected path: " + str(item))


def main():
    physical(PATCH)
    physical(OUTPUT)
    receipt_bytes = CPU_RECEIPT.read_bytes()
    receipt = json.loads(receipt_bytes)
    if (receipt["status"] != "passed" or receipt["prototype_sources"] != EXPECTED
            or receipt["asi_compile_only_modes"] != ["off", "on"]
            or not receipt["no_gpu_or_live_api"]):
        raise RuntimeError("reviewed CPU delivery changed")
    base = json.loads((BUILD / "base-stage-receipt.json").read_text(encoding="utf-8"))
    for relative, digest in base["base_files"].items():
        target = OUTPUT / relative
        physical(target)
        if sha(target.read_bytes()) != digest:
            raise RuntimeError("fresh exact base changed: " + relative)
    files = {}
    for relative in CHANGED:
        path = PATCH / relative
        physical(path)
        data = path.read_bytes()
        if sha(data) != EXPECTED[relative]:
            raise RuntimeError("reviewed patch changed: " + relative)
        files[relative] = data
    test = PATCH / "tests/state_proof_cpu.cpp"
    physical(test)
    files["tests/state_proof_cpu.cpp"] = test.read_bytes()
    cmake = OUTPUT / "CMakeLists.txt"
    files["CMakeLists.txt"] = (cmake.read_text(encoding="utf-8") + CMAKE_ADDITION).encode("utf-8")
    for relative, data in files.items():
        target = OUTPUT / relative
        physical(target)
        if not target.resolve().is_relative_to(OUTPUT.resolve()):
            raise RuntimeError("target escaped build source stage")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    source_files = {p.relative_to(OUTPUT).as_posix(): sha(p.read_bytes())
                    for p in OUTPUT.rglob("*") if p.is_file()}
    changed = [name for name, digest in base["base_files"].items()
               if sha((OUTPUT / name).read_bytes()) != digest]
    if {name.replace("\\", "/") for name in changed} != CHANGED | {"CMakeLists.txt"}:
        raise RuntimeError("unexpected native or runtime delta")
    report = {"status": "reviewed_overlay_staged", "source_root": str(OUTPUT),
              "source_files": source_files, "changed_files": sorted(CHANGED | {"CMakeLists.txt", "tests/state_proof_cpu.cpp"}),
              "worker_cpu_receipt_sha256": sha(receipt_bytes),
              "GPU_executed": False, "G_writes": False,
              "child_environment": {"NRB_DLSS_COLOR_STATE_PROOF_SCOPE": "legacy-direct-reviewed-v1"}}
    with (BUILD / "reviewed-overlay-receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "changed_files": report["changed_files"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
