"""Build startup recovery and test pre-NR observations without loading a DLL."""
from __future__ import annotations

import ast
import argparse
import ctypes as C
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "artifacts/periodic-flash-startup-shadow-v1-source-20261001"
BUILD = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001\periodic-flash-startup-shadow-v1-native-build")
CMAKE = r"E:\VS2022BT\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
CTEST = str(Path(CMAKE).with_name("ctest.exe"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, filename):
    with (BUILD / filename).open("x", encoding="utf-8") as handle:
        result = subprocess.run(args, cwd=BUILD, stdout=handle, stderr=subprocess.STDOUT, check=False)
    print(json.dumps({"step": filename, "exit_code": result.returncode}), flush=True)
    if result.returncode:
        raise RuntimeError("CPU check failed: " + filename)


def host_tests():
    helper_path = SOURCE / "game/periodic_flash_snapshot_v2.py"
    spec = importlib.util.spec_from_file_location("periodic_flash_snapshot_v2", helper_path)
    helper = importlib.util.module_from_spec(spec)
    tree = ast.parse((SOURCE / "game/nr_game_pre_xess_host.py").read_text(encoding="utf-8"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "periodic_flash_diagnostics")
    with mock.patch.object(C, "CDLL", side_effect=AssertionError("CPU test cannot load a DLL")):
        spec.loader.exec_module(helper)
        calls = []

        class SnapshotExport:
            def __call__(self, pointer):
                calls.append("native_snapshot")
                snapshot = C.cast(pointer, C.POINTER(helper.Snapshot)).contents
                snapshot.abi_size, snapshot.version = helper.ABI_SIZE, helper.VERSION
                snapshot.counters[helper.COUNTERS.index("seen")] = 101
                snapshot.counters[helper.COUNTERS.index("original_fallback")] = 101
                snapshot.frames_retained = 1
                snapshot.current.sr_sequence = 101
                snapshot.current.source_bits = (1 << 10) | (1 << 12) | (5 << 16)
                snapshot.frames[0].context = snapshot.current
                snapshot.frames[0].context.stage_bits = (1 << 8)
                snapshot.frames[0].reason = 2
                return 1

        dll = SimpleNamespace(NRB_GetPeriodicFlashDiagV2=SnapshotExport())
        web = SimpleNamespace(_native=dll)
        namespace = {"sys": sys, "_periodic_flash_sampler": None,
                     "_periodic_flash_diagnostics": None}
        with mock.patch.dict(sys.modules, {"periodic_flash_snapshot_v2": helper,
                                          "cyberpunk_nr_web": web}):
            exec(compile(ast.Module(body=[function], type_ignores=[]), "startup_observer", "exec"), namespace)
            value = namespace["periodic_flash_diagnostics"]()
            assert value["status"] == "ok"
            assert value["python_nr_call_observed"] is False
            assert value["counters"]["seen"] == 101
            assert value["counters"]["nr_started"] == 0
            assert value["recent_frames"][0]["python_join"] == "unavailable"
            assert value["recent_frames"][0]["raw_fallback"] is True
            assert value["negative_evidence_complete"] is False
            assert namespace["_periodic_flash_diagnostics"] is None
            assert calls == ["native_snapshot"]
            web._native = None
            unavailable = namespace["periodic_flash_diagnostics"]()
            assert unavailable["status"] == "unavailable"
            assert unavailable["reason"] == "no_existing_dll"
            assert unavailable["python_nr_call_observed"] is False
            web._native = dll
            namespace["_periodic_flash_diagnostics"] = helper.PythonFrameRecorder()
            observed = namespace["periodic_flash_diagnostics"]()
            assert observed["status"] == "ok" and observed["python_nr_call_observed"] is True
    return {"status": "passed", "checks": ["native rejection visible before first NR",
            "empty Python ring cannot fake NR completion", "unavailable DLL remains unavailable",
            "existing recorder path retained"], "GPU_or_real_DLL_loaded": False}


def main():
    global SOURCE, BUILD
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("startup-shadow-v1", "alias-state-v2", "list-state-v3", "local-decl-v4", "cp-default-v5"), default="startup-shadow-v1")
    parser.add_argument("--resume-receipt", action="store_true")
    options = parser.parse_args()
    if options.variant != "startup-shadow-v1":
        SOURCE = PROJECT / ("artifacts/periodic-flash-" + options.variant + "-source-20261001")
        BUILD = BUILD.parent / ("periodic-flash-" + options.variant + "-native-build")
    receipt = json.loads((BUILD / "source-receipt.json").read_text(encoding="utf-8"))
    for name, digest in receipt["source_files"].items():
        if sha(SOURCE / name) != digest:
            raise RuntimeError("prepared source changed: " + name)
    check = host_tests()
    (BUILD / "startup-observer-cpu.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
    args = [CMAKE, "-S", str(SOURCE), "-B", str(BUILD), "-G", "Visual Studio 17 2022", "-A", "x64",
            "-DCMAKE_GENERATOR_INSTANCE=C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools",
            "-DOPTISCALER_SOURCE=D:/Codex-NR-Experiments/cyberpunk-opt/optiscaler-sol-ref",
            "-DCMAKE_CXX_FLAGS=/MP14"]
    for flag in ("NRB_XESS_IDENTITY_TEST", "NRB_DEFERRED_IDENTITY_TEST", "NRB_LIVE_NR_TEST",
                 "NRB_MULTI_ROUTE_LIVE", "NRB_TAIL_ORDER_PROBE", "NRB_DLSS_COLOR_STATE_PROOF_V1"):
        args.append("-D" + flag + "=ON")
    args += ["-DNRB_ONE_FRAME_COLOR_READBACK=OFF", "-DNRB_BUILD_GENERIC_DLSS_PROBE=OFF"]
    if options.resume_receipt:
        tests = (BUILD / "cpu-tests.log").read_text(encoding="utf-8", errors="replace")
        built = (BUILD / "build.log").read_text(encoding="utf-8", errors="replace")
        if ("100% tests passed, 0 tests failed" not in tests
                or "CyberpunkNRBridge.asi" not in built
                or not (BUILD / "Release/nr_color_state_proof_tests.exe").is_file()):
            raise RuntimeError("completed CPU compile/test logs required to finish receipt")
    else:
        run(args, "configure.log")
        run([CMAKE, "--build", str(BUILD), "--config", "Release", "--parallel", "14", "--target",
             "CyberpunkNRBridge", "nr_color_state_proof_tests", "--", "/p:CL_MPCount=14"], "build.log")
        run([CTEST, "--test-dir", str(BUILD), "-C", "Release", "-R", "^nr_color_state_proof_tests$",
             "--output-on-failure"], "cpu-tests.log")
    for name, digest in receipt["source_files"].items():
        if sha(SOURCE / name) != digest:
            raise RuntimeError("source changed while building: " + name)
    binary = BUILD / "Release/CyberpunkNRBridge.asi"
    receipt.update(status="cpu_build_passed", asi_path=str(binary), asi_sha256=sha(binary),
                   build_workers=14, startup_observer_check=check,
                   CPU_checks={"startup-shadow-v1": "14", "alias-state-v2": "15", "list-state-v3": "17", "local-decl-v4": "17", "cp-default-v5": "18"}[options.variant]
                              + " recorded-state cases plus startup observer tests", GPU_executed=False)
    with (BUILD / "build-receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
    print(json.dumps({"status": receipt["status"], "asi_sha256": receipt["asi_sha256"]}), flush=True)


if __name__ == "__main__":
    main()
