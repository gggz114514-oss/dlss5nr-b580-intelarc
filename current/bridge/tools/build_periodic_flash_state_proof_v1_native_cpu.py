"""Link the scoped native fix with 14 build workers and run CPU-only cases."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

PROJECT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT / "artifacts/periodic-flash-state-proof-v1-build-source-20261001"
BUILD = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001\periodic-flash-state-proof-v1-native-build")
CMAKE = r"E:\VS2022BT\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
CTEST = r"E:\VS2022BT\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\ctest.exe"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(arguments, logfile):
    with (BUILD / logfile).open("x", encoding="utf-8") as handle:
        result = subprocess.run(arguments, cwd=BUILD, stdout=handle, stderr=subprocess.STDOUT,
                                text=True, check=False)
    print(json.dumps({"step": logfile, "exit_code": result.returncode}), flush=True)
    if result.returncode:
        raise RuntimeError("CPU build/check failed: " + logfile)


def main():
    report = json.loads((BUILD / "reviewed-overlay-receipt.json").read_text(encoding="utf-8"))
    for name, digest in report["source_files"].items():
        if sha(SOURCE / name) != digest:
            raise RuntimeError("reviewed source changed: " + name)
    run([CMAKE, "-S", str(SOURCE), "-B", str(BUILD), "-DNRB_DLSS_COLOR_STATE_PROOF_V1=ON",
         "-DCMAKE_CXX_FLAGS=/MP14"], "configure-proof-on.log")
    run([CMAKE, "--build", str(BUILD), "--config", "Release", "--parallel", "14",
         "--target", "CyberpunkNRBridge", "nr_color_state_proof_tests",
         "--", "/p:CL_MPCount=14"], "build-proof-on.log")
    run([CTEST, "--test-dir", str(BUILD), "-C", "Release", "-R", "^nr_color_state_proof_tests$",
         "--output-on-failure"], "state-proof-cpu-tests.log")
    for name, digest in report["source_files"].items():
        if sha(SOURCE / name) != digest:
            raise RuntimeError("source changed while compiling: " + name)
    binary = BUILD / "Release/CyberpunkNRBridge.asi"
    report.update(status="cpu_build_passed", asi_path=str(binary), asi_sha256=sha(binary),
                  build_workers=14, CPU_checks="13 scoped recorded-state cases; no GPU imports",
                  GPU_executed=False)
    with (BUILD / "build-receipt.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"status": report["status"], "asi_sha256": report["asi_sha256"],
                      "receipt": str(BUILD / "build-receipt.json")}), flush=True)


if __name__ == "__main__":
    main()
