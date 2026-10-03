"""CPU build and timing ABI checks; never loads graphics devices or installs."""
from __future__ import annotations
import ast
import argparse
import ctypes
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

PROJECT=Path(__file__).resolve().parents[1]
STAGE=PROJECT/"artifacts/bridge-outer-timing-v1-20261001/payload"
BUILD=Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001\bridge-outer-timing-v1-build")
CMAKE=Path(r"E:\VS2022BT\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe")
SDK=Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\optiscaler-sol-ref")

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    global STAGE,BUILD
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",type=Path,default=STAGE)
    parser.add_argument("--build",type=Path,default=BUILD)
    args=parser.parse_args();STAGE=args.stage;BUILD=args.build
    if not STAGE.resolve().is_relative_to(PROJECT.resolve()):
        raise RuntimeError("Source stage outside project")
    allowed=Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001")
    if not BUILD.resolve().is_relative_to(allowed.resolve()):
        raise RuntimeError("Build outside task experiment root")
    for target in (STAGE,BUILD):
        for part in (target,*target.parents):
            if part.is_symlink() or part.is_junction():raise RuntimeError("Redirected build target")
    BUILD.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((STAGE.parent/"source-manifest.json").read_text(encoding="utf-8"))
    for name,value in manifest.get("staged_pins",manifest.get("source_pins",{})).items():
        if name=="CMakeLists.txt" and (BUILD/"build-receipt.json").exists():
            continue
        if digest(STAGE/name)!=value:raise RuntimeError("Staged source changed: "+name)
    cmake=STAGE/"CMakeLists.txt"
    text=cmake.read_text(encoding="utf-8")
    if "nr_record_timing_tests" not in text:
        text+='\nadd_executable(nr_record_timing_tests tests/record_timing_cpu.cpp)\ntarget_include_directories(nr_record_timing_tests PRIVATE include "${OPTISCALER_SOURCE}/external/nvngx_dlss_sdk")\ntarget_compile_definitions(nr_record_timing_tests PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)\nadd_test(NAME nr_record_timing_tests COMMAND nr_record_timing_tests)\n'
        cmake.write_text(text,encoding="utf-8",newline="\n")
    pychecks=0
    for name in ("cyberpunk_nr_web.py","nr_game_controls.py"):
        path=STAGE/"game"/name;ast.parse(path.read_text(encoding="utf-8"),filename=str(path));pychecks+=1
    spec=importlib.util.spec_from_file_location("outer_web_cpu",STAGE/"game/cyberpunk_nr_web.py")
    web=importlib.util.module_from_spec(spec);spec.loader.exec_module(web)
    if ctypes.sizeof(web.StageTimes)!=80 or ctypes.sizeof(web.RecordTimes)!=120:
        raise RuntimeError("Timing ABI changed unexpectedly")
    pychecks+=1
    if hasattr(web,"HdrCacheStats"):
        if ctypes.sizeof(web.HdrCacheStats)!=96:raise RuntimeError("Cache ABI mismatch")
        pychecks+=1
    pins={path.relative_to(STAGE).as_posix():digest(path) for path in STAGE.rglob("*") if path.is_file() and "__pycache__" not in path.parts}
    def run(arguments,name):
        with (BUILD/name).open("w",encoding="utf-8") as log:
            result=subprocess.run([str(v) for v in arguments],stdout=log,stderr=subprocess.STDOUT,check=False)
        print(json.dumps({"step":name,"exit_code":result.returncode}),flush=True)
        if result.returncode:raise RuntimeError("CPU build/check failed: "+str(BUILD/name))
    run([CMAKE,"-S",STAGE,"-B",BUILD,"-G","Visual Studio 17 2022","-A","x64",f"-DOPTISCALER_SOURCE={SDK}","-DCMAKE_CXX_FLAGS=/MP14 /EHsc","-DNRB_XESS_IDENTITY_TEST=ON","-DNRB_TAIL_ORDER_PROBE=ON","-DNRB_DEFERRED_IDENTITY_TEST=ON","-DNRB_LIVE_NR_TEST=ON","-DNRB_MULTI_ROUTE_LIVE=ON","-DNRB_DLSS_COLOR_STATE_PROOF_V1=ON"],"configure.log")
    run([CMAKE,"--build",BUILD,"--config","Release","--parallel","14","--target","CyberpunkNRBridge","nr_record_timing_tests","nr_color_state_proof_tests","--","/p:CL_MPCount=14"],"build.log")
    run([BUILD/"Release/nr_record_timing_tests.exe"],"record-tests.log")
    run([BUILD/"Release/nr_color_state_proof_tests.exe"],"state-proof-tests.log")
    for name,value in pins.items():
        if digest(STAGE/name)!=value:raise RuntimeError("Source changed during build: "+name)
    report={"status":"cpu_build_passed","source":str(STAGE),"source_pins":pins,"asi":str(BUILD/"Release/CyberpunkNRBridge.asi"),"asi_sha256":digest(BUILD/"Release/CyberpunkNRBridge.asi"),"python_checks":pychecks,"build_workers":14,"GPU_executed":False,"G_writes":False}
    (BUILD/"build-receipt.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":report["status"],"receipt":str(BUILD/"build-receipt.json")}),flush=True)

if __name__=="__main__":main()
