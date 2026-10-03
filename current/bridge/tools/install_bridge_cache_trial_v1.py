"""Install only a pinned, GPU-verified bridge candidate while games are closed."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

PROJECT=Path(__file__).resolve().parents[1]
PLUGINS=Path(r"G:\epic\Cyberpunk2077\bin\x64\plugins")
RUNTIME=Path(r"G:\SteamLibrary\steamapps\common\Resident Evil Village BIOHAZARD VILLAGE\nr-runtime")
OUTPUT=Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\numeric-game-stage-720-20260930\live-web-perf-20261001\bridge-cache-trial-v1")
EXPECTED_ASI="5b5ebabc315390d5076a585263429a32aee993e1eeb5f83c85a63e3db2515c73"
EXPECTED_PYTHON={
    PLUGINS/"cyberpunk_nr_adapter.py":"3a40d6189f4cab2b1c232871135cdd07b3b56d76afefa1eb1d07b67192b292a2",
    PLUGINS/"cyberpunk_nr_web.py":"e024ac1f18f589281add029264030d6e2f3d62b7d1997833c9013c1a0d9bf1ad",
    RUNTIME/"game/nr_game_controls.py":"1d45081a72f2015ad4f1877a63b89994e78fc1130e2e84b1ade594d1fb329079",
}

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def physical(path,root):
    if not path.resolve().is_relative_to(root.resolve()):raise RuntimeError("Path escaped target: "+str(path))
    for part in (path,*path.parents):
        if part.is_symlink() or part.is_junction():raise RuntimeError("Redirected path: "+str(part))
def closed():
    code="if (Get-Process -Name Cyberpunk2077,re8 -ErrorAction SilentlyContinue) { exit 11 } else { exit 0 }"
    result=subprocess.run(["powershell","-NoProfile","-Command",code],capture_output=True,text=True,check=False)
    if result.returncode:raise RuntimeError("Close Cyberpunk/RE8 before replacement")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--build-receipt",required=True,type=Path)
    parser.add_argument("--gpu-receipt",required=True,type=Path)
    parser.add_argument("--install",action="store_true")
    args=parser.parse_args()
    build=json.loads(args.build_receipt.read_text(encoding="utf-8"))
    gpu=json.loads(args.gpu_receipt.read_text(encoding="utf-8"))
    if build.get("status")!="cpu_build_passed":raise RuntimeError("CPU build receipt missing")
    if gpu.get("status")!="passed" or gpu.get("completed") is not True or gpu.get("outputs_identical") is not True or gpu.get("stable_cache_counters") is not True:
        raise RuntimeError("Completed cache GPU identity/counter verification required")
    source=Path(build["source"]);binary=Path(build["asi"])
    for name,pin in build["source_pins"].items():
        physical(source/name,PROJECT)
        if digest(source/name)!=pin:raise RuntimeError("Build source changed: "+name)
    if digest(binary)!=build["asi_sha256"]:raise RuntimeError("Built ASI changed")
    expected={name:build["source_pins"][name] for name in ("src/nr_hdr_proxy.cpp","src/nr_hdr_proxy.h")}
    if gpu.get("cache_source_pins")!=expected:raise RuntimeError("GPU proof tested different cache source")
    shader=PLUGINS/"nr_hdr_proxy.hlsl";physical(shader,PLUGINS)
    shader_pin=build["source_pins"]["shaders/nr_hdr_proxy.hlsl"]
    if digest(shader)!=shader_pin or gpu.get("shader_identity",{}).get("source_sha256")!=shader_pin:
        raise RuntimeError("Installed shader differs from the GPU-tested source")
    targets=[(binary,PLUGINS/"CyberpunkNRBridge.asi",PLUGINS),
             (source/"game/cyberpunk_nr_web.py",PLUGINS/"cyberpunk_nr_web.py",PLUGINS),
             (source/"game/cyberpunk_nr_adapter.py",PLUGINS/"cyberpunk_nr_adapter.py",PLUGINS),
             (source/"game/nr_game_controls.py",RUNTIME/"game/nr_game_controls.py",RUNTIME)]
    for src,dest,root in targets:
        physical(dest,root)
        if not dest.is_file():raise RuntimeError("Installed file missing: "+str(dest))
    if digest(PLUGINS/"CyberpunkNRBridge.asi")!=EXPECTED_ASI:
        raise RuntimeError("Installed native base changed; review before replacing")
    for target,pin in EXPECTED_PYTHON.items():
        if digest(target)!=pin:raise RuntimeError("Installed Python base changed: "+str(target))
    closed()
    plan=[{"source":str(src),"target":str(dest),"old_sha256":digest(dest),"new_sha256":digest(src)} for src,dest,_ in targets]
    if not args.install:
        print(json.dumps({"status":"reviewed_dry_run","files":plan},ensure_ascii=False));return
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    backup=OUTPUT/stamp
    physical(backup,OUTPUT);backup.mkdir(parents=True)
    for i,(_,dest,_) in enumerate(targets):
        saved=backup/f"{i}-{dest.name}";saved.write_bytes(dest.read_bytes());plan[i]["backup"]=str(saved)
        if digest(saved)!=plan[i]["old_sha256"]:raise RuntimeError("Backup failed")
    receipt={"status":"backed_up","files":plan,"GPU_receipt":str(args.gpu_receipt),"build_receipt":str(args.build_receipt)}
    (backup/"installed.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    replaced=[]
    try:
        closed()
        for i,(src,dest,root) in enumerate(targets):
            if digest(dest)!=plan[i]["old_sha256"]:raise RuntimeError("Target changed after review")
            temp=dest.with_name(dest.name+".bridge-cache-"+stamp+".tmp");physical(temp,root)
            if temp.exists():raise RuntimeError("Fresh temporary target required")
            temp.write_bytes(src.read_bytes())
            if digest(temp)!=plan[i]["new_sha256"]:raise RuntimeError("Staged write failed")
            os.replace(temp,dest);replaced.append(i)
        for i,(_,dest,_) in enumerate(targets):
            if digest(dest)!=plan[i]["new_sha256"]:raise RuntimeError("Installed verification failed")
    except Exception:
        for i in reversed(replaced):
            dest=targets[i][1];dest.write_bytes(Path(plan[i]["backup"]).read_bytes())
        receipt["status"]="rolled_back"
        (backup/"installed.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
        raise
    receipt["status"]="installed_trial"
    (backup/"installed.json").write_text(json.dumps(receipt,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":receipt["status"],"receipt":str(backup/"installed.json")},ensure_ascii=False))

if __name__=="__main__":main()
