"""Separate imported-semaphore reuse from the underlying DX12 fence identity."""
import ast
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OLD = PROJECT / "artifacts/dx12-xpu-fence-probe-v4-20261002"
NEW = PROJECT / "artifacts/dx12-xpu-fence-probe-v5-20261002"
DATA = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\dx12-xpu-fence-probe-v4-20261002")
CPU = DATA / "cpu-_ryy353y/CPU_COMPILE_RECEIPT.json"
PASSED = DATA / "runtime-trace-v1/RUNTIME_RECEIPT.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("Changed source anchor: " + old[:100])
    return text.replace(old, new)


def main():
    cpu = json.loads(CPU.read_text(encoding="utf-8"))
    passed = json.loads(PASSED.read_text(encoding="utf-8"))
    if not passed["passed"] or not passed["child_result"]["cleanup_ok"]:
        raise RuntimeError("Retired v4 GPU pass required")
    for name, expected in cpu["source_pins"].items():
        if sha(OLD / name) != expected:
            raise RuntimeError("Frozen source changed: " + name)
    if NEW.exists() or not NEW.resolve().is_relative_to(PROJECT.resolve()):
        raise RuntimeError("Fresh E source stage required")
    NEW.mkdir()
    for name in (*cpu["source_pins"], "INPUT_PINS.json"):
        path = NEW / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((OLD / name).read_bytes())
    path = NEW / "native/dx12_xpu_fence_probe_v1.cpp"
    text = path.read_text(encoding="utf-8")
    text = one(text, '''        trace.phase(stage,"ID3D12Device::CreateFence(SHARED,initial value0)");
        check(state.device->CreateFence(0,D3D12_FENCE_FLAG_SHARED,IID_PPV_ARGS(&sem.fence)),"shared DX12 fence");''', '''        if(!sem.fence) {
            trace.phase(stage,"ID3D12Device::CreateFence(SHARED,initial value0)");
            check(state.device->CreateFence(0,D3D12_FENCE_FLAG_SHARED,IID_PPV_ARGS(&sem.fence)),"shared DX12 fence");
        }''')
    text = one(text, '''    // v4 diagnostic: fresh forward fence per round; backward timeline and
    // feedback stay shared. This is not a production fence-allocation plan.
    for(auto& sem:state.forward)
        import(sem,5,"import_external_semaphore(win32_nt_dx12_fence,forward per round)");''', '''    // v5 diagnostic: one DX12 forward timeline, two independent imports.
    // Backward timeline/feedback and all numeric gates remain unchanged.
    for(uint32_t r=0;r<fence_probe::rounds;++r) {
        if(r) state.forward[r].fence=state.forward[0].fence;
        import(state.forward[r],5,"import_external_semaphore(same DX12 fence,new import per round)");
        require(state.forward[r].fence.Get()==state.forward[0].fence.Get(),"forward fence identity changed");
    }''')
    path.write_text(text, encoding="utf-8", newline="\n")
    path = NEW / "build_cpu.py"
    text = path.read_text(encoding="utf-8").replace("dx12-xpu-fence-probe-v4-20261002", "dx12-xpu-fence-probe-v5-20261002")
    path.write_text(text, encoding="utf-8", newline="\n")
    path = NEW / "luna_runner.py"
    text = path.read_text(encoding="utf-8").replace("separate_forward_fence_each_round_v4", "same_forward_fence_separate_imports_v5")
    path.write_text(text, encoding="utf-8", newline="\n")
    for path in NEW.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    receipt = {"status": "source_staged_GPU_unverified", "previous_cpu_receipt": str(CPU),
        "previous_pass": str(PASSED), "previous_pass_sha256": sha(PASSED),
        "difference": "Same DX12 forward fence identity, one independent external import per round. No CPU handoff wait or new queue/context.",
        "acceptance": "Both rounds 256/256 data, same queue/context/LUID, unchanged GPU graph, safe drained cleanup. No game or performance acceptance.",
        "source_pins": {p.relative_to(NEW).as_posix(): sha(p) for p in NEW.rglob("*") if p.is_file()},
        "GPU_executed": False, "G_writes": False}
    (NEW / "STAGE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "staged", "path": str(NEW), "GPU_executed": False}))


if __name__ == "__main__": main()
