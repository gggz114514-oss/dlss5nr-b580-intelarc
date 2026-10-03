"""Isolate external forward-fence reuse after two retired ordering failures."""
import ast
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OLD = PROJECT / "artifacts/dx12-xpu-fence-probe-v3-20261002"
NEW = PROJECT / "artifacts/dx12-xpu-fence-probe-v4-20261002"
DATA = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\dx12-xpu-fence-probe-v3-20261002")
CPU = DATA / "cpu-jmpo55_7/CPU_COMPILE_RECEIPT.json"
FAILED = DATA / "runtime-trace-v1/RUNTIME_RECEIPT.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def one(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("Frozen anchor changed: " + old[:100])
    return text.replace(old, new)


def main():
    cpu = json.loads(CPU.read_text(encoding="utf-8"))
    failure = json.loads(FAILED.read_text(encoding="utf-8"))
    child = failure["child_result"]
    if failure["status"] != "failed" or child["stage"] != 53 or not child["cleanup_ok"]:
        raise RuntimeError("Retired two-round data mismatch required")
    if child["rounds"][0]["mismatch_index"] != 0xffffffff or child["rounds"][1]["mismatch_index"] != 0:
        raise RuntimeError("Unexpected failure shape")
    for name, expected in cpu["source_pins"].items():
        if sha(OLD / name) != expected:
            raise RuntimeError("Frozen source changed: " + name)
    if NEW.exists() or not NEW.resolve().is_relative_to(PROJECT.resolve()):
        raise RuntimeError("Fresh E source stage required")
    for parent in (NEW.parent, *NEW.parent.parents):
        if parent.is_junction() or parent.is_symlink():
            raise RuntimeError("Redirected source stage")
    NEW.mkdir()
    for name in (*cpu["source_pins"], "INPUT_PINS.json"):
        path = NEW / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((OLD / name).read_bytes())
    path = NEW / "native/dx12_xpu_fence_probe_v1.cpp"
    text = path.read_text(encoding="utf-8")
    text = one(text, "    Semaphore forward,backward;", "    std::array<Semaphore,fence_probe::rounds> forward;\n    Semaphore backward;")
    text = one(text, '''        if(forward.valid && !release("release_external_semaphore(forward)",[&] {
            sx::release_external_semaphore(forward.imported,*xpu);forward.valid=false;
        })) return false;''', '''        for(auto& sem:forward) {
            if(sem.valid && !release("release_external_semaphore(forward per round)",[&] {
                sx::release_external_semaphore(sem.imported,*xpu);sem.valid=false;
            })) return false;
        }''')
    text = one(text, '    import(state.forward,5,"import_external_semaphore(win32_nt_dx12_fence,forward)");', '''    // v4 diagnostic: fresh forward fence per round; backward timeline and
    // feedback stay shared. This is not a production fence-allocation plan.
    for(auto& sem:state.forward)
        import(sem,5,"import_external_semaphore(win32_nt_dx12_fence,forward per round)");''')
    text = one(text, "state.forward.imported,fence_probe::producer_value(r)", "state.forward[r].imported,fence_probe::producer_value(r)")
    text = one(text, "state.forward.fence.Get(),fence_probe::producer_value(r)", "state.forward[r].fence.Get(),fence_probe::producer_value(r)")
    text = one(text, "out.producer_completed=state.forward.fence->GetCompletedValue();", "out.producer_completed=state.forward.back().fence->GetCompletedValue();")
    path.write_text(text, encoding="utf-8", newline="\n")
    path = NEW / "build_cpu.py"
    text = path.read_text(encoding="utf-8").replace("dx12-xpu-fence-probe-v3-20261002", "dx12-xpu-fence-probe-v4-20261002")
    path.write_text(text, encoding="utf-8", newline="\n")
    path = NEW / "luna_runner.py"
    text = path.read_text(encoding="utf-8").replace("explicit_handler_wait_value_dependency_v3", "separate_forward_fence_each_round_v4")
    path.write_text(text, encoding="utf-8", newline="\n")
    for path in NEW.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    receipt = {
        "status": "source_staged_GPU_unverified", "previous_cpu_receipt": str(CPU),
        "previous_failure": str(FAILED), "previous_failure_sha256": sha(FAILED),
        "difference": "Only forward fence/import identity becomes per-round. Same in-order borrowed queue, values, backward fence, feedback, patterns and terminal gates.",
        "acceptance": "Both rounds 256/256 expected data; same queue/context/LUID; GPU-only relay; drained safe cleanup. Passing only isolates fence reuse, not game performance.",
        "source_pins": {p.relative_to(NEW).as_posix(): sha(p) for p in NEW.rglob("*") if p.is_file()},
        "GPU_executed": False, "G_writes": False,
    }
    (NEW / "STAGE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "staged", "path": str(NEW), "GPU_executed": False}))


if __name__ == "__main__":
    main()
