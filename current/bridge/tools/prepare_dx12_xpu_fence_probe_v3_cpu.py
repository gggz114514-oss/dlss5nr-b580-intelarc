"""Try explicit handler dependencies after concrete second-round ordering failure."""
import ast
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OLD = PROJECT / "artifacts/dx12-xpu-fence-probe-v2-20261002"
NEW = PROJECT / "artifacts/dx12-xpu-fence-probe-v3-20261002"
DATA = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\dx12-xpu-fence-probe-v2-20261002")
CPU = DATA / "cpu-gak6h7te/CPU_COMPILE_RECEIPT.json"
FAILED = DATA / "runtime-v1/RUNTIME_RECEIPT.json"

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def one(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("Frozen anchor changed: " + old[:100])
    return text.replace(old, new)

def rot(x, n):
    return ((x << n) | (x >> (32-n))) & 0xffffffff

def prod(x, i, r):
    return (0x13579bdf ^ (i*0x10203+7)) if r == 0 else ((x+0x2468ace1+i*13) & 0xffffffff) ^ 0x55aa33cc

def modify(x, i, r):
    return (rot(x ^ ((0xa5a55a5a+r*0x1020304) & 0xffffffff), 5)+i*17+r) & 0xffffffff

def consume(x, i, r):
    return rot((x+0x31415927+r+i*3) & 0xffffffff, 11) ^ 0xc001d00d

def main():
    cpu = json.loads(CPU.read_text(encoding="utf-8"))
    failed = json.loads(FAILED.read_text(encoding="utf-8"))
    child = failed["child_result"]
    if failed["status"] != "failed" or child["stage"] != 53 or child["cleanup_ok"] != 1:
        raise RuntimeError("Concrete retired data-mismatch receipt required")
    rounds = child["rounds"]
    first = [consume(modify(prod(0, i, 0), i, 0), i, 0) for i in range(256)]
    premature = [consume(prod(modify(first[i], i, 1), i, 1), i, 1) for i in range(256)]
    correct = [consume(modify(prod(first[i], i, 1), i, 1), i, 1) for i in range(256)]
    if rounds[0]["observed"] != first or rounds[1]["observed"] != premature or premature == correct:
        raise RuntimeError("Second round is not the diagnosed modify-before-producer signature")
    review_path = DATA / "runtime-v1/MAIN_ORDER_REVIEW.json"
    if not review_path.exists():
        review_path.write_text(json.dumps({
            "status": "cpu_formula_review_passed", "round0_matches_expected": 256,
            "round1_matches_modify_then_produce_then_consume": 256,
            "round1_matches_correct_order": sum(a == b for a, b in zip(premature, correct)),
            "receipt_sha256": sha(FAILED), "GPU_executed": False,
            "scope": "Exact result signature. Does not by itself identify which runtime/driver dependency failed."
        }, indent=2) + "\n", encoding="utf-8")
    for name, digest in cpu["source_pins"].items():
        if sha(OLD / name) != digest:
            raise RuntimeError("Preserve v2 source bytes: " + name)
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
    text = one(text, '''        const auto wait=r==0 ? q.ext_oneapi_wait_external_semaphore(state.forward.imported,fence_probe::producer_value(r)):
            q.ext_oneapi_wait_external_semaphore(state.forward.imported,fence_probe::producer_value(r),state.events.back());
''', '''        // Explicit command-group dependency/value instead of the queue's
        // value+event convenience overload, whose second-round path failed.
        const auto wait=q.submit([&](sycl::handler& handler) {
            if(r) handler.depends_on(state.events.back());
            handler.ext_oneapi_wait_external_semaphore(state.forward.imported,fence_probe::producer_value(r));
        });
''')
    path.write_text(text, encoding="utf-8", newline="\n")
    path = NEW / "build_cpu.py"
    text = path.read_text(encoding="utf-8")
    text = one(text,
        'OUTPUT = Path(r"D:\\Codex-NR-Experiments\\cyberpunk-opt\\dx12-xpu-fence-probe-v2-20261002")',
        'OUTPUT = Path(r"D:\\Codex-NR-Experiments\\cyberpunk-opt\\dx12-xpu-fence-probe-v3-20261002")')
    text = text.replace('"q.ext_oneapi_wait_external_semaphore"', '"handler.ext_oneapi_wait_external_semaphore"')
    path.write_text(text, encoding="utf-8", newline="\n")
    path = NEW / "luna_runner.py"
    text = path.read_text(encoding="utf-8").replace('producer_admitted_before_SYCL_signal_v2', 'explicit_handler_wait_value_dependency_v3')
    path.write_text(text, encoding="utf-8", newline="\n")
    for path in NEW.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    (NEW / "STAGE_RECEIPT.json").write_text(json.dumps({
        "status": "source_staged_GPU_unverified", "previous_cpu_receipt": str(CPU),
        "previous_failure": str(FAILED), "previous_failure_sha256": sha(FAILED),
        "order_review": str(review_path), "order_review_sha256": sha(review_path),
        "source_pins": {p.relative_to(NEW).as_posix(): sha(p) for p in NEW.rglob("*") if p.is_file()},
        "difference": "Explicit SYCL handler wait value and previous event dependency; GPU timelines/feedback unchanged",
        "GPU_executed": False, "G_writes": False,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "staged", "path": str(NEW), "GPU_executed": False}))

if __name__ == "__main__":
    main()
