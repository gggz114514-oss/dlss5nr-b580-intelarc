"""Stage the producer-admission variant; preserve v1 source and timeout evidence."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OLD = PROJECT / "artifacts/dx12-xpu-fence-probe-v1-20261001"
NEW = PROJECT / "artifacts/dx12-xpu-fence-probe-v2-20261002"
CPU = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\dx12-xpu-fence-probe-v1-20261001\cpu-06xhjf2l\CPU_COMPILE_RECEIPT.json")
FAILED = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\dx12-xpu-fence-probe-v1-20261001\runtime-v1\RUNTIME_RECEIPT.json")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def replace_one(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("Frozen source anchor changed: " + old[:100])
    return text.replace(old, new)

def main():
    receipt = json.loads(CPU.read_text(encoding="utf-8"))
    failure = json.loads(FAILED.read_text(encoding="utf-8"))
    if receipt["status"] != "cpu_compile_passed_runtime_unverified":
        raise RuntimeError("Original compiled probe required")
    if (failure["status"] != "timeout_api_or_driver_hang" or
            failure["last_progress"]["stage"] != 38 or
            failure["last_progress"]["producer_completed"] != 0):
        raise RuntimeError("Require the concrete unadmitted producer timeout")
    old_pins = receipt["source_pins"]
    for name, digest in old_pins.items():
        if sha(OLD / name) != digest:
            raise RuntimeError("Preserve original probe bytes: " + name)
    if NEW.exists() or not NEW.resolve().is_relative_to(PROJECT.resolve()):
        raise RuntimeError("Fresh bounded E source artifact required")
    for parent in (NEW.parent, *NEW.parent.parents):
        if parent.is_junction() or parent.is_symlink():
            raise RuntimeError("Redirected source path")
    NEW.mkdir()
    for name in (*old_pins, "INPUT_PINS.json"):
        destination = NEW / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((OLD / name).read_bytes())

    source = NEW / "native/dx12_xpu_fence_probe_v1.cpp"
    text = source.read_text(encoding="utf-8")
    signal = '''        trace.phase(38+r,"queue.ext_oneapi_signal_external_semaphore(backward,value,kernel event)");
        const auto signal=q.ext_oneapi_signal_external_semaphore(state.backward.imported,fence_probe::xpu_value(r),modified);
        state.events.push_back(signal);++out.gpu_signal_calls;
    }
    for(uint32_t r=0;r<fence_probe::rounds;++r) {
'''
    text = replace_one(text, signal, '''        // v2: admit this producer after the real unsatisfied SYCL wait/kernel
        // are enqueued, but before an external-signal API can block on them.
        // The consumer waits were queued first; feedback remains GPU-only.
''')
    producer = '''        check(state.producer->Signal(state.forward.fence.Get(),fence_probe::producer_value(r)),"producer GPU signal");
        ++out.gpu_signal_calls;
'''
    text = replace_one(text, producer, producer + '''        trace.phase(38+r,"queue.ext_oneapi_signal_external_semaphore(backward,value,kernel event)");
        const auto signal=q.ext_oneapi_signal_external_semaphore(state.backward.imported,fence_probe::xpu_value(r),modified);
        state.events.push_back(signal);++out.gpu_signal_calls;
''')
    source.write_text(text, encoding="utf-8", newline="\n")

    build = NEW / "build_cpu.py"
    text = build.read_text(encoding="utf-8")
    text = replace_one(text,
        'OUTPUT = Path(r"D:\\Codex-NR-Experiments\\cyberpunk-opt\\dx12-xpu-fence-probe-v1-20261001")',
        'OUTPUT = Path(r"D:\\Codex-NR-Experiments\\cyberpunk-opt\\dx12-xpu-fence-probe-v2-20261002")')
    anchor = '    assert submit.index("q.ext_oneapi_wait_external_semaphore") < submit.index("state.producer->ExecuteCommandLists")\n'
    text = replace_one(text, anchor, anchor +
        '    assert submit.index("handler.depends_on(wait)") < submit.index("state.producer->ExecuteCommandLists")\n'
        '    assert submit.index("state.producer->ExecuteCommandLists") < submit.index("q.ext_oneapi_signal_external_semaphore")\n')
    build.write_text(text, encoding="utf-8", newline="\n")

    runner = NEW / "luna_runner.py"
    text = runner.read_text(encoding="utf-8")
    text = replace_one(text,
        '        @callback_type\n        def callback(raw, _):\n            write_json(progress, snapshot(raw.contents))\n',
        '''        phases = []
        @callback_type
        def callback(raw, _):
            value = snapshot(raw.contents)
            value["host_monotonic_ns"] = time.monotonic_ns()
            phases.append({"stage": value["stage"], "api": value["api"],
                           "host_monotonic_ns": value["host_monotonic_ns"]})
            write_json(progress, value)
            write_json(output / "phases.json", phases)
''')
    text = replace_one(text,
        '        base.update(snapshot(result)); base["return_code"] = rc\n',
        '        base.update(snapshot(result)); base["return_code"] = rc\n'
        '        base["protocol_variant"] = "producer_admitted_before_SYCL_signal_v2"\n'
        '        base["host_phase_trace"] = phases\n'
        '        base["phase_timing_scope"] = "host callback intervals, includes diagnostic I/O; not GPU or pure API timing"\n')
    text = replace_one(text,
        '"scope": "single two-round DX12<->borrowed Torch SYCL capability check",',
        '"scope": "two-round DX12<->borrowed Torch SYCL; producer admitted before signal v2",')
    runner.write_text(text, encoding="utf-8", newline="\n")
    for path in NEW.rglob("*.py"):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for name, digest in old_pins.items():
        if sha(OLD / name) != digest:
            raise RuntimeError("Original source changed during staging")
    (NEW / "STAGE_RECEIPT.json").write_text(json.dumps({
        "status": "source_staged_GPU_unverified", "original_cpu_receipt": str(CPU),
        "original_cpu_receipt_sha256": sha(CPU), "original_failure": str(FAILED),
        "original_failure_sha256": sha(FAILED), "original_source_pins": old_pins,
        "source_pins": {p.relative_to(NEW).as_posix(): sha(p) for p in NEW.rglob("*") if p.is_file()},
        "GPU_executed": False, "G_writes": False,
        "difference": "SYCL wait/kernel before producer, producer before SYCL signal; all GPU feedback preserved",
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "staged", "path": str(NEW), "GPU_executed": False}))

if __name__ == "__main__":
    main()
