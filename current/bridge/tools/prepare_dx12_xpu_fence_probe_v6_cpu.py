"""Extend the proven separate-import protocol to 32 feedback-linked rounds."""
import ast
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
OLD = PROJECT / "artifacts/dx12-xpu-fence-probe-v5-20261002"
NEW = PROJECT / "artifacts/dx12-xpu-fence-probe-v6-20261002"
DATA = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\dx12-xpu-fence-probe-v5-20261002")


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cpu = json.loads((DATA / "cpu-pckblmyb/CPU_COMPILE_RECEIPT.json").read_text())
    passed = DATA / "runtime-trace-v1/RUNTIME_RECEIPT.json"
    receipt = json.loads(passed.read_text())
    if not receipt["passed"] or not receipt["child_result"]["cleanup_ok"]:
        raise RuntimeError("Retired v5 pass required")
    for name, expected in cpu["source_pins"].items():
        if sha(OLD / name) != expected: raise RuntimeError("Frozen source changed: " + name)
    if NEW.exists(): raise RuntimeError("Fresh source stage required")
    NEW.mkdir()
    for name in (*cpu["source_pins"], "INPUT_PINS.json"):
        target = NEW / name; target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((OLD / name).read_bytes())
    replacements = {
        "native/probe_patterns.h": [("words=256,rounds=2", "words=256,rounds=32")],
        "native/dx12_xpu_fence_probe_v1.h": [("NR_FENCE_PROBE_ROUNDS 2u", "NR_FENCE_PROBE_ROUNDS 32u")],
        "tests/abi_pattern_shader_cpu.cpp": [("NR_FENCE_PROBE_ROUNDS==2", "NR_FENCE_PROBE_ROUNDS==32")],
        "native/dx12_xpu_fence_probe_v1.cpp": [
            ("std::array<Commands,2>", "std::array<Commands,fence_probe::rounds>"),
            ("state.events.reserve(6)", "state.events.reserve(fence_probe::rounds*3)"),
            ("fence_probe::consumer_value(1)", "fence_probe::consumer_value(fence_probe::rounds-1)"),
            ("out.gpu_wait_calls==5 && out.gpu_signal_calls==6", "out.gpu_wait_calls==fence_probe::rounds*3-1 && out.gpu_signal_calls==fence_probe::rounds*3"),
            ("final SetEventOnCompletion(consumer ack31)", "final SetEventOnCompletion(last consumer ack)"),
            ("both rounds actual integer-pattern comparison", "all32 rounds actual integer-pattern comparison")],
        "build_cpu.py": [("dx12-xpu-fence-probe-v5-20261002", "dx12-xpu-fence-probe-v6-20261002")],
        "luna_runner.py": [
            ("WORDS, ROUNDS = 256, 2", "WORDS, ROUNDS = 256, 32"),
            ("result.rounds_checked == 2", "result.rounds_checked == ROUNDS"),
            ("result.gpu_wait_calls == 5 and result.gpu_signal_calls == 6", "result.gpu_wait_calls == ROUNDS*3-1 and result.gpu_signal_calls == ROUNDS*3"),
            ("same_forward_fence_separate_imports_v5", "same_forward_fence_32_separate_imports_v6"),
            ("two-round DX12<->borrowed Torch SYCL; producer admitted before signal v2", "32-round DX12<->same borrowed Torch SYCL; one forward DX12 fence, distinct imports")],
    }
    for name, changes in replacements.items():
        path = NEW / name; text = path.read_text(encoding="utf-8")
        for old, new in changes:
            if old not in text: raise RuntimeError("Changed anchor: " + old)
            text = text.replace(old, new)
        path.write_text(text, encoding="utf-8", newline="\n")
    for path in NEW.rglob("*.py"): ast.parse(path.read_text(encoding="utf-8"))
    status = {"status": "source_staged_GPU_unverified", "previous_pass": str(passed),
        "previous_pass_sha256": sha(passed), "rounds": 32, "expected_words": 8192,
        "difference": "Same v5 protocol extended32 rounds. All initial forward imports use same DX12 fence. Not yet dynamic per-frame import release or product acceptance.",
        "acceptance": "All8192 values,95GPUwaits96GPU signals,same queue/context,actual terminal cleanup,noCPUhandoff.",
        "source_pins": {p.relative_to(NEW).as_posix(): sha(p) for p in NEW.rglob("*") if p.is_file()},
        "GPU_executed": False, "G_writes": False}
    (NEW / "STAGE_RECEIPT.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"status": "staged", "path": str(NEW)}))


if __name__ == "__main__": main()
