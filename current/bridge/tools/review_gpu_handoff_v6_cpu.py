"""Independently inspect recorded GPU results; never load the GPU runtime."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
DATA = Path(r"D:\Codex-NR-Experiments\cyberpunk-opt\dx12-xpu-fence-probe-v6-20261002")
MASK = (1 << 32) - 1


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotate(value: int, bits: int) -> int:
    value &= MASK
    return ((value << bits) | (value >> (32 - bits))) & MASK


def expected_frame(previous: list[int], frame: int) -> list[int]:
    values = []
    for i, before in enumerate(previous):
        produced = (0x13579BDF ^ (i * 0x10203 + 7)) if frame == 0 else (
            ((before + 0x2468ACE1 + i * 13) & MASK) ^ 0x55AA33CC)
        modified = (rotate(produced ^ ((0xA5A55A5A + frame * 0x1020304) & MASK), 5)
                    + i * 17 + frame) & MASK
        values.append(rotate(modified + 0x31415927 + frame + i * 3, 11) ^ 0xC001D00D)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, default=DATA / "runtime-v1/RUNTIME_RECEIPT.json")
    parser.add_argument("--build", type=Path, default=DATA / "cpu-frkbmctl/CPU_COMPILE_RECEIPT.json")
    parser.add_argument("--output", type=Path, default=DATA / "MAIN_ORDER_REVIEW.json")
    args = parser.parse_args()
    if args.output.resolve().drive.upper() != "D:":
        raise RuntimeError("Review output must remain on D")
    run = json.loads(args.receipt.read_text(encoding="utf-8"))
    build = json.loads(args.build.read_text(encoding="utf-8"))
    stage = PROJECT / "artifacts/dx12-xpu-fence-probe-v6-20261002"
    for name, pin in build["source_pins"].items():
        if sha(stage / name) != pin:
            raise RuntimeError("Changed compiled source: " + name)
    if sha(Path(build["dll"])) != build["dll_sha256"]:
        raise RuntimeError("Changed compiled DLL")
    if run["status"] != "passed" or run["exit_code"] != 0 or run["watchdog_terminated"]:
        raise RuntimeError("Probe did not finish successfully")
    child = run["child_result"]
    for key in ("passed", "queue_equal", "context_equal", "in_order", "cleanup_ok"):
        if not child[key]:
            raise RuntimeError("Missing successful runtime gate: " + key)
    if child["quarantined"] or child["rounds_checked"] != 32:
        raise RuntimeError("Incomplete terminal retirement")
    if (child["gpu_wait_calls"], child["gpu_signal_calls"], child["cpu_final_wait_calls"]) != (95, 96, 1):
        raise RuntimeError("Unexpected synchronization counts")
    if (child["producer_completed"], child["xpu_completed"], child["consumer_completed"]) != (41, 51, 61):
        raise RuntimeError("Final fence values differ")
    previous = [0] * 256
    if len(child["rounds"]) != 32:
        raise RuntimeError("Round count differs")
    for index, row in enumerate(child["rounds"]):
        previous = expected_frame(previous, index)
        if row["observed"] != previous or row["checked_words"] != 256:
            raise RuntimeError(f"Independent integer check failed at round {index}")
        if (row["producer_value"], row["xpu_value"], row["consumer_value"]) != (10 + index, 20 + index, 30 + index):
            raise RuntimeError(f"Fence sequence differs at round {index}")
    result = {
        "status": "recorded_GPU_results_independently_verified",
        "receipt": str(args.receipt), "receipt_sha256": sha(args.receipt),
        "compiled_source_pins_unchanged": True, "integer_words_checked": 8192,
        "rounds": 32, "gpu_wait_calls": 95, "gpu_signal_calls": 96,
        "cpu_wait_scope": "one final readback audit, no intermediate handoff waits",
        "actual_terminal_cleanup": True,
        "scope": "same D3D12 fence with a different precreated import for each wait; product and dynamic import lifetime unverified",
        "GPU_executed_by_main": False, "G_writes": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
