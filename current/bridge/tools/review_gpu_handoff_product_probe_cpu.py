"""Recheck frozen build and Luna's texture/graph receipts without loading a GPU DLL."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def review(cpu_path, runtime_path, stage):
    cpu = json.loads(cpu_path.read_text(encoding="utf-8"))
    check(cpu["status"] == "CPU_compiled_passed_GPU_UNTESTED", "CPU build did not pass")
    check(cpu["GPU_created_or_executed"] is False, "CPU build ran GPU work")
    check(cpu["installed_old_ABI_preserved"] is True, "Old native ABI was not preserved")
    for name, expected in cpu["source_pins"].items():
        check(digest(stage / name) == expected, "Build source drift: " + name)
    for name, expected in cpu["input_pins"].items():
        check(digest(name) == expected, "Pinned input drift: " + name)
    for name in ("candidate", "probe"):
        check(digest(cpu[name + "_DLL"]) == cpu[name + "_sha256"], "Binary drift: " + name)
    check(all(cpu["structure"].get(k) is True for k in (
        "HLSL_exact_unchanged", "legacy_prepare_export_read_inputs_unchanged",
        "old_C_ABI_header_prefix_and_Frame_ABI_unchanged", "no_explicit_CPU_frame_handoff_wait",
        "pack_submit_signal_precedes_XPU_signal", "separate_pack_unpack_records",
        "per_frame_forward_import", "actual_completion_event_and_consumer_proof",
    )), "Native structure contract failed")
    output = {
        "status": "CPU_build_pins_reviewed_GPU_UNTESTED",
        "cpu_receipt": str(cpu_path), "cpu_receipt_sha256": digest(cpu_path),
        "source_pins_checked": len(cpu["source_pins"]),
        "input_pins_checked": len(cpu["input_pins"]),
        "main_GPU_executed": False, "game_or_performance_acceptance": False,
    }
    if runtime_path is None:
        return output
    runtime = json.loads(runtime_path.read_text(encoding="utf-8"))
    check(runtime["exit_code"] == 0 and runtime["watchdog_terminated"] is False,
          "GPU child did not exit cleanly")
    child = runtime["child_result"]
    check(child["passed"] is True and child["GPU_attempted"] is True, "GPU probe did not pass")
    check(child["whole_model_or_game_test"] is False and child["performance_measured"] is False,
          "Wrong proof scope")
    loaded = child["actual_loaded_sycl"]
    check(loaded["sha256"] == cpu["input_pins"][loaded["path"]], "Loaded SYCL is not the pinned runtime")
    native = child["native_probe"]
    check(native["status"] == 0 and native["cleanup_ok"] == 1 and not native["quarantined"],
          "Native cleanup failed")
    for key, count in (("frames_compared", 34), ("async_frames_compared", 32),
                       ("off_controls_compared", 2), ("byte_checks", 204),
                       ("producer_gate_returns", 32)):
        check(native[key] == count, "Incomplete native coverage: " + key)
    native_stats = native["final_stats"]
    for key in ("enabled", "active", "poisoned", "forward_live"):
        check(native_stats[key] == 0, "Native owner still active: " + key)
    for key in ("frames_started", "frames_retired", "forward_imports", "forward_releases",
                "pack_submits", "unpack_submits", "xpu_waits", "xpu_signals", "consumer_registrations"):
        check(native_stats[key] == 32, "Native lease count mismatch: " + key)
    graph = child["Torch_graph_probe"]
    check(graph["status"] == "passed" and graph["cleanup_ok"] and not graph["quarantined"],
          "Torch graph probe failed")
    for key, count in (("real_shared_texture_frames", 32), ("Torch_graph_captures", 3),
                       ("Torch_graph_replays", 96), ("output_texture_byte_comparisons", 64),
                       ("input_byte_comparisons", 128), ("nonzero_motion_checks", 96)):
        check(graph[key] == count, "Incomplete graph coverage: " + key)
    check(graph["actual_native_retirement_and_tensor_refs_released"] is True,
          "Tensor/consumer retirement not proven")
    check(graph["same_current_stream_queue"] == child["borrowed_queue"] == native["borrowed_queue"],
          "Graph and native queues differ")
    for key in ("frames_started", "frames_retired", "forward_imports", "forward_releases",
                "pack_submits", "unpack_submits", "xpu_waits", "xpu_signals", "consumer_registrations"):
        check(graph["final_async_stats"][key] == 32, "Graph lease count mismatch: " + key)
    check(graph["final_async_stats"]["forward_live"] == 0, "Graph forward import still live")
    output.update(status="native_and_Torch_graph_texture_protocol_reviewed",
                  runtime_receipt=str(runtime_path), runtime_receipt_sha256=digest(runtime_path),
                  native_frames=34, native_byte_checks=204, graph_frames=32, graph_replays=96,
                  no_input_output_algorithm_changes=True, actual_cleanup_proven=True,
                  performance_measured=False)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cpu-receipt", type=Path, required=True)
    parser.add_argument("--runtime-receipt", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", type=Path, default=Path(__file__).resolve().parents[1] /
                        "artifacts/gpu-handoff-bridge-v1-20261002")
    args = parser.parse_args()
    result = review(args.cpu_receipt, args.runtime_receipt, args.stage)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
