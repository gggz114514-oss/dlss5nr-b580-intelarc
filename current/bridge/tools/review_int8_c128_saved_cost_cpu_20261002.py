"""Recompute saved INT8 results without importing Torch or running a GPU.

Old launch Event spans are not pure matrix time. This receipt binds old
executed compiler keys to cache IR and records current-vs-old source scope.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import statistics


PROJECT = Path(__file__).resolve().parents[1]
OLD = Path(r"D:\Codex-NR-Experiments\nr-b580\re8-c128-int8-chain-720-20260928")
REPORTS = OLD / "smoke-rerun-8e610abfa6b3-20260928T225052651" / "reports"
CACHE = OLD / "smoke-8e610abfa6b3-20260928T223530966" / "cache"
CANDIDATE = Path(r"E:\ComfyUI-aki-v3-IntelArc_20260722\.codex-worktrees\re8-fp8-unround-fast-20260928\game\c128_continuous_int8_one_v1.py")
CURRENT_PAIR = PROJECT / "artifacts/vit-current-provider-gpu-runner-v4-20261002/source/parent/game/branched_mlp_pairwise_720_v1.py"
EXPECTED_REPORTS = {
    "smoke.json": "39f9a0ba941546d8cf6e15463db91920bae9e13f32e93faa9d9870d7891296d1",
    "segment-4warm-12measure.json": "75e69ac3bf70ed744f74d0dd2bf98ea94bbb4bedaa3e68446ee126f1ff540cf3",
    "kernel-times.json": "b46be28392339038f6eba2b1aeb30fd630822f838eba4b94d772976f71c125cc",
    "pairs-cost-attribution.json": "f73407e7620dca24996668e873f005bbd980a74e9ad40f39c7b5e650c83a24f5",
    "pair-tiles.json": "032df57debd860319f429e02be6df383bfccb007543aa7ee493c08b50902d817",
}
GROUPS = {
    "_pairs": "6RCGP45A5LFX7CYL2DAWOOGJXQV62JUCO2VDSSBMZPNO2M55SXFQ",
    "_project_dual": "EMHJSW73JPGLXVLQ4HHDMLDQOFQGVDL6E4J2FPOTBWNSTIVJ5UVQ",
    "_qkv_direct": "IW3OXWWUPXBFD7HIOM4GZVKAAHFG3UH7RFFQJOWYWKKN3TB6RJGA",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_function(path: Path, name: str) -> dict:
    text = path.read_text(encoding="utf-8")
    node = next(n for n in ast.parse(text).body
                if isinstance(n, ast.FunctionDef) and n.name == name)
    return {"path": str(path), "file_sha256": sha(path), "function": name,
            "start": node.lineno, "end": node.end_lineno,
            "body": ast.get_source_segment(text, node)}


def stats(values: list[float]) -> dict:
    assert values and all(math.isfinite(v) and v > 0 for v in values)
    order = sorted(values)
    return {"n": len(values), "mean_ms": statistics.fmean(values),
            "p50_ms": statistics.median(values),
            "nearest_rank_p95_ms": order[math.ceil(0.95 * len(order)) - 1],
            "min_ms": min(values), "max_ms": max(values)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    assert args.out.is_absolute() and args.out.drive.upper() == "D:"
    assert not args.out.exists(), "Keep prior receipts immutable"
    documents = {}
    pins = []
    for name, expected in EXPECTED_REPORTS.items():
        path = REPORTS / name
        actual = sha(path)
        assert actual == expected, (name, actual, expected)
        pins.append({"path": str(path), "sha256": actual})
        documents[name] = json.loads(path.read_text(encoding="utf-8"))
    smoke = documents["smoke.json"]
    source_expected = smoke["sources"]["files"]["candidate"]["sha256"]
    assert sha(CANDIDATE) == source_expected
    assert smoke["input"]["kind"] == "seeded_720p_rgb_motion"
    assert smoke["integer_reference_passed"] is True
    resources = smoke["segment_smoke"]["kernel_resources"]
    native = []
    for name, group in GROUPS.items():
        files = {ext: CACHE / group / (name + "." + ext)
                 for ext in ("json", "ttgir", "llir", "spv")}
        metadata = json.loads(files["json"].read_text(encoding="utf-8"))
        assert metadata["hash"] == resources[name]["hash"]
        ttgir = files["ttgir"].read_text(encoding="utf-8")
        llvm = files["llir"].read_text(encoding="utf-8")
        dots = [line.strip() for line in ttgir.splitlines() if "tt.dot " in line]
        calls = [line.strip() for line in llvm.splitlines()
                 if "call " in line and "SubgroupMatrixMultiplyAccumulateINTEL" in line]
        assert dots and all("xi8," in line and "xi32," in line for line in dots)
        assert calls and all("<8 x i32>" in line for line in calls)
        assert "dpas" in ttgir.lower()
        native.append({"kernel": name, "executed_key": metadata["hash"],
                       "recorded_resources": resources[name],
                       "compiler_settings": {k: metadata.get(k) for k in
                            ("num_warps", "num_stages", "threads_per_warp", "generate_native_code")},
                       "files": [{"path": str(p), "sha256": sha(p)} for p in files.values()],
                       "ttgir_dot_lines": dots, "llvm_intrinsic_calls": len(calls),
                       "proof_scope": "Saved executed key, integer DPAS IR and SPIR-V; not final physical ISA or measured ALU cycles"})

    segment = documents["segment-4warm-12measure.json"]
    assert [r["arm"] for r in segment["runs"]] == ["baseline", "candidate", "candidate", "baseline"]
    runs = [{"order": r["order"], "arm": r["arm"],
             "raw_ms": r["times_ms"], "recomputed": stats(r["times_ms"]),
             "output_sha256": r["output_sha256"]}
            for r in segment["runs"]]
    assert all(r["recomputed"]["n"] == 12 for r in runs)
    b = [v for r in runs if r["arm"] == "baseline" for v in r["raw_ms"]]
    c = [v for r in runs if r["arm"] == "candidate" for v in r["raw_ms"]]
    b_group_p50 = statistics.fmean(r["recomputed"]["p50_ms"] for r in runs if r["arm"] == "baseline")
    c_group_p50 = statistics.fmean(r["recomputed"]["p50_ms"] for r in runs if r["arm"] == "candidate")
    assert runs[0]["output_sha256"] == runs[3]["output_sha256"]
    assert runs[1]["output_sha256"] == runs[2]["output_sha256"]

    ablation = documents["pairs-cost-attribution.json"]
    assert ablation["source_sha256"] == source_expected
    a = ablation["timing"]
    tiles = documents["pair-tiles.json"]
    assert tiles["source_sha256"] == source_expected
    tile_groups = {}
    for r in tiles["timing"]:
        tile_groups.setdefault(r["name"], []).append(r["median_ms"])
    tile_average_group_p50 = {k: statistics.fmean(v) for k, v in tile_groups.items()}
    old_pair = source_function(CANDIDATE, "_pairs")
    current_pair = source_function(CURRENT_PAIR, "_pairs_pairwise")
    assert "for part in range(4)" in old_pair["body"]
    assert "for pair in range(2)" in current_pair["body"]
    assert current_pair["body"].count("tl.dot(x,") == 2
    assert "ROUND_REDUCED" in current_pair["body"]
    for rec in (old_pair, current_pair):
        del rec["body"]
    rows, channels, branches = 96 * 160, 128, 4
    logical_f16 = rows * channels * branches * 2 * 2
    logical_i8 = rows * channels * branches * 4 * 1
    assert logical_f16 == logical_i8
    result = {
        "passed": True, "cpu_only": True, "report_pins": pins,
        "old_input_provenance": smoke["input"],
        "old_misleading_activation_source_field": smoke["activation_source"],
        "correct_scope": "Old model activation from seeded RGB/motion, encoder[2][1], padded 104x168x128; not real game/video or current C512+K8",
        "integer_reference": smoke["integer_reference"],
        "native_int8_proof": native,
        "old_complete_block_event": {"runs": runs, "pooled_baseline": stats(b),
            "pooled_candidate": stats(c), "mean_delta_candidate_minus_baseline_ms": statistics.fmean(c) - statistics.fmean(b),
            "average_group_p50_baseline_ms": b_group_p50,
            "average_group_p50_candidate_ms": c_group_p50,
            "average_group_p50_delta_ms": c_group_p50 - b_group_p50,
            "scope": "MLP + QKV + unchanged rest of selected block, launch Events; not a graph, frame or pure matrix ALU measurement",
            "output_error_vs_old_fp16": segment["runs"][1]["difference_vs_first_baseline"]},
        "old_kernel_summary_only": documents["kernel-times.json"]["kernels"],
        "old_ablation_summary_only": {
            "raw_unavailable": True, "timing": a,
            "full_minus_without_lut_ms": a["full_variant"]["median_ms"] - a["without_LUT"]["median_ms"],
            "without_lut_minus_without_q12_and_lut_ms": a["without_LUT"]["median_ms"] - a["without_Q12_and_LUT"]["median_ms"],
            "caveat": "Same dot sequences, changed numeric semantics; diagnostic ablation, not acceptable product output or additive stage budget"},
        "old_tile_summary_only": {"raw_unavailable": True,
            "average_group_p50_ms": tile_average_group_p50,
            "best_int8_minus_old_fp16_ms": tile_average_group_p50["int8_32x1"] - tile_average_group_p50["fp16_current_32x2"],
            "int8_outputs_byte_equal": tiles["int8_tile_outputs_byte_equal"],
            "caveat": "Different old FP16 source, only pair kernel, padded geometry, launch Events"},
        "source_dataflow": {"integer_pair": old_pair, "current_pairwise_fp16": current_pair,
            "integer_qkv_consumes_qmlp_directly": True,
            "integer_project_publishes_both_int8_and_fp16_residual": True,
            "lut_semantics": "Per-channel INT8 cubic activation calibration, not FP8 encoding",
            "q12_semantics": "INT32 rescale into next INT8 range, not emulation of 4060 FP8 rounding",
            "logical_expansion_input_request_example": {
                "rows": rows, "channels": channels, "branches": branches,
                "fp16_input_reuses_per_branch": 2,
                "int8_input_reuses_per_branch": 4,
                "fp16_bytes_per_value": 2, "int8_bytes_per_value": 1,
                "fp16_logical_input_bytes": logical_f16,
                "int8_logical_input_bytes": logical_i8,
                "entry_logical_read_plus_write_bytes": rows * channels * 3,
                "caveat": "Static logical requests on current unpadded geometry; compiler/cache reuse may reduce DRAM traffic; not measured bandwidth or cost attribution"}},
        "next_required_check": "Current owned 720p C128 fixture: preallocated complete FP16/INT8 modules and repeated graph Event samples; independently measure quantization entry and prequantized core, retaining correct activation and consumer dataflow",
        "not_claimed": ["All prior INT8 kernels were non-native", "Whole-game speedup from old local data", "All INT8 overhead is FP16 conversion", "Final driver physical ISA inspected", "INT8 is intrinsically slower", "All lookup tables are removable FP8 emulation"]}
    related = [
        ("old_C512", Path(r"D:\Codex-NR-Experiments\nr-b580\pure-int8-c512-full-v1\first-c512-full-probe.json"),
         "f7e1412a9e925d1e7d8576ead40d8711810d94fe988afcf96bce59c2cd0af94b", "timings",
         ["existing_mixed_ffn_five_kernels", "integer_chain_i32_if_safe", "integer_chain_i64", "entry_integer_exit"]),
        ("old_C32", Path(r"D:\Codex-NR-Experiments\nr-b580\fixed-int8-c32-v1\probe-direct-chain.json"),
         "81c08648a3465cb26136ef57174dc4d722d0f6abe18a683e48c5bfef46a41169", "timing",
         ["existing_full_chain", "int8_two_kernel_direct_chain", "int8_full_chain_with_both_boundaries"]),
    ]
    result["related_old_raw_recomputed"] = {}
    for label, path, expected, field, arms in related:
        assert sha(path) == expected
        doc = json.loads(path.read_text(encoding="utf-8"))
        result["related_old_raw_recomputed"][label] = {
            "path": str(path), "sha256": expected,
            "arms": {arm: {"raw_ms": doc[field][arm]["samples_ms"],
                            "recomputed": stats(doc[field][arm]["samples_ms"])} for arm in arms},
            "baseline_matches_captured_output": doc.get("baseline_mixed_matches_captured_output"),
            "scope": "Old offline launch Event experiment; neither current C512+K8 provider nor game FPS; independent stage medians are non-additive",
        }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"passed": True, "out": str(args.out), "sha256": sha(args.out),
                      "old_block_group_p50_delta_ms": result["old_complete_block_event"]["average_group_p50_delta_ms"],
                      "native_int8_kernels_bound": len(native),
                      "logical_input_bytes_equal": logical_f16 == logical_i8}, ensure_ascii=False))


if __name__ == "__main__":
    main()
