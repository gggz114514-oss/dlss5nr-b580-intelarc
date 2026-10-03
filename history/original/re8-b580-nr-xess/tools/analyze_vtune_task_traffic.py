"""Offline VTune computing-task classification and bandwidth exposure report.

This consumes an already exported TSV. It neither collects GPU counters nor
imports the game backend. In VTune's grouped report, bandwidth can describe
GPU-wide sampled activity while task durations can overlap. Consequently
bandwidth * grouped task time is a *proxy*, never attributed DRAM bytes or a
decomposition of kernel time into compute and memory stalls.

Usage:
    python analyze_vtune_task_traffic.py existing.tsv --output-dir D:/analysis
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re


CATEGORY_ORDER = (
    "explicit_copy",
    "layout_or_fill",
    "quantization_or_pack_mixed",
    "matrix_mixed",
    "fusion_mixed",
    "unclassified",
)

# Only names with an identified role receive a non-unknown class. A matrix
# kernel loads/stores; pack and fused kernels may do numerical work too.
COPY = re.compile(r"^zeCommandListAppendMemoryCopy", re.I)
LAYOUT = re.compile(r"CopyScalarFunc|FillFunctor|^zeCommandListAppendMemoryFill", re.I)
QUANT = re.compile(r"^_pack$|^_entry$|(?:de)?quantiz|^_q(?:uant)?_", re.I)
MATRIX = re.compile(r"matmul|_dot|gemm|_mma|^_expand$|^_project$|^_tiled$|^_quantized$", re.I)
FUSION = re.compile(r"^_(?:pairs|five_tap|unpack|reduce|axis|prepare_axes)$", re.I)

BANDWIDTH = {
    "gpu_memory": (
        "GPU Memory Bandwidth, GB/sec:Read",
        "GPU Memory Bandwidth, GB/sec:Write",
    ),
    "gpu_l3": (
        "GPU L3:Average Bandwidth, GB/s:Read",
        "GPU L3:Average Bandwidth, GB/s:Write",
    ),
}


def classify(name: str) -> str:
    if COPY.search(name):
        return "explicit_copy"
    if LAYOUT.search(name):
        return "layout_or_fill"
    if QUANT.search(name):
        return "quantization_or_pack_mixed"
    if MATRIX.search(name):
        return "matrix_mixed"
    if FUSION.search(name):
        return "fusion_mixed"
    return "unclassified"


def number(value: str | None) -> float | None:
    if value is None or value.strip() in ("", "-", "N/A", "n/a"):
        return None
    try:
        parsed = float(value.strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def new_category() -> dict:
    return {
        "rows": 0,
        "instances": 0,
        "task_duration_sum_seconds": 0.0,
        "explicit_transfer_size_bytes": 0,
        "bandwidth": {
            source: {
                direction: {
                    "valid_rows": 0,
                    "positive_rows": 0,
                    "covered_task_seconds": 0.0,
                    "duration_weighted_proxy_GB": 0.0,
                    "maximum_observed_GB_per_second": 0.0,
                }
                for direction in ("read", "write")
            }
            for source in BANDWIDTH
        },
    }


def analyze(path: Path) -> dict:
    categories = {name: new_category() for name in CATEGORY_ORDER}
    unclassified: dict[str, dict[str, float | int]] = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        columns = set(reader.fieldnames or ())
        required = {
            "Computing Task", "Computing Task:Total Time",
            "Computing Task:Instance Count",
        }
        missing = sorted(required - columns)
        if missing:
            raise ValueError(f"Missing required TSV columns: {', '.join(missing)}")
        for line_no, row in enumerate(reader, 2):
            name = (row.get("Computing Task") or "").strip()
            duration = number(row.get("Computing Task:Total Time"))
            count = number(row.get("Computing Task:Instance Count"))
            if not name or duration is None or duration < 0 or count is None or count < 0 or not count.is_integer():
                raise ValueError(f"Invalid task, total time, or instance count at line {line_no}")
            kind = classify(name)
            category = categories[kind]
            category["rows"] += 1
            category["instances"] += int(count)
            category["task_duration_sum_seconds"] += duration
            if kind == "explicit_copy":
                transfer = number(row.get("Transfer Size"))
                if transfer is not None and transfer >= 0:
                    category["explicit_transfer_size_bytes"] += int(transfer)
            if kind == "unclassified":
                unknown = unclassified.setdefault(name, {"rows": 0, "task_duration_sum_seconds": 0.0})
                unknown["rows"] += 1
                unknown["task_duration_sum_seconds"] += duration
            for source, (read_col, write_col) in BANDWIDTH.items():
                for direction, col in (("read", read_col), ("write", write_col)):
                    bw = number(row.get(col))
                    if bw is None or bw < 0:
                        continue
                    measure = category["bandwidth"][source][direction]
                    measure["valid_rows"] += 1
                    measure["positive_rows"] += int(bw > 0)
                    measure["covered_task_seconds"] += duration
                    measure["duration_weighted_proxy_GB"] += bw * duration
                    measure["maximum_observed_GB_per_second"] = max(
                        measure["maximum_observed_GB_per_second"], bw
                    )

    duration_sum = sum(c["task_duration_sum_seconds"] for c in categories.values())
    for category in categories.values():
        duration = category["task_duration_sum_seconds"]
        category["share_of_task_duration_percent"] = (
            100 * duration / duration_sum if duration_sum else 0.0
        )
        for source in BANDWIDTH:
            for direction in ("read", "write"):
                measure = category["bandwidth"][source][direction]
                # A loose rate envelope conditional on the report's sampled
                # maximum applying to all covered task-seconds. It is NOT a
                # rigorous physical-memory traffic upper bound.
                measure["conditional_rate_envelope_GB"] = (
                    measure["maximum_observed_GB_per_second"]
                    * measure["covered_task_seconds"]
                )

    copy = categories["explicit_copy"]
    copy_memory = copy["bandwidth"]["gpu_memory"]
    copy_proxy = sum(copy_memory[d]["duration_weighted_proxy_GB"]
                     for d in ("read", "write"))

    return {
        "source": str(path.resolve()),
        "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": sum(c["rows"] for c in categories.values()),
        "instances": sum(c["instances"] for c in categories.values()),
        "task_duration_sum_seconds": duration_sum,
        "available_bandwidth_columns": {
            source: {"read": pair[0] in columns, "write": pair[1] in columns}
            for source, pair in BANDWIDTH.items()
        },
        "categories": categories,
        "cross_metric_check": {
            "explicit_copy_transfer_size_GB": copy["explicit_transfer_size_bytes"] / 1e9,
            "explicit_copy_gpu_memory_bandwidth_proxy_read_plus_write_GB": copy_proxy,
            "ratio_proxy_to_reported_transfer_size": (
                copy_proxy / (copy["explicit_transfer_size_bytes"] / 1e9)
                if copy["explicit_transfer_size_bytes"] else None
            ),
            "caveat": "Transfer Size and GPU memory bandwidth cover different memory domains and reporting scopes. Their ratio is a comparability warning, not an error or a correction factor.",
        },
        "top_unclassified": [
            {"name": name, **values}
            for name, values in sorted(
                unclassified.items(),
                key=lambda item: item[1]["task_duration_sum_seconds"],
                reverse=True,
            )[:12]
        ],
        "interpretation": [
            "Computing Task:Total Time is the sum of grouped GPU task durations, not frame wall time; task instances and queues may overlap.",
            "GPU Memory Bandwidth columns are sampled rates in decimal GB/s. Their attribution to a grouped task is not guaranteed to be exclusive.",
            "duration_weighted_proxy_GB = sum(rate_GB/s * grouped_task_seconds) for rows with valid bandwidth. It is a naive exposure proxy, not measured bytes for that category.",
            "conditional_rate_envelope_GB = max_observed_rate * covered_task_seconds. This is conditional and is not a strict upper bound on physical traffic.",
            "No reliable numeric upper bound on actual per-category physical traffic can be extracted from this grouped TSV; the reported envelope only bounds the rate-weighted proxy under its stated assumptions.",
            "GPU L3 is on-chip cache traffic and must not be added to GPU memory traffic. Cache hits, overlap and sampling gaps can make either proxy misleading.",
            "Explicit Transfer Size is reported by VTune for copy tasks; it is separate from the bandwidth-derived proxy and must not be added to it.",
            "A matrix, quantization or fusion kernel also loads and stores. Kernel time cannot be divided into pure arithmetic and pure movement from this TSV.",
            "CopyScalarFunc/FillFunctor are placed in layout_or_fill by name; a copy may cast values, so this class is not a proof of memory-only instructions. _pack/_entry are quantization-or-pack mixed, not pure quantization time.",
            "Generic _kernel names remain unclassified instead of treating VTune's Compute label as matrix work.",
        ],
        "next_gpu_sampling": [
            "Record per-instance GPU start/end timestamps, queue and dependency edges, not only grouped task names; bracket graph replay with same-queue stage markers.",
            "Capture the actual 360p/480p/540p model and repeated frames with nonzero motion; map kernels to pre, encoder, C512, ViT, decoder and post.",
            "Where supported, collect GPU memory read/write counters, L3/SLM traffic, XMX activity, occupancy, register and spill metrics alongside per-instance events.",
            "Compare profiler stage intervals separately from no-profiler B-C-C-B whole-frame timings and continuous-history output checks.",
        ],
    }


def markdown(result: dict) -> str:
    def shown(source: str, direction: str, value: float) -> str:
        return (f"{value:.6f}" if result["available_bandwidth_columns"][source][direction]
                else "n/a")

    lines = [
        "# VTune 540p 稳态任务与带宽离线分析",
        "",
        f"源文件：`{result['source']}`",
        f"SHA-256：`{result['source_sha256']}`",
        f"任务表行数 {result['rows']}，实例数 {result['instances']}；累计任务时长 {result['task_duration_sum_seconds']:.6f} s。",
        "",
        "| 类别 | 任务累计 ms | 占任务累计 % | 显式拷贝字节 | GPU 显存读代理 GB | 写代理 GB | 条件速率包络读/写 GB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, category in result["categories"].items():
        memory = category["bandwidth"]["gpu_memory"]
        read, write = memory["read"], memory["write"]
        lines.append(
            f"| {name} | {category['task_duration_sum_seconds']*1000:.3f} | "
            f"{category['share_of_task_duration_percent']:.2f} | "
            f"{category['explicit_transfer_size_bytes']:,} | "
            f"{shown('gpu_memory', 'read', read['duration_weighted_proxy_GB'])} | "
            f"{shown('gpu_memory', 'write', write['duration_weighted_proxy_GB'])} | "
            f"{shown('gpu_memory', 'read', read['conditional_rate_envelope_GB'])} / "
            f"{shown('gpu_memory', 'write', write['conditional_rate_envelope_GB'])} |"
        )
    lines += [
        "",
        "带宽列覆盖（有效行数／该类总行数；零值可能是有效读数，也可能是采样空洞）：",
        "",
        "| 类别 | GPU 显存读 | GPU 显存写 | L3 读 | L3 写 | L3 读/写代理 GB |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, category in result["categories"].items():
        gm = category["bandwidth"]["gpu_memory"]
        l3 = category["bandwidth"]["gpu_l3"]
        total = category["rows"]
        lines.append(
            f"| {name} | {gm['read']['valid_rows']}/{total} | "
            f"{gm['write']['valid_rows']}/{total} | {l3['read']['valid_rows']}/{total} | "
            f"{l3['write']['valid_rows']}/{total} | "
            f"{shown('gpu_l3', 'read', l3['read']['duration_weighted_proxy_GB'])} / "
            f"{shown('gpu_l3', 'write', l3['write']['duration_weighted_proxy_GB'])} |"
        )
    check = result["cross_metric_check"]
    lines += [
        "",
        f"交叉口径提醒：显式拷贝 `Transfer Size` 为 {check['explicit_copy_transfer_size_GB']:.3f} GB，"
        f"同类 GPU 显存带宽×任务时间的读写代理合计 {check['explicit_copy_gpu_memory_bandwidth_proxy_read_plus_write_GB']:.3f} GB。"
        "两者覆盖的内存域和采样口径不同，不能用后者当作真实拷贝字节或物理流量严格上界。",
    ]
    lines += ["", "## 口径与限制", ""]
    lines += [f"- {item}" for item in result["interpretation"]]
    lines += ["", "## 仍需识别的主要任务名", ""]
    for item in result["top_unclassified"]:
        lines.append(f"- `{item['name'][:120]}`：{item['task_duration_sum_seconds']*1000:.3f} ms")
    lines += ["", "## 下一轮 GPU 采样", ""]
    lines += [f"- {item}" for item in result["next_gpu_sampling"]]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tsv", type=Path, help="Existing VTune computing-task TSV")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(args.tsv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "vtune_task_traffic.json"
    md_path = args.output_dir / "vtune_task_traffic.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"json": str(json_path.resolve()), "markdown": str(md_path.resolve()),
                      "rows": report["rows"], "instances": report["instances"],
                      "task_duration_sum_seconds": report["task_duration_sum_seconds"]},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
