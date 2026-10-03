"""Classify VTune computing-task rows without pretending unknown kernels are math."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


CATEGORIES = (
    ("explicit_memcopy", re.compile(r"^zeCommandListAppendMemoryCopy")),
    ("copy_or_fill_kernel", re.compile(r"CopyScalarFunc|FillFunctor")),
    # A dot kernel also loads, stages and stores data; its full GPU duration is
    # not a measurement of arithmetic instructions alone.
    ("named_matrix_mixed", re.compile(r"_matmul|_dot|_gemm|_mma")),
    # _pairs includes cubic lookup and FP8 rounding; _project includes skip
    # and output writes; _tiled includes shared-exponent conversion.
    ("named_fused_mixed", re.compile(r"^(_pairs|_project|_tiled)$")),
    ("named_pack_quant", re.compile(r"^_pack|^_quantized|_quantize")),
    ("named_elementwise", re.compile(r"^VectorizedElementwiseKernel")),
    ("ambiguous_triton_kernel", re.compile(r"^_kernel")),
)


def classify_task(name: str) -> str:
    return next((kind for kind, pattern in CATEGORIES
                 if pattern.search(name)), "other")


def summarize(path: Path) -> dict:
    result = {"source": str(path.resolve()), "categories": {}}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            name = row["Computing Task"]
            kind = classify_task(name)
            category = result["categories"].setdefault(
                kind, {"seconds": 0.0, "instances": 0, "rows": 0})
            category["seconds"] += float(row["Computing Task:Total Time"])
            category["instances"] += int(row["Computing Task:Instance Count"])
            category["rows"] += 1
    total = sum(item["seconds"] for item in result["categories"].values())
    result["task_duration_sum_seconds"] = total
    for category in result["categories"].values():
        category["share_of_task_duration_percent"] = 100 * category["seconds"] / total
    result["interpretation"] = (
        "Rows are grouped kernel names; task durations may overlap and are not "
        "wall-clock time. Matrix and name-matched fused kernels are mixed "
        "compute/load/store work, not pure arithmetic time. Generic kernels "
        "remain unclassified. Pack/quantize also perform numeric work and "
        "are not equivalent to pure memory transfer.")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
