"""Sequential GPU pipeline: capture the real chain, gate it, then benchmark."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_logged(argv: list[str], stdout_path: Path, stderr_path: Path) -> int:
    with stdout_path.open("w", encoding="utf-8", newline="\n") as stdout, \
            stderr_path.open("w", encoding="utf-8", newline="\n") as stderr:
        process = subprocess.Popen(argv, stdout=stdout, stderr=stderr, text=True)
        return process.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--exact-validation", type=Path)
    parser.add_argument("--baseline-validation", type=Path)
    parser.add_argument("--benchmark-only", action="store_true")
    parser.add_argument("--capture-dir", type=Path)
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--warmups", type=int, default=5)
    args = parser.parse_args()
    out, cache = args.out.resolve(), args.cache.resolve()
    capture_dir = out / "capture"
    if args.benchmark_only:
        capture_dir = (args.capture_dir or capture_dir).resolve()
        old_pipeline_path = out / "pipeline.json"
        old_pipeline = json.loads(old_pipeline_path.read_text(encoding="utf-8-sig")) \
            if old_pipeline_path.is_file() else {}
        capture_json = capture_dir / "capture.json"
        validation_json = capture_dir / "run" / "validation.json"
        if not capture_json.is_file() or not validation_json.is_file():
            raise RuntimeError("Benchmark-only mode requires a completed capture and validation report")
        capture_report = json.loads(capture_json.read_text(encoding="utf-8-sig"))
        validation_report = json.loads(validation_json.read_text(encoding="utf-8-sig"))
        old_rc = old_pipeline.get("returncode") if old_pipeline.get("stage") == "capture" else 0
        if old_rc not in (0, 3221226505) or not capture_report.get("passed") or not validation_report.get("passed"):
            raise RuntimeError(f"Capture gate failed: old_rc={old_rc}, capture={capture_report.get('passed')}, "
                               f"validation={validation_report.get('passed')}")
        if old_pipeline_path.is_file() and not (out / "capture-pipeline-original.json").exists():
            (out / "capture-pipeline-original.json").write_bytes(old_pipeline_path.read_bytes())

        check_cmd = [sys.executable, "-B", "-X", "utf8", "-u", str(HERE / "benchmark_chain.py"),
                     "--check", "--capture-dir", str(capture_dir)]
        check_rc = run_logged(check_cmd, out / "capture-check.stdout.log", out / "capture-check.stderr.log")
        if check_rc != 0:
            raise RuntimeError(f"Runtime capture identity gate failed with rc={check_rc}")
        bench_cmd = [sys.executable, "-B", "-X", "utf8", "-u", str(HERE / "benchmark_chain.py"),
                     "--capture-dir", str(capture_dir), "--cache", str(cache),
                     "--out", str(out / "result.json"), "--repeats", str(args.repeats),
                     "--warmups", str(args.warmups)]
        print("[chain] real capture and runtime identity passed; benchmark under current lease", flush=True)
        bench_rc = run_logged(bench_cmd, out / "benchmark.stdout.log", out / "benchmark.stderr.log")
        result_path = out / "result.json"
        pipeline = {"passed": bench_rc == 0, "stage": "benchmark", "returncode": bench_rc,
                    "capture_returncode": old_rc,
                    "known_teardown_failfast_tolerated": old_rc == 3221226505,
                    "capture_report": str(capture_json), "result": str(result_path) if result_path.is_file() else None}
        (out / "pipeline.json").write_text(json.dumps(pipeline, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(pipeline, ensure_ascii=False), flush=True)
        return bench_rc

    if args.exact_validation is None or args.baseline_validation is None:
        parser.error("Capture mode requires --exact-validation and --baseline-validation")
    capture_cmd = [sys.executable, "-B", "-X", "utf8", "-u", str(HERE / "capture_chain.py"),
                   "--out", str(capture_dir), "--cache", str(cache),
                   "--exact-validation", str(args.exact_validation.resolve()),
                   "--baseline-validation", str(args.baseline_validation.resolve()),
                   "--limit", str(args.limit)]
    print("[chain] capture actual fast-line producer/FP8/C32 consumer", flush=True)
    capture_rc = run_logged(capture_cmd, out / "capture.stdout.log", out / "capture.stderr.log")
    if capture_rc != 0:
        (out / "pipeline.json").write_text(json.dumps({"passed": False,
            "stage": "capture", "returncode": capture_rc}, indent=2) + "\n", encoding="utf-8")
        print(f"[chain] capture failed rc={capture_rc}", flush=True)
        return capture_rc

    check_cmd = [sys.executable, "-B", "-X", "utf8", "-u", str(HERE / "benchmark_chain.py"),
                 "--check", "--capture-dir", str(capture_dir)]
    check_rc = run_logged(check_cmd, out / "capture-check.stdout.log", out / "capture-check.stderr.log")
    if check_rc != 0:
        (out / "pipeline.json").write_text(json.dumps({"passed": False,
            "stage": "runtime-capture-match", "returncode": check_rc}, indent=2) + "\n", encoding="utf-8")
        print(f"[chain] runtime capture identity check failed rc={check_rc}; benchmark skipped", flush=True)
        return check_rc

    bench_cmd = [sys.executable, "-B", "-X", "utf8", "-u", str(HERE / "benchmark_chain.py"),
                 "--capture-dir", str(capture_dir), "--cache", str(cache),
                 "--out", str(out / "result.json"), "--repeats", str(args.repeats),
                 "--warmups", str(args.warmups)]
    print("[chain] capture identity verified; warm and time split/fused/FP16/two-kernel paths", flush=True)
    bench_rc = run_logged(bench_cmd, out / "benchmark.stdout.log", out / "benchmark.stderr.log")
    result_path = out / "result.json"
    pipeline = {"passed": bench_rc == 0, "stage": "benchmark", "returncode": bench_rc,
                "capture_report": str(capture_dir / "capture.json"),
                "result": str(result_path) if result_path.is_file() else None}
    (out / "pipeline.json").write_text(json.dumps(pipeline, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(pipeline, ensure_ascii=False), flush=True)
    return bench_rc


if __name__ == "__main__":
    raise SystemExit(main())
