"""Paired timings with candidate diagnostics confined to warmup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

BASE = Path(__file__).resolve().parents[2]
DATA = Path('D:/Codex-NR-Experiments/nr-b580')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ['TRITON_CACHE_DIR'] = str(DATA / 'reference/triton-cache-c32-triton38-v1')
    sys.path.insert(0, str(BASE / 'nr-b580-int8/product/comfy'))
    from runtime_environment import isolate
    handles = isolate()
    sys.path[:0] = [str(DATA / 'reference/toolchains/triton-xpu-3.8.0-git1e2d42a0/site'),
                   str(BASE / 'nr-b580/backend'), str(BASE / 'nr-b580-int8/product')]
    import numpy as np
    import torch
    from PIL import Image
    from nr_exact_controls_candidate_v1 import Session as Baseline
    from nr_exact_shortfp8_candidate_v1 import Session as Candidate
    reference = BASE / 'nr-b580/reference'
    meta = json.loads((reference / '4060-depth-zero-fine-sequence-v1.json').read_text(encoding='utf-8-sig'))
    native = DATA / 'reference/results' / Path(meta['runDirectory']).name / 'output'
    frames = []
    for i in (0, 1):
        rgb = np.asarray(Image.open(reference / f'inputs/temporal-v1/frame{i:02d}.png').convert('RGB'), dtype='f4') / 255
        motion = np.fromfile(native / f'frame{i:02d}.png_motion.rg32f.bin', '<f4').reshape(256, 256, 2).copy()
        frames.append((torch.from_numpy(rgb).to('xpu'), torch.from_numpy(motion).to('xpu')))
    sessions = [Baseline.create(BASE / 'nr-b580'), Candidate.create(BASE / 'nr-b580')]
    from triton.compiler.compiler import CompiledKernel
    original_metadata = CompiledKernel.launch_metadata
    source_paths = [Path(__file__), BASE / "nr-b580-int8/product/nr_exact_shortfp8_candidate_v1.py"] + list((BASE / "nr-b580-int8/product/exact_shortfp8_v1").glob("*.py"))
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    rows = []
    try:
        for iteration in range(8):
            sessions[1].diagnostics = iteration < 2
            before_calls = dict(Candidate.calls)
            before_resources = dict(Candidate.resources)
            times, outputs, histories = {}, {}, {}
            for index in ([0, 1] if iteration % 2 == 0 else [1, 0]):
                outputs[index], histories[index] = [], []
                total = 0
                for frame, (rgb, motion) in enumerate(frames):
                    torch.xpu.synchronize()
                    start = time.perf_counter()
                    value = sessions[index].process(rgb, motion, reset=frame == 0)
                    total += time.perf_counter() - start
                    outputs[index].append(value.color.cpu().numpy().tobytes())
                    histories[index].append(sessions[index]._model._previous.cpu().numpy().tobytes())
                times[index] = total * 1000 / 2
            assert outputs[0] == outputs[1] and histories[0] == histories[1]
            assert CompiledKernel.launch_metadata is original_metadata
            if iteration >= 2:
                assert Candidate.calls == before_calls and Candidate.resources == before_resources
            row = dict(diagnostics_enabled=iteration < 2, iteration=iteration, warmup=iteration < 2, baseline_ms=times[0], candidate_ms=times[1])
            rows.append(row)
            print(json.dumps(row), flush=True)
    finally:
        for session in sessions:
            session.close()
    baseline = statistics.median(row['baseline_ms'] for row in rows[2:])
    candidate = statistics.median(row['candidate_ms'] for row in rows[2:])
    assert sources == {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    assert Candidate.resources and all(r["spills"] == 0 for r in Candidate.resources.values())
    report = dict(passed=True, sources=sources, diagnostics_only_during_warmup=True, measured_diagnostic_records_unchanged=True, warmup_resources=Candidate.resources, scope='256 SDR default controls; paired reset+temporal Session latency; excludes load and readback',
                  rows=rows, baseline_median_ms=baseline, candidate_median_ms=candidate,
                  reduction_percent=(1 - candidate / baseline) * 100)
    (args.out / 'benchmark.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
