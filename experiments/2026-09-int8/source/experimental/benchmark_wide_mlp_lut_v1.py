"""Real wide-MLP operands: explicit cubic lookup, launch tuning and full-body bytes.

The baseline is the currently selected streaming FP16 implementation, including
the original small-row branch fallback. Four candidates retain every reduction
and rounding boundary. Timing includes an identical persistent output copy on
every graph. These are primitive timings, not entire-frame speed claims.
"""
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
import urllib.request
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXACT = ROOT.parent / 'nr-b580'
R = EXACT / 'reference'
D = Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT = D / 'experimental/wide-mlp-lut-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR'] = str(D / 'triton-cache-sm89-v1')
sys.path[:0] = [str(R), str(ROOT / 'backend')]
from attention_weights_main_gate_v1 import authenticate_main
gate = authenticate_main()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
records = {}
sources = {}
for key, relative, digest in [
    ('pairs', 'experimental/branched-mlp-operands-v2', 'b8ee4d2478b11a7fab585c4528a13262bc407617c246b99f2a892bbabe9c4722'),
    ('split', 'experimental/fused-split-ffwd-v1', '7bfd22bf8f23f0d8117ea82cf425f212c55f3a7a6a0c2d7bc2f44d8c5e0aa9fc'),
    ('body', 'experimental/fused-body-stages-v2', '85462097455e379845eb3c1244f22fa91329dd5ddaaf708d438424dd58f3efac'),
    ('c32', 'experimental/c32-mlp-lut-v1', 'cff24169c9c037ff243b33205a2f6ec2bc5c3c041e069af33c77c1d582400ea6'),
    ('stack', 'results/c32-lut-residual256-v2', '00156f5e5e01c2918f6fbca4c03449ffa5045feae45f111eec8e20e87e997782')]:
    path = D / relative / 'validation.json'
    assert sha(path) == digest
    record = js(path)
    assert record['passed'] and js(path.parent.with_suffix('.log.lease.json'))['returncode'] == 0
    assert all(sha(p) == h for p, h in record['sources'].items())
    records[key] = record
    sources.update(record['sources'])
    sources[str(path)] = digest
for name in ('benchmark_wide_mlp_lut_v1.py', 'Run-WideMLPLutV1.cmd',
             'fused_mlp_pair_lut_v1.py', 'fused_branched_pairs_lut_v1.py', 'fused_split_ffwd_lut_v1.py'):
    sources[str(HERE / name)] = sha(HERE / name)
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
import torch
from nr_backend.temporal import MotionNR
from nr_backend.multihead_block import BranchedMLP
from nr_backend.execution import use_arithmetic_backend
from cubic_lut_constant_v1 import register, Constant, TABLE
from fused_branched_pairs_v1 import forward as old_pair
from fused_branched_pairs_v2 import FusedPairs as OldPairs
from fused_branched_pairs_lut_v1 import forward as new_pair, FusedPairs
from fused_split_ffwd_v1 import forward as old_split
from fused_split_ffwd_v2 import FusedSplit as OldSplit
from fused_split_ffwd_lut_v1 import forward as new_split, FusedSplit
from fused_c32_mlp_lut_v1 import FusedC32
from strided_batched_v2 import StridedMatrices
from fused_vit_projection_v3 import FusedVitProjection
from fused_swin_scheduling_v1 import FusedSwin
from c32_chunk_layout_v2 import ChunkedHeadLayout
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
import immutable_artifacts_v1 as artifacts

OUT.mkdir()
report = dict(scope=__doc__, sources=sources, exact_gate=gate, passed=False,
              pair_cases=[], split_cases=[], body_checks=[], complete_migration=False)
graphs = []
pair_configs = [dict(bm=bm, warps=4, stages=stage) for stage in (1, 2) for bm in (16, 32)]
split_configs = [dict(bm=bm, bn=bn, stages=1) for bm, bn in ((16, 32), (16, 64), (32, 32), (32, 64))]


def save():
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def raw(t):
    return t.cpu().numpy().tobytes()


def graph_for(fn):
    stream = torch.xpu.Stream()
    torch.xpu.synchronize()
    with stream:
        for _ in range(2):
            warm = fn()
    torch.xpu.synchronize()
    out = torch.empty_like(warm)
    del warm
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph, stream=stream):
        temporary = fn()
        out.copy_(temporary)
    del temporary
    graph.replay()
    torch.xpu.synchronize()
    return graph, out, stream


def check_case(family, name, metadata, synthetic, timed):
    saved = {n: arrays.load(m) for n, m in metadata.items()}
    row = dict(name=name, arrays=metadata, synthetic=synthetic, timed=timed, candidates=[])
    report[family + '_cases'].append(row)
    target_name = 'output' if family == 'pair' else 'expected'
    expected = saved[target_name].tobytes()
    values = {n: torch.from_numpy(a).to('xpu') for n, a in saved.items() if n != target_name}
    if family == 'pair':
        features = values['features']
        c = features.shape[-1]
        m = features.numel() // c
        weights = {n: values[n] for n in ('expand', 'reduce', 'project', 'skip_scale')}
        module = SimpleNamespace(channels=c, **weights)
        baseline = (lambda: BranchedMLP.forward_unquantized(module, features)) if m < 1024 else (
            lambda: old_pair(features, **weights, bm=16, warps=4, stages=2 if c == 256 else 1)[0])
        row.update(channels=c, rows=m, baseline='original_small_fallback' if m < 1024 else 'current_streaming_pair')
        candidate = lambda config: new_pair(features, **weights, lut=lut, **config)
        configs = pair_configs
    else:
        z = values['z']
        m = z.numel() // 512
        weights = {n: values[n] for n in ('expand', 'reduce')}
        baseline = lambda: old_split(z, **weights, bm=16, bn=32 if m <= 512 else 64, stages=1)[0]
        row.update(channels=512, rows=m, baseline='current_streaming_group')
        candidate = lambda config: new_split(z, **weights, lut=lut, **config)
        configs = split_configs
    assert raw(baseline()) == expected, (family, name, 'baseline')
    functions = [baseline]
    for config in configs:
        value, kernel = candidate(config)
        equal = raw(value) == expected
        ir = str(kernel.asm['ttgir'])
        row['candidates'].append(dict(config=config, byte_equal=equal, spills=kernel.n_spills,
                                       ttgir=artifacts.text(ir, 'ttgir')))
        assert equal and 'ttig.dpas' in ir, (family, name, config)
        functions.append(lambda config=config: candidate(config)[0])
    if timed:
        entries = [graph_for(fn) for fn in functions]
        graphs.extend(e[0] for e in entries)
        samples = [[] for _ in entries]
        for repetition in range(5):
            order = list(range(len(entries)))
            order = order[repetition:] + order[:repetition]
            for i in order:
                graph, value, _ = entries[i]
                torch.xpu.synchronize()
                started = time.perf_counter()
                for _ in range(10):
                    graph.replay()
                torch.xpu.synchronize()
                samples[i].append((time.perf_counter() - started) / 10)
                assert raw(value) == expected
        row.update(samples_seconds=samples, median_seconds=[statistics.median(s) for s in samples])
        for graph, _, _ in entries:
            graph.reset()
            graphs.remove(graph)
        print(json.dumps(dict(family=family, name=name, median_ms=[t*1000 for t in row['median_seconds']])), flush=True)
    assert all(raw(v) == saved[n].tobytes() for n, v in values.items())
    row['operands_unchanged'] = True
    save()


try:
    torch.set_num_threads(2)
    model = MotionNR.from_assets(EXACT / 'model-assets/sf-v2/WEIGHTS_HT.bin',
                                 EXACT / 'model-assets/noise-sm89-v2',
                                 EXACT / 'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    lut = register(model)
    constant = Constant(model)
    provider = StridedMatrices()
    provider.select('fp16_xmx')
    with torch.inference_mode(), provider.installed(), use_arithmetic_backend('triton'):
        for name, source in records['pairs']['cases'].items():
            assert source['captured']
            check_case('pair', name, source['arrays'], False, True)
        for c in (64, 128, 256):
            source = next(r for r in records['pairs']['cases'].values() if r['channels'] == c)
            for count in (19, 577):
                metadata = dict(source['arrays'])
                for field in ('features', 'output'):
                    a = arrays.load(metadata[field]).reshape(-1, c)[:count].copy()
                    assert len(a) == count
                    metadata[field] = arrays.save(a)
                check_case('pair', f'tail_{count}_c{c}', metadata, True, True)
        for i, source in enumerate(records['split']['cases']):
            check_case('split', source['name'], source['arrays'], source['synthetic'],
                       i in (0, 7, 8, 15, 16, 17, 18))
        # Choose from recorded primitive measurements; full pipelines decide adoption.
        selected_pairs = {}
        for c in (64, 128, 256):
            rows = [r for r in report['pair_cases'] if r['channels'] == c and not r['synthetic']]
            scores = [statistics.mean(r['median_seconds'][i+1] for r in rows) for i in range(4)]
            selected_pairs[c] = pair_configs[min(range(4), key=lambda i: scores[i])]
        selected_split = {}
        for key, small in (('small', True), ('large', False)):
            rows = [r for r in report['split_cases'] if r['timed'] and (r['rows'] <= 512) == small]
            scores = [statistics.mean(r['median_seconds'][i+1] for r in rows) for i in range(4)]
            selected_split[key] = split_configs[min(range(4), key=lambda i: scores[i])]
        report['selected_primitive_configs'] = dict(pairs=selected_pairs, split=selected_split,
            limitation='Large split tuning uses labeled repeated operands, not a large captured video frame.')
        inputs = {n: torch.from_numpy(arrays.load(meta)).to('xpu') for n, meta in records['body']['body_inputs'].items()}
        target = arrays.load(records['body']['expected_output']).tobytes()
        c32, vit = FusedC32(model, provider), FusedVitProjection(model, provider)
        scheduler, layout = FusedSwin(provider), ChunkedHeadLayout(model, provider)
        options = dict(sigmoid=model.sigmoid, blend_scale=model.blend_scale, return_float32=False)
        with c32.installed(), vit.installed(), scheduler.installed(), layout.installed(), body.installed():
            for name, use_pair, use_split, min_rows in [('pairs_only', True, False, 1024),
                    ('split_only', False, True, 1024), ('both', True, True, 1024), ('both_all_rows', True, True, 0)]:
                pairs = FusedPairs(model, provider, launch_configs=selected_pairs, min_rows=min_rows) if use_pair else OldPairs(model, provider)
                split = FusedSplit(model, provider, configuration=selected_split['small']) if use_split else OldSplit(model, provider)
                with pairs.installed(), split.installed(), use_arithmetic_backend('triton') as dispatch:
                    value = body.forward_front(model, **inputs, **options)
                equal = raw(value) == target
                report['body_checks'].append(dict(name=name, byte_equal=equal, pair_calls=dict(pairs.calls),
                                                  split_calls=split.calls, dispatch=dict(dispatch)))
                assert equal and split.calls == 16, name
                print('Whole body bytes passed: ' + name, flush=True)
    assert constant.require() is lut and raw(lut) == np.load(TABLE, allow_pickle=False).view('i2').tobytes()
    report.update(passed=True, lut_unchanged=True, total_primitive_comparisons=148)
except BaseException as error:
    report.update(error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    for graph in graphs:
        graph.reset()
    assert all(sha(p) == h for p, h in sources.items())
    authenticate_main()
    save()
print(json.dumps(dict(passed=report['passed'], selection=report.get('selected_primitive_configs'))), flush=True)
