"""Batch branch work on all real NR256 MLPs and existing full480 operands.

Complete output comparisons precede timings. Reduced and full workloads use
their respective fastest previous branch implementations. Actual reduced
operands drive reduced tuning; full480 operands drive full tuning separately.
Primitive timings include the same persistent graph-output copy on every path.
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
OUT = D / 'experimental/batched-branches-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR'] = str(D / 'triton-cache-sm89-v1')
sys.path[:0] = [str(R), str(ROOT / 'backend')]
from attention_weights_main_gate_v1 import authenticate_main
gate = authenticate_main()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
records, sources = {}, {}
for key, folder, digest in [
    ('small', 'experimental/small-branched-mlp-operands-v1', '0c3ee88b397831266e6359e5eb24db5d77f3279f5bd9c67f9bda8ae11df4c11b'),
    ('wide', 'experimental/wide-mlp-lut-v1', '3e71f13778ea5e9348c53c950b689e2cc08c5398e0ad4c56414682f7ab0d2679'),
    ('full', 'results/wide-lut-full480-v1', 'cd96f20a70c54c7751d75b13a6e37dfa76932d3ab8fbce33b1767badf8fed335')]:
    p = D / folder / 'validation.json'
    assert sha(p) == digest
    r = js(p)
    assert r['passed'] and js(p.parent.with_suffix('.log.lease.json'))['returncode'] == 0
    assert all(sha(p) == h for p, h in r['sources'].items())
    records[key] = r
    sources.update(r['sources'])
    sources[str(p)] = digest
for p in (Path(__file__), HERE / 'Run-BatchedBranchesV1.cmd', HERE / 'batched_branched_mlp_v1.py'):
    sources[str(p)] = sha(p)
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
from fused_branched_pairs_lut_v1 import forward as old_lookup_pair
from fused_split_ffwd_v2 import FusedSplit
from fused_c32_mlp_lut_v1 import FusedC32
from strided_batched_v2 import StridedMatrices
from fused_vit_projection_v3 import FusedVitProjection
from fused_swin_scheduling_v1 import FusedSwin
from c32_chunk_layout_v2 import ChunkedHeadLayout
from batched_branched_mlp_v1 import forward, FusedBatched
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
import immutable_artifacts_v1 as artifacts

OUT.mkdir()
report = dict(scope=__doc__, sources=sources, exact_gate=gate, passed=False,
              cases=[], body_checks=[], complete_migration=False)
graphs = []
configs = [dict(pair_bm=bm, pair_stages=stage, project_bm=16, project_bn=bn)
           for bm, stage, bn in ((16,1,32), (16,2,32), (16,1,64), (16,2,64), (32,1,64), (32,2,64))]
full_pair_config = {int(c): v for c, v in records['wide']['selected_primitive_configs']['pairs'].items()}


def save():
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


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


def check_case(scope, name, metadata, synthetic, timed):
    saved = {n: arrays.load(m) for n, m in metadata.items()}
    features = torch.from_numpy(saved['features']).to('xpu')
    weights = {n: torch.from_numpy(saved[n]).to('xpu') for n in ('expand', 'reduce', 'project', 'skip_scale')}
    c, m = features.shape[-1], features.numel()//features.shape[-1]
    expected = saved['output'].tobytes()
    module = SimpleNamespace(channels=c, **weights)
    if m < 1024:
        baseline = lambda: BranchedMLP.forward_unquantized(module, features)
        kind = 'original_small'
    elif scope == 'small':
        baseline = lambda: old_pair(features, **weights, bm=16, warps=4, stages=2 if c == 256 else 1)[0]
        kind = 'previous_reduced_pair'
    else:
        baseline = lambda: old_lookup_pair(features, **weights, lut=lut, **full_pair_config[c])[0]
        kind = 'previous_full_lookup_pair'
    row = dict(scope=scope, name=name, rows=m, channels=c, shape=list(features.shape),
               synthetic=synthetic, timed=timed, arrays=metadata, baseline=kind, candidates=[])
    report['cases'].append(row)
    with use_arithmetic_backend('triton') as baseline_dispatch:
        assert raw(baseline()) == expected, (scope, name, 'baseline')
    row['baseline_dispatch'] = dict(baseline_dispatch)
    functions = [baseline]
    for config in configs:
        with use_arithmetic_backend('triton') as dispatch:
            value, kernels = forward(features, **weights, lut=lut, **config)
        equal = raw(value) == expected
        compiled = []
        for kernel in kernels:
            ir = str(kernel.asm['ttgir'])
            assert 'ttig.dpas' in ir
            compiled.append(dict(spills=kernel.n_spills, ttgir=artifacts.text(ir, 'ttgir')))
        row['candidates'].append(dict(config=config, byte_equal=equal, dispatch=dict(dispatch), kernels=compiled))
        assert equal and dict(dispatch) == dict(baseline_dispatch), (scope, name, config)
        functions.append(lambda config=config: forward(features, **weights, lut=lut, **config)[0])
    if timed:
        entries = [graph_for(fn) for fn in functions]
        graphs.extend(e[0] for e in entries)
        samples, orders = [[] for _ in entries], []
        for repetition in range(7):
            order = list(range(len(entries)))
            order = order[repetition:] + order[:repetition]
            orders.append(order)
            for i in order:
                graph, value, _ = entries[i]
                torch.xpu.synchronize()
                started = time.perf_counter()
                for _ in range(10):
                    graph.replay()
                torch.xpu.synchronize()
                samples[i].append((time.perf_counter()-started)/10)
                assert raw(value) == expected
        row.update(samples_seconds=samples, median_seconds=[statistics.median(s) for s in samples], orders=orders)
        for graph, _, _ in entries:
            graph.reset()
            graphs.remove(graph)
        print(json.dumps(dict(scope=scope, name=name, median_ms=[v*1000 for v in row['median_seconds']])), flush=True)
    assert raw(features) == saved['features'].tobytes()
    assert all(raw(value) == saved[n].tobytes() for n, value in weights.items())
    row['operands_unchanged'] = True
    save()


try:
    torch.set_num_threads(2)
    model = MotionNR.from_assets(EXACT / 'model-assets/sf-v2/WEIGHTS_HT.bin',
        EXACT / 'model-assets/noise-sm89-v2', EXACT / 'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    lut = register(model)
    constant = Constant(model)
    provider = StridedMatrices()
    provider.select('fp16_xmx')
    with torch.inference_mode(), provider.installed(), use_arithmetic_backend('triton'):
        timed_shapes = set()
        for source in records['small']['cases']:
            shape = (source['rows'], source['channels'])
            timed = shape not in timed_shapes
            timed_shapes.add(shape)
            check_case('small', source['name'], source['arrays'], False, timed)
        for source in records['wide']['pair_cases']:
            check_case('full', source['name'], source['arrays'], source['synthetic'], True)
        report['selected_configs'] = {}
        for scope in ('small', 'full'):
            selected = {}
            for c in (64, 128, 256):
                rows = [r for r in report['cases'] if r['scope'] == scope and r['channels'] == c and r['timed'] and not r['synthetic']]
                scores = [statistics.mean(r['median_seconds'][i+1] for r in rows) for i in range(len(configs))]
                selected[c] = configs[min(range(len(configs)), key=lambda i: scores[i])]
            report['selected_configs'][scope] = selected
        inputs = {n: torch.from_numpy(arrays.load(m)).to('xpu') for n, m in records['small']['body_inputs'].items()}
        target = arrays.load(records['small']['expected_output']).tobytes()
        split, c32 = FusedSplit(model, provider), FusedC32(model, provider)
        vit, scheduler = FusedVitProjection(model, provider), FusedSwin(provider)
        layout = ChunkedHeadLayout(model, provider)
        with split.installed(), c32.installed(), vit.installed(), scheduler.installed(), layout.installed(), body.installed():
            for scope, selected in report['selected_configs'].items():
                fusion = FusedBatched(model, provider, configurations=selected)
                with fusion.installed(), use_arithmetic_backend('triton') as dispatch:
                    value = body.forward_front(model, **inputs, sigmoid=model.sigmoid,
                        blend_scale=model.blend_scale, return_float32=False)
                equal = raw(value) == target
                report['body_checks'].append(dict(selection=scope, byte_equal=equal, calls=dict(fusion.calls), dispatch=dict(dispatch)))
                assert equal and sum(fusion.calls.values()) == 36
                assert dict(dispatch) == records['small']['dispatch']
                print('Complete body matched with '+scope+' selection', flush=True)
    assert constant.require() is lut and raw(lut) == np.load(TABLE, allow_pickle=False).view('i2').tobytes()
    report.update(passed=True, lut_unchanged=True, primitive_comparisons=len(report['cases'])*len(configs))
except BaseException as error:
    report.update(error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    for graph in graphs:
        graph.reset()
    assert all(sha(p) == h for p, h in sources.items())
    authenticate_main()
    save()
print(json.dumps(dict(passed=report['passed'], selection=report.get('selected_configs'))), flush=True)
