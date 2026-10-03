"""Exhaustive current cubic compatibility, real C32 primitives and whole-body bytes."""
import hashlib
import json
import os
import statistics
import sys
import time
import traceback
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXACT = ROOT.parent / 'nr-b580'
R = EXACT / 'reference'
D = Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT = D / 'experimental/c32-mlp-lut-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR'] = str(D / 'triton-cache-sm89-v1')
sys.path[:0] = [str(R), str(ROOT / 'backend')]
from attention_weights_main_gate_v1 import authenticate_main
gate = authenticate_main()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
capture_path = D / 'experimental/c32-mlp-operands-v1/validation.json'
body_path = D / 'experimental/fused-body-stages-v2/validation.json'
stack_path = D / 'results/c32-chunk-residual256-v1/validation.json'
for p, digest in [(capture_path, '7b865eee932721ca1066b38973b2a2ddae215d4dbba9bd298edeebe8dece3b57'),
                  (body_path, '85462097455e379845eb3c1244f22fa91329dd5ddaaf708d438424dd58f3efac'),
                  (stack_path, 'd9dfa0ba985b5577958e619884b2b278faa54ef0c76b827ff124c30407df290e')]:
    assert sha(p) == digest and js(p)['passed']
    assert js(p.parent.with_suffix('.log.lease.json'))['returncode'] == 0
    assert all(sha(s) == h for s, h in js(p)['sources'].items())
capture, body_record, stack = js(capture_path), js(body_path), js(stack_path)
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
import torch
import triton
import triton.language as tl
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from cubic_lut_constant_v1 import register, Constant, BUFFER, TABLE
from fused_c32_mlp_lut_v1 import forward, lookup, FusedC32
from fused_c32_mlp_v1 import forward as previous
from fused_branched_mlp_v1 import _cubic
from strided_batched_v2 import StridedMatrices
from fused_branched_pairs_v2 import FusedPairs
from fused_split_ffwd_v2 import FusedSplit
from fused_vit_projection_v3 import FusedVitProjection
from fused_swin_scheduling_v1 import FusedSwin
from c32_chunk_layout_v2 import ChunkedHeadLayout
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
import immutable_artifacts_v1 as artifacts

paths = [Path(__file__), HERE / 'Run-C32MLPLutV1.cmd', HERE / 'cubic_lut_constant_v1.py',
         HERE / 'fused_c32_mlp_lut_v1.py', TABLE, D / 'cubic-fp8-fused-cpu-v3.json',
         D / 'cubic-fp8-fused-xpu-v3.json', capture_path, body_path, stack_path,
         *[Path(p) for record in (capture, body_record, stack) for p in record['sources']]]
frozen = {str(p): sha(p) for p in paths}
OUT.mkdir()
report = dict(scope=__doc__, sources=frozen, exact_gate=gate, passed=False, cases=[], body_checks=[],
              complete_migration=False, timing='Five rotated rounds of 10 static graph calls, with matching full output copies for every variant.')
graphs = []


def save():
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def raw(t):
    return t.cpu().numpy().tobytes()


@triton.jit
def scalar_pair(X, LUT, A, B, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, i < N, other=0)
    tl.store(A + i, _cubic(x), i < N)
    tl.store(B + i, lookup(x, LUT), i < N)


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


try:
    torch.set_num_threads(2)
    model = MotionNR.from_assets(EXACT / 'model-assets/sf-v2/WEIGHTS_HT.bin',
                                 EXACT / 'model-assets/noise-sm89-v2',
                                 EXACT / 'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    lut = register(model)
    guard = Constant(model)
    assert guard.require() is lut
    # Exercise identity/version rejection on an isolated constant, not this LUT.
    dummy = torch.nn.Module()
    dummy.register_buffer(BUFFER, lut.clone())
    test = Constant(dummy)
    with torch.no_grad():
        getattr(dummy, BUFFER)[0] += 1
    try:
        test.require()
    except RuntimeError:
        pass
    else:
        raise AssertionError('Changed LUT accepted')
    test = Constant(dummy)
    setattr(dummy, BUFFER, getattr(dummy, BUFFER).clone())
    try:
        test.require()
    except RuntimeError:
        pass
    else:
        raise AssertionError('Replaced LUT accepted')
    del dummy, test
    report['constant_identity_and_version_guards'] = True
    provider = StridedMatrices()
    provider.select('fp16_xmx')
    configs = [dict(bm=bm, warps=4, stages=stages) for stages in (1, 2) for bm in (16, 32)]
    with torch.inference_mode(), provider.installed(), use_arithmetic_backend('triton'):
        bits = np.arange(65536, dtype='u2')
        x = torch.from_numpy(bits.view('f2').copy()).to('xpu')
        direct, looked = torch.empty_like(x), torch.empty_like(x)
        scalar_pair[(triton.cdiv(65536, 512),)](x, lut, direct, looked, 65536, 512, enable_fp_fusion=False)
        expected_table = np.load(TABLE, allow_pickle=False)
        assert raw(direct) == raw(looked) == expected_table.tobytes()
        report['all_half_encodings'] = dict(count=65536, byte_equal_current_helper=True,
                                           output=arrays.save(direct.cpu().numpy()))
        save()
        print('All 65536 current cubic half encodings match the existing table', flush=True)
        cases = []
        for key, row in capture['cases'].items():
            assert row['captured']
            saved = {n: arrays.load(meta) for n, meta in row['arrays'].items()}
            assert all(sha(meta['path']) == meta['sha256'] for meta in row['arrays'].values())
            cases.append((key, saved, False))
        first = cases[0][1]
        for name, count in (('tail19', 19), ('repeated65539', 65539)):
            saved = dict(first)
            for field in ('features', 'output'):
                a = first[field].reshape(-1, 32)
                saved[field] = np.tile(a, ((count + len(a) - 1) // len(a), 1))[:count].copy()
            cases.append((name, saved, True))
        for name, saved, synthetic in cases:
            weights = {n: torch.from_numpy(saved[n]).to('xpu') for n in ('expansion', 'contraction', 'skip_scale')}
            features = torch.from_numpy(saved['features']).to('xpu')
            expected = saved['output'].tobytes()
            assert raw(previous(features, **weights, bm=32, warps=4, stages=1)[0]) == expected
            row = dict(name=name, synthetic=synthetic, shape=list(features.shape),
                       arrays={n: arrays.save(a) for n, a in saved.items()}, candidates=[])
            report['cases'].append(row)
            functions = [lambda: previous(features, **weights, bm=32, warps=4, stages=1)[0]]
            for config in configs:
                value, kernel = forward(features, **weights, lut=lut, **config)
                equal = raw(value) == expected
                row['candidates'].append(dict(config=config, byte_equal=equal,
                    spills=kernel.n_spills, registers=getattr(kernel, 'n_regs', None),
                    ttgir=artifacts.text(str(kernel.asm['ttgir']), 'ttgir')))
                assert equal and 'ttig.dpas' in str(kernel.asm['ttgir']), (name, config)
                functions.append(lambda config=config: forward(features, **weights, lut=lut, **config)[0])
            entries = [graph_for(fn) for fn in functions]
            graphs.extend(e[0] for e in entries)
            samples = [[] for _ in entries]
            for repetition in range(5):
                order = list(range(len(entries)))
                order = order[repetition:] + order[:repetition]
                for candidate in order:
                    graph, value, _ = entries[candidate]
                    torch.xpu.synchronize()
                    started = time.perf_counter()
                    for _ in range(10):
                        graph.replay()
                    torch.xpu.synchronize()
                    samples[candidate].append((time.perf_counter() - started) / 10)
                    assert raw(value) == expected
            assert raw(features) == saved['features'].tobytes()
            assert all(raw(v) == saved[n].tobytes() for n, v in weights.items())
            row.update(samples_seconds=samples, median_seconds=[statistics.median(s) for s in samples], operands_unchanged=True)
            for graph, _, _ in entries:
                graph.reset()
                graphs.remove(graph)
            del entries
            torch.xpu.empty_cache()
            save()
            print(json.dumps(dict(case=name, median_ms=[v * 1000 for v in row['median_seconds']])), flush=True)
        pairs, split, vit = FusedPairs(model, provider), FusedSplit(model, provider), FusedVitProjection(model, provider)
        scheduler, layout = FusedSwin(provider), ChunkedHeadLayout(model, provider)
        inputs = {n: torch.from_numpy(arrays.load(meta)).to('xpu') for n, meta in body_record['body_inputs'].items()}
        options = dict(sigmoid=model.sigmoid, blend_scale=model.blend_scale, return_float32=False)
        with pairs.installed(), split.installed(), vit.installed(), scheduler.installed(), layout.installed(), body.installed():
            for config in configs:
                c32 = FusedC32(model, provider, **config)
                with c32.installed(), use_arithmetic_backend('triton') as dispatch:
                    value = body.forward_front(model, **inputs, **options)
                equal = raw(value) == arrays.load(body_record['expected_output']).tobytes()
                report['body_checks'].append(dict(config=config, byte_equal=equal, calls=c32.calls, dispatch=dict(dispatch)))
                assert equal and sum(c32.calls.values()) == 10
    assert guard.require() is lut and raw(lut) == np.load(TABLE, allow_pickle=False).view('i2').tobytes()
    report.update(passed=True, lut_bytes=lut.numel() * lut.element_size(), lut_unchanged=True)
except BaseException as exc:
    report.update(error=repr(exc), traceback=traceback.format_exc())
    raise
finally:
    for graph in graphs:
        graph.reset()
    assert all(sha(p) == h for p, h in frozen.items())
    save()
print('C32 embedded LUT primitive and body checks passed', flush=True)
