"""Current WindowBlocks-v3 body stage diagnostics on authenticated temporal input.

Complete-body and stage graphs run the selected FP16/FP8 arithmetic. Stage input
proofs come from the full functional trace and preserve strides and aliasing.
Isolated stage replays include public copies and completion; their medians must
not be summed as additive whole-frame costs. No model optimization is selected.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import traceback
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXACT = ROOT.parent / 'nr-b580'
R = EXACT / 'reference'
D = Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT = D / 'experimental/selected-body-stages-v1'
assert not OUT.exists()
TOOLCHAIN = D / 'toolchains/triton-xpu-3.8.0-git1e2d42a0'
os.environ['TRITON_CACHE_DIR'] = str(D / 'triton-cache-c32-triton38-v1')
sys.path[:0] = [str(TOOLCHAIN / 'site'), str(R), str(ROOT / 'backend')]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate = authenticate_main()
pins = {
    TOOLCHAIN / 'provision-v1.json': 'e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07',
    D / 'experimental/current-body-stages-v2/validation.json': '144d6f4aa8a28f37a4870af671821a3a30ab22b23614eea23f442c1e3aa1c448',
    D / 'results/esimd-vit-native-parity-v2/validation.json': '42dff3710e3db9b5f0312c63bd234acfe49c6aeafea1a2690040cb00db9092f3',
}
sources = {}
reports = {}
for p, h in pins.items():
    assert sha(p) == h
    reports[p] = js(p)
    sources[str(p)] = h
    for file, digest in reports[p].get('sources', {}).items():
        assert sha(file) == digest
        sources[file] = digest
provision = reports[TOOLCHAIN / 'provision-v1.json']
assert all(sha(TOOLCHAIN / 'site' / p) == h for p, h in provision['files'].items())
fixture_path = D / 'experimental/current-body-stages-v2/validation.json'
fixture = reports[fixture_path]
assert fixture['passed'] and js(fixture_path.parent.with_suffix('.log.lease.json'))['returncode'] == 0
for p in (Path(__file__), HERE / 'Run-SelectedBodyStagesV1.cmd',
          HERE / 'selected_body_stage_graphs_v1.py', HERE / 'nr256_selected_stack_v2.py'):
    sources[str(p)] = sha(p)
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']

import torch
import triton
from nr256_selected_stack_v2 import Stack
from nr_backend.execution import use_arithmetic_backend
from full_body_dataflow_v1 import DetailedDataflow
from selected_body_stage_graphs_v1 import Recorder, forward_staged, own_inputs, run_seeded, signature
import capture_body_v1 as body
import compressed_arrays_v1 as arrays

assert Path(triton.__file__).is_relative_to(TOOLCHAIN / 'site')
assert triton.__version__.startswith('3.8.0')
OUT.mkdir()
report = dict(scope=__doc__, sources=sources, exact_gate=gate, passed=False,
    complete_migration=False, candidate_promoted=False, selected_runtime='fee6d0d WindowBlocks v3',
    torch_version=torch.__version__, triton_version=triton.__version__, stages=[],
    input_scope='Frozen temporal frame 1 (source 181), actual residual NR256 body inputs',
    timing_scope='Isolated static stage/complete-body graphs, host replay through completion; '
                 'excludes dynamic front, warp, model validation/history, scaling, flow, upload, display and JIT.',
    full_body_samples_seconds=[], orders=[], expected_output=fixture['expected_output'],
    body_inputs=fixture['body_inputs'])
save = lambda: (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
values = lambda result: result if isinstance(result, tuple) else (result,)
raw = lambda t: t.cpu().numpy().tobytes()
digest = lambda t: hashlib.sha256(raw(t)).hexdigest()
stack = None
graphs = []

try:
    torch.set_num_threads(2)
    stack = Stack(EXACT)
    model = stack.model
    static = {name: None if meta is None else torch.from_numpy(arrays.load(meta)).to('xpu')
              for name, meta in fixture['body_inputs'].items()}
    expected = arrays.load(fixture['expected_output']).tobytes()
    options = dict(sigmoid=model.sigmoid, blend_scale=model.blend_scale, return_float32=False)
    # Saved static descriptors normally omit these constants; do not duplicate kwargs.
    options = {k: v for k, v in options.items() if k not in static}
    before_history = model._previous
    before_seed = model.next_seed
    with torch.inference_mode(), stack.installed(), body.installed(), use_arithmetic_backend('triton'):
        full_trace = DetailedDataflow()
        with full_trace.installed():
            full = stack.rewrite.original(model, **static, **options)
        assert raw(full) == expected
        staged_trace = DetailedDataflow()
        recorder = Recorder(staged_trace)
        with staged_trace.installed():
            staged = forward_staged(model, **static, stage=recorder, **options)
        assert raw(staged) == expected
        assert signature(full_trace.events) == signature(staged_trace.events)
        for key in ('triton_calls', 'standalone_fp8', 'quantization_calls', 'elided_fp8'):
            assert full_trace.rewrite_summary()[key] == staged_trace.rewrite_summary()[key]
        assert full_trace.rewrite_summary()['triton_calls'] == 683
        assert full_trace.rewrite_summary()['elided_fp8'] == 231
        assert len(recorder.stages) == 13
        report['full_trace_summary'] = full_trace.rewrite_summary()
        report['full_staged_physical_sequence_equal'] = True
        report['full_staged_output_byte_equal'] = True
        # The event stream is metadata only; saved tensors are content-addressed.
        trace_path = OUT / 'physical-sequence.json'
        trace_path.write_text(json.dumps(signature(full_trace.events), indent=2) + '\n', encoding='utf-8')
        report['physical_sequence'] = dict(path=str(trace_path), sha256=sha(trace_path))
        stream = torch.xpu.Stream()
        pool = torch.xpu.graph_pool_handle()

        def capture(fn):
            with stream:
                for _ in range(2):
                    warm = values(fn())
            torch.xpu.synchronize()
            public = tuple(torch.empty_like(t, memory_format=torch.contiguous_format) for t in warm)
            del warm
            graph = torch.xpu.XPUGraph()
            graphs.append(graph)
            with torch.xpu.graph(graph, stream=stream, pool=pool):
                temporary = values(fn())
                for target, source in zip(public, temporary):
                    target.copy_(source)
            del temporary
            graph.replay()
            torch.xpu.synchronize()
            return graph, public

        full_graph, full_public = capture(lambda: body.forward_front(model, **static, **options))
        assert raw(full_public[0]) == expected
        items = []
        for item in recorder.stages:
            args, roots = own_inputs(item)
            row = dict(name=item['name'], input_provenance=item['input_provenance'],
                inputs=[None if t is None else arrays.save(t.cpu().numpy()) for t in args],
                outputs=[arrays.save(t.cpu().numpy()) for t in item['outputs']],
                quantization_calls=item['quantization_calls'], elided_fp8=item['elided_fp8'],
                triton_calls=len(signature(item['events'])), samples_seconds=[])
            root_hashes = [digest(r['tensor']) for r in roots]
            fn = lambda it=item, ar=args, rt=roots: run_seeded(it, ar, rt)[0]
            graph, public = capture(fn)
            expected_stage = [raw(t) for t in item['outputs']]
            assert [raw(t) for t in public] == expected_stage, item['name']
            row['physical_sequence_and_fp8_elisions_equal_full_stage'] = True
            row['captured_output_byte_equal'] = True
            items.append(dict(row=row, graph=graph, public=public, args=args, roots=roots,
                              root_hashes=root_hashes, expected=expected_stage))
            report['stages'].append(row)
            print('Captured selected stage ' + item['name'], flush=True)
            save()

        persistent = [*full_public, *[t for t in static.values() if t is not None]]
        for item in items:
            persistent += [*item['public'], *[r['tensor'] for r in item['roots']]]
        segments = torch.xpu.memory_snapshot(pool)
        assert all(not any(s['address'] <= t.data_ptr() < s['address'] + s['total_size']
                           for s in segments) for t in persistent)
        report['persistent_io_outside_shared_pool'] = True
        # Interleave the intact body with the stages, rotating every round.
        workloads = [dict(name='full_body', graph=full_graph, public=full_public,
                         expected=[expected], samples=report['full_body_samples_seconds'])]
        workloads += [dict(name=i['row']['name'], graph=i['graph'], public=i['public'],
                           expected=i['expected'], samples=i['row']['samples_seconds']) for i in items]
        for repetition in range(7):
            offset = (repetition * 3) % len(workloads)
            order = workloads[offset:] + workloads[:offset]
            report['orders'].append([w['name'] for w in order])
            for workload in order:
                torch.xpu.synchronize()
                started = time.perf_counter()
                for _ in range(10):
                    workload['graph'].replay()
                torch.xpu.synchronize()
                workload['samples'].append((time.perf_counter() - started) / 10)
                assert [raw(t) for t in workload['public']] == workload['expected'], workload['name']
            print('Measured current selected stage round ' + str(repetition), flush=True)
        for item in items:
            item['row']['median_seconds'] = statistics.median(item['row']['samples_seconds'])
            assert [digest(r['tensor']) for r in item['roots']] == item['root_hashes']
        report['full_body_median_seconds'] = statistics.median(report['full_body_samples_seconds'])
        report['stage_inputs_unchanged'] = True
        report['diagnostic_medians_are_not_additive'] = True
        assert model._previous is before_history and model.next_seed == before_seed
        stack.graph._validate()
        assert all(None if t is None else raw(t) == arrays.load(fixture['body_inputs'][name]).tobytes()
                   for name, t in static.items() if t is not None)
        report['history_seed_model_constants_and_static_inputs_unchanged'] = True
        report['passed'] = True
except Exception:
    report['error'] = traceback.format_exc()
    raise
finally:
    for graph in graphs:
        graph.reset()
    if stack is not None:
        stack.close()
    assert all(sha(p) == h for p, h in sources.items())
    save()
print(json.dumps(dict(passed=report['passed'], full_body_ms=report.get('full_body_median_seconds', 0) * 1000,
    stages_ms={r['name']: r.get('median_seconds', 0) * 1000 for r in report['stages']}), indent=2), flush=True)
