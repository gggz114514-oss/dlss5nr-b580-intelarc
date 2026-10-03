"""Exhaustive half-domain FP8 proof and paired current whole-body screen.

Only the FP8 conversion implementation changes, including inlined callers.
Whole-body graph timings are static diagnostics, not complete NR/frame latency.
No source files, numeric model, installed toolchain or selected runtime change.
"""
from contextlib import contextmanager, nullcontext
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
OUT = D / 'experimental/short-fp8-body-v1'
assert not OUT.exists()
TOOLCHAIN = D / 'toolchains/triton-xpu-3.8.0-git1e2d42a0'
os.environ['TRITON_CACHE_DIR'] = str(D / 'triton-cache-c32-triton38-v1')
sys.path[:0] = [str(TOOLCHAIN / 'site'), str(R), str(ROOT / 'backend')]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate = authenticate_main()
ap = D / 'experimental/selected-body-stages-checkpoint-v1/saved-audit-v1.json'
assert sha(ap) == '159eed47ca0ecb4bec7c84c84f8c70cbcb5e31af060fa8e4446a383e36d080b3'
audit = js(ap)
assert audit['passed']
sources = dict(audit['sources'])
sources[str(ap)] = sha(ap)
for p in (Path(__file__), HERE / 'Run-ShortFp8BodyV1.cmd', HERE / 'short_fp8_v1.py', HERE / 'fork_fp8_jit_v1.py'):
    sources[str(p)] = sha(p)
assert all(sha(p) == h for p, h in sources.items())
provision = TOOLCHAIN / 'provision-v1.json'
assert sha(provision) == 'e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN / 'site' / p) == h for p, h in js(provision)['files'].items())
fixture_path = D / 'experimental/selected-body-stages-v1/validation.json'
assert sha(fixture_path) == 'f2553253c77b49e535c50f176afec1246abf5ee44784643bde1d25a820661e66'
fixture = js(fixture_path)
sources[str(fixture_path)] = sha(fixture_path)
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']

import numpy as np
import torch
import triton
from triton.compiler.compiler import CompiledKernel
from nr256_selected_stack_v2 import Stack
from nr_backend.execution import use_arithmetic_backend
from short_fp8_v1 import compare_kernel
from fork_fp8_jit_v1 import Fork
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
import immutable_artifacts_v1 as artifacts
assert Path(triton.__file__).is_relative_to(TOOLCHAIN / 'site') and triton.__version__.startswith('3.8.0')
OUT.mkdir()
report = dict(scope=__doc__, sources=sources, exact_gate=gate, passed=False,
    complete_migration=False, candidate_promoted=False, resources={}, builds={},
    torch_version=torch.__version__, triton_version=triton.__version__,
    body_inputs=fixture['body_inputs'], expected_output=fixture['expected_output'],
    samples_seconds={'selected': [], 'short_fp8': []}, orders=[], changed_input_checks=[])
save = lambda: (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
raw = lambda t: t.cpu().numpy().tobytes()
graphs = []
stack, fork = None, None


@contextmanager
def resource_gate():
    original = CompiledKernel.launch_metadata
    candidates = set(fork.memo.values())
    def checked(kernel, grid, stream, *args):
        if kernel.src.fn in candidates:
            kernel._init_handles()
            name = f'{kernel.src.fn.fn.__module__}.{kernel.src.fn.fn.__name__}'
            record = dict(name=name, spills=kernel.n_spills, registers=kernel.n_regs,
                          shared_bytes=kernel.metadata.shared)
            report['resources'][kernel.hash] = record
            if not isinstance(kernel.n_spills, int) or kernel.n_spills != 0:
                save()
                raise RuntimeError('Candidate register spill rejected before dispatch: ' + str(record))
        return original(kernel, grid, stream, *args)
    CompiledKernel.launch_metadata = checked
    try:
        yield
    finally:
        assert CompiledKernel.launch_metadata is checked
        CompiledKernel.launch_metadata = original


try:
    torch.set_num_threads(2)
    bits = np.arange(65536, dtype=np.uint16)
    x = torch.from_numpy(bits.view(np.float16)).to('xpu')
    old, new = torch.empty_like(x), torch.empty_like(x)
    compiled = compare_kernel.warmup(x, old, new, 65536, 256, grid=(256,), num_warps=4, enable_fp_fusion=False)
    compiled._init_handles()
    assert compiled.n_spills == 0
    compare_kernel[(256,)](x, old, new, 65536, 256, num_warps=4, enable_fp_fusion=False)
    a, b = old.cpu().numpy(), new.cpu().numpy()
    equal = a.tobytes() == b.tobytes()
    report['exhaustive'] = dict(encodings=65536, byte_equal=equal,
        different_words=int(np.count_nonzero(a.view('u2') != b.view('u2'))),
        input=arrays.save(bits.view(np.float16)), old=arrays.save(a), new=arrays.save(b),
        spills=compiled.n_spills, ir={k: artifacts.text(v, k) for k, v in compiled.asm.items()
                                     if k in ('ttgir', 'llir') and isinstance(v, str)})
    save()
    assert equal
    print('All 65536 half encodings byte-equal', flush=True)
    stack = Stack(EXACT)
    static = {name: None if meta is None else torch.from_numpy(arrays.load(meta)).to('xpu')
              for name, meta in fixture['body_inputs'].items()}
    expected = arrays.load(fixture['expected_output']).tobytes()
    options = dict(sigmoid=stack.model.sigmoid, blend_scale=stack.model.blend_scale, return_float32=False)
    with torch.inference_mode(), stack.installed(), body.installed(), use_arithmetic_backend('triton'):
        # Load all lazy modules by executing the selected body before forking.
        assert raw(body.forward_front(stack.model, **static, **options)) == expected
        fork = Fork(ROOT)
        sources.update(fork.sources)
        report['forks'] = fork.forks
        report['binding_count'] = len(fork.bindings)
        report['private_jit_globals_and_caches'] = True
        pool, stream = torch.xpu.graph_pool_handle(), torch.xpu.Stream()
        entries = {}
        for name in ('selected', 'short_fp8'):
            scope = nullcontext() if name == 'selected' else fork.installed()
            guard = nullcontext() if name == 'selected' else resource_gate()
            before = len(stack.rewrite.builds)
            with scope, guard:
                def compute():
                    return body.forward_front(stack.model, **static, **options)
                with stream:
                    for _ in range(2):
                        warm = compute()
                torch.xpu.synchronize()
                owned = torch.empty_like(warm, memory_format=torch.contiguous_format)
                del warm
                graph = torch.xpu.XPUGraph()
                graphs.append(graph)
                with torch.xpu.graph(graph, stream=stream, pool=pool):
                    temporary = compute()
                    owned.copy_(temporary)
                del temporary
            graph.replay()
            torch.xpu.synchronize()
            assert raw(owned) == expected, name
            report['builds'][name] = stack.rewrite.builds[before:]
            for build in report['builds'][name]:
                assert (build['triton_calls'], build['standalone_fp8'], build['quantization_calls'], build['elided_fp8']) == (683, 196, 427, 231)
            entries[name] = (graph, owned)
            print('Captured whole body ' + name, flush=True)
            save()
        assert report['resources'] and all(r['spills'] == 0 for r in report['resources'].values())
        segments = torch.xpu.memory_snapshot(pool)
        persistent = [*[v[1] for v in entries.values()], *[v for v in static.values() if v is not None]]
        assert not any(s['address'] <= t.data_ptr() < s['address'] + s['total_size'] for s in segments for t in persistent)
        report['persistent_io_outside_pool'] = True
        # A different front must execute the captured candidate, not a stale output.
        saved_front = static['front'].clone()
        static['front'].zero_()
        for name, (graph, owned) in entries.items():
            graph.replay()
            torch.xpu.synchronize()
            output = raw(owned)
            assert output != expected
            report['changed_input_checks'].append(dict(name=name, output=arrays.save(owned.cpu().numpy())))
        assert arrays.load(report['changed_input_checks'][0]['output']).tobytes() == arrays.load(report['changed_input_checks'][1]['output']).tobytes()
        static['front'].copy_(saved_front)
        for repetition in range(7):
            order = ['selected', 'short_fp8'] if repetition % 2 == 0 else ['short_fp8', 'selected']
            report['orders'].append(order)
            for name in order:
                graph, owned = entries[name]
                torch.xpu.synchronize()
                started = time.perf_counter()
                for _ in range(20):
                    graph.replay()
                torch.xpu.synchronize()
                report['samples_seconds'][name].append((time.perf_counter() - started) / 20)
                assert raw(owned) == expected
            print('Paired body round ' + str(repetition), flush=True)
        report['median_seconds'] = {n: statistics.median(v) for n, v in report['samples_seconds'].items()}
        report['change_percent'] = (report['median_seconds']['short_fp8'] / report['median_seconds']['selected'] - 1) * 100
        report['all_outputs_byte_equal'] = True
        assert all(raw(t) == arrays.load(fixture['body_inputs'][n]).tobytes() for n, t in static.items() if t is not None)
        assert stack.model._previous is None and stack.model.next_seed == 0
        stack.graph._validate()
        report['inputs_model_history_seed_unchanged'] = True
    fork.verify_restored()
    report['bindings_restored'] = True
    report['passed'] = True
except Exception:
    report['error'] = traceback.format_exc()
    raise
finally:
    for graph in graphs:
        graph.reset()
    if stack is not None:
        stack.close()
    if fork is not None:
        fork.verify_restored()
    assert all(sha(p) == h for p, h in sources.items())
    save()
print(json.dumps(dict(passed=report['passed'], median_ms={n: v * 1000 for n, v in report.get('median_seconds', {}).items()},
                     change_percent=report.get('change_percent')), indent=2), flush=True)
