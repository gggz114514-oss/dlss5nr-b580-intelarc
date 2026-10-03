"""Paired complete static body: consume ViT head-major attention directly.

Checks all seven boundaries of all eight blocks before timing, then removes
only the intermediate head-to-token copy from the output projection chain.
Not a complete NR benchmark; C512 layout and post crop are separate candidates.
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
import urllib.error

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXACT = ROOT.parent / 'nr-b580'
R = EXACT / 'reference'
D = Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT = D / 'experimental/vit-head-layout-body-v1'
assert not OUT.exists()
TOOLCHAIN = D / 'toolchains/triton-xpu-3.8.0-git1e2d42a0'
os.environ['TRITON_CACHE_DIR'] = str(D / 'triton-cache-c32-triton38-v1')
sys.path[:0] = [str(TOOLCHAIN / 'site'), str(R), str(ROOT / 'backend')]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate = authenticate_main()
ap = D / 'experimental/short-fp8-checkpoint-v1/saved-audit-v1.json'
assert sha(ap) == '1413f26f3269ed8b7379763077314f3090de1467b76309d1af0ff063bb4db085'
audit = js(ap)
assert audit['passed']
sources = dict(audit['sources'])
for p,h in audit['new_files'].items():sources[str(ROOT/p)]=h
sources[str(ap)] = sha(ap)
for p in (Path(__file__), HERE / 'Run-VitHeadLayoutBodyV1.cmd', HERE / 'vit_head_layout_scope_v1.py', HERE / 'vit_head_major_projection_v1.py'):
    sources[str(p)] = sha(p)
assert all(sha(p) == h for p, h in sources.items())
provision = TOOLCHAIN / 'provision-v1.json'
assert sha(provision) == 'e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN / 'site' / p) == h for p, h in js(provision)['files'].items())
fixture_path = D / 'experimental/selected-body-stages-v1/validation.json'
assert sha(fixture_path) == 'f2553253c77b49e535c50f176afec1246abf5ee44784643bde1d25a820661e66'
fixture = js(fixture_path)
sources[str(fixture_path)] = sha(fixture_path)
try:
    with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=3) as response:queue=json.load(response)
    assert not queue['queue_running'] and not queue['queue_pending']
    queue_check='running_and_empty'
except urllib.error.URLError as error:
    assert getattr(error.reason,'winerror',None)==10061 or getattr(error.reason,'errno',None)==10061
    queue_check='connection_refused_service_not_running'

import numpy as np
import torch
import triton
from triton.compiler.compiler import CompiledKernel
from nr256_selected_stack_v3 import Stack
from nr_backend.execution import use_arithmetic_backend
from vit_head_layout_scope_v1 import VitHeadLayout
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
import immutable_artifacts_v1 as artifacts
assert Path(triton.__file__).is_relative_to(TOOLCHAIN / 'site') and triton.__version__.startswith('3.8.0')
OUT.mkdir()
report = dict(scope=__doc__, sources=sources, exact_gate=gate, passed=False,
    complete_migration=False, candidate_promoted=False, resources={}, builds={},
    torch_version=torch.__version__, triton_version=triton.__version__,
    body_inputs=fixture['body_inputs'], expected_output=fixture['expected_output'],
    samples_seconds={'selected': [], 'vit_layout': []}, orders=[], changed_input_checks=[])
save = lambda: (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
raw = lambda t: t.cpu().numpy().tobytes()
graphs = []
stack=None
scopes={}


try:
    torch.set_num_threads(2)
    stack = Stack(EXACT)
    static = {name: None if meta is None else torch.from_numpy(arrays.load(meta)).to('xpu')
              for name, meta in fixture['body_inputs'].items()}
    expected = arrays.load(fixture['expected_output']).tobytes()
    options = dict(sigmoid=stack.model.sigmoid, blend_scale=stack.model.blend_scale, return_float32=False)
    references={}
    report['boundaries']=[]
    with torch.inference_mode(), stack.installed(), body.installed(), use_arithmetic_backend('triton'):
        adapter=stack.components[-1];original_apply=adapter.apply
        assert 'apply' not in adapter.__dict__
        indices={id(b):i for i,b in enumerate(stack.model.vit)}
        def reference_apply(module,features):
            outputs=original_apply(module,features);i=indices[id(module)]
            assert i not in references
            references[i]=dict(input=arrays.save(features.cpu().numpy()),outputs=[arrays.save(t.cpu().numpy()) for t in outputs])
            return outputs
        adapter.apply=reference_apply
        try:
            assert raw(body.forward_front(stack.model,**static,**options))==expected
        finally:
            del adapter.apply
        assert len(references)==8
        def probe(index,features,outputs,heads):
            ref=references[index]
            assert raw(features)==arrays.load(ref['input']).tobytes()
            assert len(outputs)==len(ref['outputs'])==7
            actual=[arrays.save(t.cpu().numpy()) for t in outputs]
            for a,b in zip(actual,ref['outputs']):assert arrays.load(a).tobytes()==arrays.load(b).tobytes(),(index,'boundary')
            packed=arrays.save(heads.cpu().numpy())
            assert arrays.load(packed).transpose(1,0,2).reshape(64,1024).tobytes()==arrays.load(ref['outputs'][5]).tobytes()
            report['boundaries'].append(dict(block=index,input=ref['input'],reference=ref['outputs'],actual=actual,heads=packed,all_seven_equal=True))
        boundary_scope=VitHeadLayout(stack.model,stack.provider)
        boundary_scope.probe=probe
        with boundary_scope.installed():
            assert raw(body.forward_front(stack.model,**static,**options))==expected
        assert len(report['boundaries'])==8 and boundary_scope.calls==8
        boundary_scope.verify_restored();save()
        sources.update(stack.rewrite.fork.sources)
        report['queue_check']=queue_check
        pool, stream = torch.xpu.graph_pool_handle(), torch.xpu.Stream()
        entries = {}
        for name in ('selected', 'vit_layout'):
            scope = VitHeadLayout(stack.model,stack.provider,enabled=name=='vit_layout')
            scopes[name]=scope
            before = len(stack.rewrite.builds)
            with scope.installed():
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
                if name=='selected':
                    assert (build['triton_calls'],build['standalone_fp8'],build['quantization_calls'],build['elided_fp8'])==(683,196,427,231)
                else:
                    assert build['triton_calls']>0 and build['elided_fp8']>0
            entries[name] = (graph, owned)
            print('Captured whole body ' + name, flush=True)
            save()
        report['vit_calls']={n:s.calls for n,s in scopes.items()}
        report['vit_blocks']=scopes['vit_layout'].blocks
        assert scopes['selected'].calls==0 and scopes['vit_layout'].calls==24
        assert scopes['vit_layout'].blocks==dict.fromkeys(range(8),3)
        report['resources']=scopes['vit_layout'].resources
        assert report['resources'] and all(v['spills']==0 for v in report['resources'].values())
        report['short_fp8_resources']=stack.rewrite.metadata()['resources']
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
            order = ['selected', 'vit_layout'] if repetition % 2 == 0 else ['vit_layout', 'selected']
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
        report['change_percent'] = (report['median_seconds']['vit_layout'] / report['median_seconds']['selected'] - 1) * 100
        report['all_outputs_byte_equal'] = True
        assert all(raw(t) == arrays.load(fixture['body_inputs'][n]).tobytes() for n, t in static.items() if t is not None)
        assert stack.model._previous is None and stack.model.next_seed == 0
        stack.graph._validate()
        report['inputs_model_history_seed_unchanged'] = True
    for scope in scopes.values():scope.verify_restored()
    report['bindings_restored'] = True
    report['passed'] = True
except Exception:
    report['resources']={n:s.resources for n,s in scopes.items()}
    report['error'] = traceback.format_exc()
    raise
finally:
    for graph in graphs:
        graph.reset()
    if stack is not None:
        stack.close()
    for scope in scopes.values():scope.verify_restored()
    try:
        assert all(sha(p) == h for p,h in sources.items());authenticate_main()
    except BaseException:
        report.update(passed=False,finalization_error=traceback.format_exc());raise
    finally:save()
print(json.dumps(dict(passed=report['passed'], median_ms={n: v * 1000 for n, v in report.get('median_seconds', {}).items()},
                     change_percent=report.get('change_percent')), indent=2), flush=True)
