"""Three paired rounds of complete 1080p residual scaling plus NR256.

Includes prepare/composite, complete model call, history commit and completion.
Excludes decode, flow estimation, upload, verification, display, JIT and game
contention. All full outputs and low NR results must match frozen fast outputs.
"""
from layout_crop_validation_env_v1 import *
import statistics
import time
import traceback
OUT = D / 'results/layout-crop-residual256-v1'
assert not OUT.exists()
for p in (Path(__file__), HERE / 'Run-LayoutCropValidationSuiteV1.cmd', HERE / 'run_layout_crop_validation_suite_v1.py', HERE / 'residual_scale_v1.py', HERE / 'cubic_lut_graph_guards_v2.py'):
    sources[str(p)] = sha(p)
native_path = D / 'results/layout-crop-native-parity-v1/validation.json'
native = receipt(native_path,'e54f31291b1b628d2d988bf19bbc8fe43cdb8f0c19d80def21ec7f2418973109')
assert native['passed'] and native['full_outputs_verified'] == 720
assert js(native_path.parent.with_suffix('.log.lease.json'))['returncode'] == 0
assert all(sha(p) == h for p,h in native['sources'].items())
sources.update(native['sources'])
sources[str(native_path)] = sha(native_path)
prior = receipt(D / 'results/residual-scale-fp16_xmx-256-v1/validation.json',
    '056251b4e2a839fcc44845d9646d2cd8ae1e9ed7f99fd1858dc7e3d51e3e50f9')
inputs = R / 'inputs/flow-full-1920x1080-v3'
manifest_path = inputs / 'manifest.json'
manifest = js(manifest_path)
sources[str(manifest_path)] = sha(manifest_path)
import numpy as np
from PIL import Image
import torch
import triton
from nr256_selected_stack_v3 import Stack
from layout_crop_stack_v1 import Stack as CandidateStack
from nr_backend.execution import use_arithmetic_backend
from residual_scale_v1 import ResidualScale
from serial_graph_workspace_v1 import share_before_capture
from cubic_lut_constant_v1 import Constant,TABLE
from cubic_lut_graph_guards_v2 import verify as verify_lut_guards
import compressed_arrays_v1 as arrays
assert Path(triton.__file__).is_relative_to(TOOLCHAIN / 'site')
OUT.mkdir()
report = dict(scope=__doc__, sources=sources, exact_gate=gate, passed=False,
    complete_migration=False, candidate_promoted=False, runs={'selected': [], 'layout_crop': []},
    guards=[], progress_fallback=[], queue_check=queue_check, logical_dispatch_delta={})
save = lambda: (OUT / 'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
raw = lambda t: t.cpu().numpy().tobytes()
stacks, held = {}, {}

def tensors(index):
    spec = manifest['frames'][index]
    paths = [inputs / spec['file'], inputs / spec['motion_file']]
    for p in paths:
        actual = sha(p)
        if str(p) in sources:
            assert sources[str(p)] == actual
        sources[str(p)] = actual
    pixels = np.asarray(Image.open(paths[0]).convert('RGB'),dtype='f4') / 255
    flow = np.fromfile(paths[1],'<f4').reshape(1080,1920,2)
    return pixels,flow,torch.from_numpy(pixels).to('xpu'),torch.from_numpy(flow).to('xpu')

def run(name,rgb,motion,reset):
    s = stacks[name]
    with s.installed(),use_arithmetic_backend('triton') as dispatch:
        torch.xpu.synchronize()
        start = time.perf_counter()
        low_rgb,low_motion = scaler.prepare(rgb,motion)
        low = s.model(low_rgb,low_motion,reset=reset)
        output = scaler.composite(rgb,low_rgb,low)
        torch.xpu.synchronize()
        ms = (time.perf_counter()-start)*1000
    effective = dict(dispatch)
    assert effective.pop('xpu_graph_replay') == 1
    for key,count in s.graph.last_entry.dispatch.items():
        if key != 'backend':
            effective[key] = effective.get(key,0)+count
    return output,low,ms,effective

try:
    torch.set_num_threads(2)
    scaler = ResidualScale(256)
    expected = [(arrays.load(row['output']),arrays.load(row['low_nr'])) for row in prior['runs'][:13]]
    assert len(manifest['frames']) == len(expected) == 13
    stacks['selected'] = Stack(EXACT)
    stacks['layout_crop'] = CandidateStack(EXACT,share_with=stacks['selected'])
    sources.update(stacks['layout_crop'].rewrite.fork.sources)
    report['shared_constants'] = stacks['layout_crop'].shared
    report['shared_transient_pool'] = share_before_capture([s.graph for s in stacks.values()])
    report['runtime'] = dict(torch=torch.__version__,triton=triton.__version__,device=torch.xpu.get_device_name())
    with torch.inference_mode():
        for name in ('layout_crop','selected'):
            s = stacks[name]
            for i in (0,1):
                _,_,rgb,motion = tensors(i)
                output,low,_,_ = run(name,rgb,motion,i==0)
                assert raw(output) == expected[i][0].tobytes() and raw(low) == expected[i][1].tobytes()
            s.model.reset()
            print('Prewarmed '+name,flush=True)
        for r in range(3):
            for i,spec in enumerate(manifest['frames']):
                pixels,flow,rgb,motion = tensors(i)
                order = list(stacks)
                if (r+i)%2:
                    order.reverse()
                dispatches = {}
                for name in order:
                    output,low,ms,effective = run(name,rgb,motion,spec['reset'])
                    s = stacks[name]
                    assert raw(output) == expected[i][0].tobytes() and raw(low) == expected[i][1].tobytes()
                    assert raw(s.model._previous) == raw(low)
                    assert s.model.next_seed == (1 if spec['reset'] else i+1)
                    assert np.isfinite(output.cpu().numpy()).all()
                    report['runs'][name].append(dict(round=r,frame=i,reset=spec['reset'],order=order,
                        host_ms=ms,output=prior['runs'][i]['output'],low_nr=prior['runs'][i]['low_nr'],
                        byte_equal=True,next_seed=s.model.next_seed,effective_dispatch=effective))
                    dispatches[name] = effective
                    if name not in held:
                        held[name] = output,raw(output)
                    else:
                        private = raw(s.model._previous)
                        output.zero_();low.zero_()
                        assert raw(s.model._previous) == private
                mode='reset' if spec['reset'] else 'temporal'
                delta={key:dispatches['layout_crop'].get(key,0)-dispatches['selected'].get(key,0)
                    for key in dispatches['selected'].keys()|dispatches['layout_crop'].keys()
                    if dispatches['selected'].get(key)!=dispatches['layout_crop'].get(key)}
                assert delta==native['logical_dispatch_delta'][mode]
                report['logical_dispatch_delta'][mode]=delta
                for value,previous in held.values():
                    assert raw(value) == previous
                assert raw(rgb) == pixels.tobytes() and raw(motion) == flow.tobytes()
            save()
            print(json.dumps(dict(round=r,all13equal=True,mean_host_ms={n:statistics.mean(v['host_ms'] for v in report['runs'][n][-13:]) for n in stacks})),flush=True)
        for name,s in stacks.items():
            _,_,rgb0,flow0 = tensors(0)
            run(name,rgb0,flow0,True)
            _,_,rgb,motion = tensors(1)
            low_rgb,low_motion = scaler.prepare(rgb,motion)
            before,marks = s.graph.replays,[]
            scope_before = getattr(s.rewrite,'scopes',None)
            route_before=None if name=='selected' else (s.rewrite.layout.calls,s.rewrite.vit_layout.calls,s.rewrite.post_region.calls)
            with s.installed(),use_arithmetic_backend('triton'):
                low = s.model(low_rgb,low_motion,reset=False,progress=marks.append)
                output = scaler.composite(rgb,low_rgb,low)
            assert raw(output) == expected[1][0].tobytes() and s.model.next_seed == 2 and s.graph.replays == before
            assert marks == ['pre','encoder C32','encoder C64','encoder C128','encoder C256','encoder C512','ViT','decoder C512','decoder C256','decoder C128','decoder C64','decoder C32','RGB']
            assert getattr(s.rewrite,'scopes',None) == scope_before
            if route_before is not None:
                assert route_before==(s.rewrite.layout.calls,s.rewrite.vit_layout.calls,s.rewrite.post_region.calls)
                for scope in (s.rewrite.layout,s.rewrite.vit_layout,s.rewrite.post_region):scope.verify_restored()
            report['progress_fallback'].append(dict(name=name,marks=marks,byte_equal=True,short_rounder_bypassed=True,layout_crop_bypassed=True))
            private,private_bytes = s.model._previous,raw(s.model._previous)
            state = s.model.next_seed,s.graph.replays,s.warp.replays
            with s.installed(),use_arithmetic_backend('triton'):
                bad = low_motion.clone();bad[0,0,0] = float('nan')
                try:
                    s.model(low_rgb,bad,reset=False)
                except ValueError as error:
                    assert 'Motion must be finite' in str(error)
                else:
                    raise AssertionError('Invalid motion accepted')
                original = s.model.reciprocal.values
                with torch.inference_mode(False):
                    replacement = original.clone()
                s.model.reciprocal.values = replacement
                try:
                    try:
                        s.model(low_rgb,low_motion,reset=False)
                    except RuntimeError as error:
                        assert 'reciprocal table changed' in str(error)
                    else:
                        raise AssertionError('Changed table accepted')
                finally:
                    s.model.reciprocal.values = original
            assert s.model._previous is private and raw(private) == private_bytes
            assert (s.model.next_seed,s.graph.replays,s.warp.replays) == state
            report['guards'].append(dict(name=name,invalid_motion_and_table_rejected=True,history_seed_replays_unchanged=True))
            assert raw(Constant(s.model).require()) == np.load(TABLE,allow_pickle=False).view('i2').tobytes()
        s = stacks['layout_crop']
        with s.installed(),use_arithmetic_backend('triton'):
            report['lut_graph_guards'] = verify_lut_guards(s.model,s.graph,lambda:s.model(low_rgb,low_motion,reset=False))
        report['builds'] = {n:s.rewrite.builds for n,s in stacks.items()}
        assert all(len(s.rewrite.builds) == 6 for s in stacks.values())
        assert all((b['triton_calls'],b['standalone_fp8'],b['quantization_calls'],b['elided_fp8']) == (683,196,427,231)
            for b in stacks['selected'].rewrite.builds)
        report['candidate'] = candidate_gates(stacks['layout_crop'])
        assert stacks['selected'].rewrite.scopes==6
        report['selected_resources']=stacks['selected'].rewrite.metadata()['resources']
        assert report['selected_resources'] and all(v['spills']==0 for v in report['selected_resources'].values())
        report['summary'] = {n:{mode:dict(mean_host_ms=statistics.mean(row['host_ms'] for row in rows if mode=='all' or not row['reset']),
            round_mean_host_ms=[statistics.mean(row['host_ms'] for row in rows if row['round']==r and (mode=='all' or not row['reset'])) for r in range(3)])
            for mode in ('all','temporal')} for n,rows in report['runs'].items()}
        report['change_percent']={mode:(report['summary']['layout_crop'][mode]['mean_host_ms']
            /report['summary']['selected'][mode]['mean_host_ms']-1)*100 for mode in ('all','temporal')}
        report.update(passed=True,full_outputs_verified=78,held_outputs_unchanged=True,inputs_lut_unchanged=True,
            caller_ownership_passed=True,no_dependency_swaps_on_steady_replay=True,graphs={n:s.graph.metadata() for n,s in stacks.items()})
except BaseException:
    report['error'] = traceback.format_exc()
    raise
finally:
    try:
        for s in stacks.values():
            s.close()
        finalize_sources()
    except BaseException:
        report.update(passed=False,finalization_error=traceback.format_exc())
        raise
    finally:
        save()
print(json.dumps(dict(passed=report['passed'],summary=report.get('summary'))),flush=True)
