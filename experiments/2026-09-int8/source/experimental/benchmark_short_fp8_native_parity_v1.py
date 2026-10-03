"""Paired complete resident-input NR256 versus the selected WindowBlocks stack.

360 reset/temporal calls per variant, including state commit and completion.
No residual scaling, decode, flow estimation, upload, presentation, JIT or game
contention. Full outputs must match their original frozen fast-branch hashes.
This preserves approved fast math; it does not establish exact-native parity.
"""
from short_fp8_validation_env_v1 import *
import statistics
import time
import traceback
from pathlib import PureWindowsPath
OUT = D / 'results/short-fp8-native-parity-v1'
assert not OUT.exists()
for p in (Path(__file__), HERE / 'Run-ShortFp8NativeParityV1.cmd'):
    sources[str(p)] = sha(p)
legacy = receipt(D / 'results/nr256-native-parity-v1/validation.json',
    '5008dfcf5316aabe984e1e30d1db1c933bb3536c5e4a54a3020390c4845ff007')
native = receipt(D / 'experimental/native-steady-bench-v1/validation.json',
    '70a86900dbde71d285f187cedc42a123bc7706cf27f21f0231e515b4d964659d')
legacy_hashes = {(r['round'], r['mode'], r['sample']): r['output_raw_sha256']
    for r in legacy['measured']['native_half']}
old_path = R / '4060-real-flow-256x256-sequence-v3.json'
old = js(old_path)
folder = R / 'results' / PureWindowsPath(old['runDirectory']).name / 'output'
input_path = folder / 'frame00.png_input.rgba32f.bin'
motion_path = folder / 'frame00.png_motion.rg32f.bin'
sources[str(old_path)] = sha(old_path)
for p in (input_path, motion_path):
    assert sha(p) == next(f['sha256'].lower() for f in old['outputs'] if f['name'] == p.name)
    sources[str(p)] = sha(p)
import numpy as np
import torch
import triton
from nr256_selected_stack_v2 import Stack
from short_fp8_graph_v1 import ShortFP8Graph
from nr_backend.execution import use_arithmetic_backend
from cubic_lut_constant_v1 import Constant, TABLE
from serial_graph_workspace_v1 import share_before_capture
import compressed_arrays_v1 as arrays
assert Path(triton.__file__).is_relative_to(TOOLCHAIN / 'site')
OUT.mkdir()
report = dict(scope=__doc__, sources=sources, exact_gate=gate, passed=False,
    complete_migration=False, candidate_promoted=False, measured={'selected': [], 'short_fp8': []},
    nr_input=[256,256], native_4060_reference=native['cases']['256x256']['statistics'])
save = lambda: (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
raw = lambda t: t.cpu().numpy().tobytes()
digest = lambda a: hashlib.sha256(a.tobytes()).hexdigest()
stacks, held = {}, {}

def run(name, reset):
    s = stacks[name]
    before = getattr(s.rewrite, 'scopes', None)
    with s.installed(), use_arithmetic_backend('triton') as dispatch:
        torch.xpu.synchronize()
        start = time.perf_counter()
        value = s.model(rgb, motion, reset=reset)
        torch.xpu.synchronize()
        ms = (time.perf_counter() - start) * 1000
    effective = dict(dispatch)
    assert effective.pop('xpu_graph_replay') == 1
    for key, count in s.graph.last_entry.dispatch.items():
        if key != 'backend':
            effective[key] = effective.get(key, 0) + count
    if before is not None and s.rewrite.scopes == 6:
        assert before in (3, 6)
    return value, ms, effective

try:
    torch.set_num_threads(2)
    pixels = np.fromfile(input_path, '<f4').reshape(256,256,4)[...,:3].copy()
    flow = np.fromfile(motion_path, '<f4').reshape(256,256,2).copy()
    assert not flow.any()
    rgb, motion = torch.from_numpy(pixels).to('xpu'), torch.from_numpy(flow).to('xpu')
    stacks['selected'] = Stack(EXACT)
    stacks['short_fp8'] = Stack(EXACT, rewrite_type=ShortFP8Graph, share_with=stacks['selected'])
    sources.update(stacks['short_fp8'].rewrite.fork.sources)
    report['shared_constants'] = stacks['short_fp8'].shared
    report['shared_transient_pool'] = share_before_capture([s.graph for s in stacks.values()])
    report['runtime'] = dict(torch=torch.__version__, triton=triton.__version__, device=torch.xpu.get_device_name())
    with torch.inference_mode():
        first = {}
        # Candidate goes first: prove that preload, not baseline warmup, supplies its dependencies.
        for name in ('short_fp8', 'selected'):
            value, _, _ = run(name, True)
            first[name] = value.cpu().numpy().copy()
            held[name] = value, raw(value)
            run(name, False)
            print('Prewarmed ' + name, flush=True)
        assert first['selected'].tobytes() == first['short_fp8'].tobytes() == arrays.load(legacy['reset_output']).tobytes()
        report['reset_output'] = arrays.save(first['short_fp8'])
        assert stacks['short_fp8'].rewrite.scopes == 6
        for r in range(3):
            for position in range(2):
                mode = ('reset', 'temporal')[(position+r) % 2]
                reset = mode == 'reset'
                for name in stacks:
                    run(name, True)
                    for _ in range(20):
                        warm, _, _ = run(name, reset)
                    if name == 'selected':
                        warm_bytes = raw(warm)
                    else:
                        assert raw(warm) == warm_bytes
                for i in range(60):
                    order = list(stacks)
                    if (r+i) % 2:
                        order.reverse()
                    values, dispatches = {}, {}
                    for name in order:
                        value, ms, effective = run(name, reset)
                        a = value.cpu().numpy()
                        assert a.shape == (256,256,3) and a.dtype == np.dtype('f2') and np.isfinite(a).all()
                        assert raw(stacks[name].model._previous) == a.tobytes()
                        assert stacks[name].model.next_seed == (1 if reset else 22+i)
                        assert digest(a) == legacy_hashes[(r,mode,i)], (name,r,mode,i)
                        values[name], dispatches[name] = a.copy(), effective
                        report['measured'][name].append(dict(round=r, mode=mode, sample=i,
                            host_ms=ms, output_raw_sha256=digest(a), next_seed=stacks[name].model.next_seed))
                        value.zero_()
                        assert raw(stacks[name].model._previous) == a.tobytes()
                    assert values['selected'].tobytes() == values['short_fp8'].tobytes()
                    assert dispatches['selected'] == dispatches['short_fp8']
                for v, data in held.values():
                    assert raw(v) == data
                save()
                print(json.dumps(dict(round=r, mode=mode, all60equal=True,
                    host_ms={n:statistics.mean(v['host_ms'] for v in report['measured'][n][-60:]) for n in stacks})), flush=True)
        for name, s in stacks.items():
            value, _, _ = run(name, True)
            assert raw(value) == first[name].tobytes() and s.model.next_seed == 1
            assert raw(Constant(s.model).require()) == np.load(TABLE, allow_pickle=False).view('i2').tobytes()
            s.graph._validate()
        assert raw(rgb) == pixels.tobytes() and raw(motion) == flow.tobytes()
        report['summary'] = {n:{mode:dict(mean_host_ms=statistics.mean(s['host_ms'] for s in rows if s['mode']==mode),
            round_mean_host_ms=[statistics.mean(s['host_ms'] for s in rows if s['mode']==mode and s['round']==r) for r in range(3)])
            for mode in ('reset','temporal')} for n,rows in report['measured'].items()}
        report['builds'] = {n:s.rewrite.builds for n,s in stacks.items()}
        assert all(len(s.rewrite.builds) == 6 for s in stacks.values())
        assert all((b['triton_calls'],b['standalone_fp8'],b['quantization_calls'],b['elided_fp8']) == (683,196,427,231)
            for s in stacks.values() for b in s.rewrite.builds)
        report['candidate'] = stacks['short_fp8'].rewrite.metadata()
        assert report['candidate']['capture_scopes'] == 6 and report['candidate']['resources']
        assert all(v['spills'] == 0 for v in report['candidate']['resources'].values())
        report.update(passed=True, full_outputs_verified=720, all_outputs_match_frozen_fast=True,
            held_outputs_unchanged=True, inputs_lut_unchanged=True, caller_ownership_passed=True,
            no_dependency_swaps_on_steady_replay=True, graphs={n:s.graph.metadata() for n,s in stacks.items()})
except BaseException:
    report['error'] = traceback.format_exc()
    raise
finally:
    try:
        for s in stacks.values():
            s.close()
        finalize_sources()
    except BaseException:
        report.update(passed=False, finalization_error=traceback.format_exc())
        raise
    finally:
        save()
print(json.dumps(dict(passed=report['passed'],summary=report.get('summary'))),flush=True)
