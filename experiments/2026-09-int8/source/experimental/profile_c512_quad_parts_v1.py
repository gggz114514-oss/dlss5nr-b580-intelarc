"""Locate the neutral full-model result using frozen kernels and real inputs.

Trace all16 C512 blocks on car1 after car0 reset. Capture six isolated graphs:
old/new attention, projection, and attention->projection. Each contains all16
blocks on owned observed inputs. Medians are nonadditive diagnostics, not an
end-to-end speedup. No new arithmetic, profiler, video or raw tensor files.
"""
from layout_crop_validation_env_v1 import *
import statistics, time, traceback
import numpy as np
import torch, triton
from PIL import Image
from c512_quad_query_stack_v1 import Stack
import c512_quad_queries_v2 as quad
import window_block_attention_v3 as old_attention
import c512_window_projection_v1 as old_projection
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
from residual_scale_v1 import ResidualScale
from nr_backend.execution import use_arithmetic_backend

OUT = D/'results/c512-quad-parts-v1'
assert not OUT.exists()
full = receipt(D/'results/c512-quad-full-v1/validation.json',
    'dc739e5351b38a091a5392fe535d1a04158e6c064428848367441b424e8153e0')
screen = receipt(D/'results/c512-quad-queries-v2/validation.json',
    '546d2ee7b20478fdacaa7a6bde4bc547b02e4dd8b490685669578b405ce5bb89')
calibration = receipt(D/'results/c512-int8-range-v1/validation.json',
    '97756e1d6c4af2c7d4cf97120ed0778820b9cf594232167ffe5ea451f018fda4')
repair = receipt(D/'results/int8-ffn-range-repair-v1/validation.json',
    'bd53a2e2b818886e79fed05e836646bdd89814a434aa932ae5bdd4f7944ce2aa')
handoff = D/'results/c512-quad-full-v1-monitor-luna-v1/handoff.json'
assert sha(handoff) == '866a406cf7e726f2d1a784a72c771cfa8439435888a875904290d2eae165fda9'
assert js(handoff)['passed'] and full['all_frozen_outputs_equal']
assert not full['candidate_promoted']
expected_kernels = {v['label']: v['hash'] for v in screen['resources']}
for p in (Path(__file__), HERE/'audit_c512_quad_parts_v1.py',
          HERE/'Run-C512QuadPartsV1.cmd', HERE/'Run-C512QuadPartsV1Audit.cmd',
          HERE/'C512_QUAD_PARTS_DESIGN_V1.md', handoff):
    sources[str(p)] = sha(p)
OUT.mkdir()
report = dict(passed=False, phase='initializing', scope=__doc__, sources=sources,
    exact_gate=gate, queue_check=queue_check, warmup=[], fixtures=[], resources={},
    workloads=[], orders=[], live_checks=[], candidate_promoted=False,
    arithmetic_changed=False, default_unchanged=True, profiler_used=False,
    raw_tensor_files=False, new_video=False, end_to_end_speedup_claimed=False,
    diagnostic_medians_are_not_additive=True, rounds=12, replays_per_sample=32)
stack, fixtures, workloads = None, {}, []
raw = lambda t: t.cpu().numpy().tobytes()
digest = lambda t: hashlib.sha256(raw(t)).hexdigest()


def save():
    p = OUT/'progress.tmp'
    p.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    p.replace(OUT/'validation.json')


def phase(value):
    report['phase'] = value
    save()
    print(json.dumps(dict(phase=value)), flush=True)


def constants():
    return {n: digest(t) for n, t in stack.model.named_buffers() if n != '_previous'}


def compact(t, inverse, shift):
    a = t.cpu().numpy()[..., inverse.cpu().numpy(), :]
    hw = a.reshape(16, 2, 2, 8, 8, 32).transpose(0, 1, 3, 2, 4, 5).reshape(16, 16, 16, 32)
    sy, sx = shift
    return np.ascontiguousarray(hw[:, sy:sy+12, sx:sx+12].reshape(16, 3, 4, 3, 4, 32)
                               .transpose(0, 1, 3, 2, 4, 5).reshape(16, 9, 16, 32)).tobytes()


def probe(kind, module, operands, result):
    name = stack.c512_int8.modules[id(module)]
    if kind == 'attention':
        assert name not in fixtures
        q, k, v, bias = operands
        assert bias is module.attention.bias
        fixtures[name] = dict(module=module, q=q.clone(), k=k.clone(), v=v.clone(),
                              new=result.clone())
    else:
        assert kind == 'projection' and name in fixtures
        packed, weight, residual, scale, inverse = operands
        f = fixtures[name]
        assert raw(packed) == raw(f['new'])
        assert weight is module.projection.weight and scale is module.projection.skip_scale
        assert inverse is module.attention.pixel_inverse
        f.update(residual=residual.clone(), full=result.clone())


def compile_job(label, jit, args, grid, options):
    kernel = jit.warmup(*args, grid=grid, **options)
    kernel._init_handles()
    assert kernel.hash == expected_kernels[label] and kernel.n_spills == 0, (label, kernel.hash, kernel.n_spills)
    report['resources'][label] = dict(hash=kernel.hash, spills=kernel.n_spills,
        registers=kernel.n_regs, shared_bytes=kernel.metadata.shared, options=options)
    return jit, args, grid, options, kernel


def emit(work):
    for jit, args, grid, options, kernel in work['jobs']:
        assert jit[grid](*args, **options) is kernel


def compare_outputs(stage):
    old, new = [next(w for w in workloads if w['row']['name'] == stage+'/'+r) for r in ('old', 'quad')]
    for i, f in enumerate(fixtures.values()):
        if stage == 'attention':
            assert compact(old['out'][i], f['module'].attention.pixel_inverse, f['module'].window_shift) == raw(new['out'][i])
        else:
            assert raw(old['out'][i]) == raw(new['out'][i])


save()
try:
    assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
    torch.set_num_threads(2)
    stack = Stack(EXACT, hidden_scales=[arrays.load(r['hidden_scale']) for r in repair['calibration']['layers']],
                  calibration=calibration['calibration'])
    sources.update(stack.rewrite.fork.sources)
    report['runtime'] = dict(torch=torch.__version__, triton=triton.__version__, device=torch.xpu.get_device_name())
    report['constant_hashes_before'] = constants()
    assert report['constant_hashes_before'] == full['constant_guard']['compact']['before']
    model, scaler = stack.model, ResidualScale(256)
    inputs = R/'inputs/flow-full-1920x1080-v3'
    mp = inputs/'manifest.json'
    assert sha(mp) == full['sources'][str(mp)]
    manifest = js(mp)
    expected = {r['frame']: r for r in full['paired']['compact'] if r['round'] == 0}
    with torch.inference_mode():
        phase('warm_actual_model')
        for i in (0, 1):
            spec = manifest['frames'][i]
            p, f = inputs/spec['file'], inputs/spec['motion_file']
            for path in (p, f):
                assert sha(path) == full['sources'][str(path)]
                sources[str(path)] = sha(path)
            a = np.asarray(Image.open(p).convert('RGB')).astype('f4')/255
            m = np.fromfile(f, '<f4').reshape(1080, 1920, 2)
            rgb, motion = torch.from_numpy(a).to('xpu'), torch.from_numpy(m).to('xpu')
            with stack.installed(), use_arithmetic_backend('triton'):
                canvas, flow = scaler.prepare(rgb, motion)
                low = model(canvas, flow, reset=i == 0)
                out = scaler.composite(rgb, canvas, low)
            assert digest(low) == expected[i]['low_raw_sha256'] and digest(out) == expected[i]['full_raw_sha256']
            assert raw(rgb) == a.tobytes() and raw(motion) == m.tobytes()
            report['warmup'].append(dict(frame=i, byte_equal=True, low_raw_sha256=digest(low), full_raw_sha256=digest(out)))
        assert stack.graph.replays == 2 and stack.rewrite.scopes == 6
        entry = stack.graph.last_entry
        original_inputs = {k: None if t is None else digest(t) for k, t in entry.inputs.items()}
        expected_body = raw(entry.output)
        previous, history, seed = model._previous, raw(model._previous), model.next_seed
        phase('capture_real_operands')
        assert stack.compact_queries.probe is None
        stack.compact_queries.probe = probe
        try:
            with stack.installed(), body.installed(), use_arithmetic_backend('triton'):
                value = body.forward_front(model, **entry.inputs, sigmoid=model.sigmoid,
                                           blend_scale=model.blend_scale, return_float32=False)
            assert raw(value) == expected_body
        finally:
            stack.compact_queries.probe = None
        assert list(fixtures) == list(stack.c512_int8.packed) and len(fixtures) == 16
        frozen_boundaries = {(v['block'], v['kind']): v for v in full['boundaries'] if v['label'] == 'temporal'}
        with stack.rewrite.fork.installed(), stack.rewrite.resource_gate():
            for name, f in fixtures.items():
                module = f['module']
                old, _, _ = old_attention.forward(f['q'], f['k'], f['v'], module.attention.bias)
                assert compact(old, module.attention.pixel_inverse, module.window_shift) == raw(f['new'])
                old_full, _, _ = old_projection.forward(old, module.projection.weight, f['residual'],
                    module.projection.skip_scale, module.attention.pixel_inverse, shift=module.window_shift)
                assert raw(old_full) == raw(f['full'])
                f['old'] = old
                f['hashes'] = {k: digest(f[k]) for k in ('q', 'k', 'v', 'residual', 'old', 'new', 'full')}
                assert f['hashes']['new'] == frozen_boundaries[name, 'attention']['raw_sha256']
                assert f['hashes']['full'] == frozen_boundaries[name, 'projection']['raw_sha256']
                report['fixtures'].append(dict(name=name, shift=list(module.window_shift),
                    hashes=f['hashes'], attention_byte_equal=True, projection_byte_equal=True))
            phase('compile_and_capture_isolated_parts')
            stream, pool = torch.xpu.Stream(), torch.xpu.graph_pool_handle()
            for stage in ('attention', 'projection', 'pair'):
                for route in ('old', 'quad'):
                    row = dict(name=stage+'/'+route, stage=stage, route=route, samples_ms=[],
                               kernels_per_replay=16 if stage != 'pair' else 32)
                    work = dict(row=row, jobs=[], out=[], persistent=[])
                    workloads.append(work); report['workloads'].append(row)
                    for f in fixtures.values():
                        mod = f['module']; shift = mod.window_shift; is_old = route == 'old'
                        representation = 'old' if is_old else 'new'
                        ao = torch.empty_like(f[representation]) if stage != 'projection' else f[representation]
                        po = torch.empty_like(f['full'])
                        work['persistent'].extend([ao, po])
                        if stage != 'projection':
                            args = (f['q'], f['k'], f['v'], mod.attention.bias, ao)
                            jit, args, label, grid = ((old_attention._kernel, args+(4, 32), 'reference_attention', (2, 64))
                                if is_old else (quad._attend, args+shift, 'candidate_attention_'+str(shift), (9, 16)))
                            work['jobs'].append(compile_job(label, jit, args, grid,
                                dict(num_warps=4, num_stages=1, enable_fp_fusion=False)))
                        if stage != 'attention':
                            args = (ao, mod.projection.weight, f['residual'], mod.projection.skip_scale)
                            jit, args, label = ((old_projection._project,
                                args+(mod.attention.pixel_inverse, po, 16, 16, 12, 12, *shift, 16, 32),
                                'reference_projection_'+str(shift)) if is_old else (quad._project, args+(po,), 'candidate_projection'))
                            work['jobs'].append(compile_job(label, jit, args, (9, 16), dict(num_warps=4, enable_fp_fusion=False)))
                        work['out'].append(ao if stage == 'attention' else po)
                    with stream:
                        for _ in range(2):
                            emit(work)
                    torch.xpu.synchronize()
                    graph = torch.xpu.XPUGraph(); work['graph'] = graph
                    with torch.xpu.graph(graph, stream=stream, pool=pool):
                        emit(work)
                    graph.replay(); torch.xpu.synchronize()
                    work['expected'] = [f['hashes'][('old' if route == 'old' else 'new') if stage == 'attention' else 'full'] for f in fixtures.values()]
                    assert [digest(t) for t in work['out']] == work['expected']
                    row['output_hashes'] = work['expected']
                    row['captured_output_byte_equal'] = True
            assert set(report['resources']) == set(expected_kernels)
        # No Python numerical function or compiler call lies inside the timed loops.
        phase('live_input_checks')
        for stage, keys in [('attention', ('q', 'k', 'v')), ('projection', ('old', 'new')), ('pair', ('q', 'k', 'v'))]:
            saved = [(f[k], f[k].clone()) for f in fixtures.values() for k in keys]
            for t, _ in saved:
                t.zero_()
            targets = [w for w in workloads if w['row']['stage'] == stage]
            try:
                for w in targets:
                    w['graph'].replay()
                    assert [digest(t) for t in w['out']] != w['expected'], stage
                compare_outputs(stage)
            finally:
                for t, value in saved:
                    t.copy_(value)
                for w in targets:
                    w['graph'].replay()
                    assert [digest(t) for t in w['out']] == w['expected']
            report['live_checks'].append(dict(stage=stage, both_routes_changed=True, byte_equal=True, restored=True))
        phase('rotating_paired_parts_timing')
        for repeat in range(12):
            offset = repeat % 6
            order = workloads[offset:]+workloads[:offset]
            if repeat%2:
                order = order[::-1]
            report['orders'].append([w['row']['name'] for w in order])
            for w in order:
                torch.xpu.synchronize(); started = time.perf_counter_ns()
                for _ in range(32):
                    w['graph'].replay()
                torch.xpu.synchronize()
                w['row']['samples_ms'].append((time.perf_counter_ns()-started)/32e6)
                assert [digest(t) for t in w['out']] == w['expected']
            save()
        for w in workloads:
            w['row']['median_ms'] = statistics.median(w['row']['samples_ms'])
        for stage in ('attention', 'projection', 'pair'):
            compare_outputs(stage)
        assert all(f['hashes'] == {k: digest(f[k]) for k in f['hashes']} for f in fixtures.values())
        assert original_inputs == {k: None if t is None else digest(t) for k, t in entry.inputs.items()}
        assert raw(entry.output) == expected_body and model._previous is previous and raw(previous) == history and model.next_seed == seed
        report['constant_hashes_after'] = constants()
        assert report['constant_hashes_before'] == report['constant_hashes_after']
        persistent = [t for f in fixtures.values() for k, t in f.items() if isinstance(t, torch.Tensor)]
        persistent += [t for w in workloads for t in w['persistent']]
        segments = torch.xpu.memory_snapshot(pool)
        assert all(not any(v['address'] <= t.data_ptr() < v['address']+v['total_size'] for v in segments) for t in persistent)
        assert stack.rewrite.scopes == 7 and stack.graph.replays == 2 and stack.compact_queries.calls == 112
        with stack.installed():
            stack.graph._validate()
        stack.compact_queries.verify_restored(); stack.rewrite.fork.verify_restored()
        report.update(passed=True, phase='completed', real_body_byte_equal=True, fixture_inputs_unchanged=True,
            history_seed_and_body_inputs_unchanged=True, constants_unchanged=True, persistent_io_outside_pool=True,
            scopes_restored=True, public_graph_replays=2, rewrite_scopes=7)
except BaseException:
    report.update(passed=False, failure_phase=report['phase'], phase='failed', error=traceback.format_exc())
    raise
finally:
    try:
        for work in workloads:
            if 'graph' in work:
                work['graph'].reset()
        if stack is not None:
            stack.close()
        finalize_sources()
    except BaseException:
        report.update(passed=False, finalization_error=traceback.format_exc())
        raise
    finally:
        save()
print(json.dumps(dict(passed=True, medians_ms={w['name']: w['median_ms'] for w in report['workloads']}, diagnostic_only=True)), flush=True)
