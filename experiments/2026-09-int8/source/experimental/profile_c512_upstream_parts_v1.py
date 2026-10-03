"""Measure current C512 FFN, QKV and their real producer-consumer chain.

Frozen arithmetic,actual temporal16block inputs; isolated static graph diagnostics.
No new quantization,profiler,raw tensor files or end-to-end speedup claim.
"""
from layout_crop_validation_env_v1 import *
import statistics, time, traceback
import numpy as np
import torch, triton
from PIL import Image
from c512_quad_query_stack_v1 import Stack
import c512_upstream_jobs_v1 as upstream
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
from residual_scale_v1 import ResidualScale
from nr_backend.execution import use_arithmetic_backend

OUT = D/'results/c512-upstream-parts-v1'
assert not OUT.exists()
full = receipt(D/'results/c512-quad-full-v1/validation.json',
    'dc739e5351b38a091a5392fe535d1a04158e6c064428848367441b424e8153e0')
parts = receipt(D/'results/c512-quad-parts-v1/validation.json',
    '8972a47416ae82717f32453adbf5ab11924c9666631db5d36059515d26c652dd')
calibration = receipt(D/'results/c512-int8-range-v1/validation.json',
    '97756e1d6c4af2c7d4cf97120ed0778820b9cf594232167ffe5ea451f018fda4')
repair = receipt(D/'results/int8-ffn-range-repair-v1/validation.json',
    'bd53a2e2b818886e79fed05e836646bdd89814a434aa932ae5bdd4f7944ce2aa')
handoff = D/'results/c512-quad-full-v1-monitor-luna-v1/handoff.json'
assert sha(handoff) == '866a406cf7e726f2d1a784a72c771cfa8439435888a875904290d2eae165fda9'
assert js(handoff)['passed'] and full['all_frozen_outputs_equal']
assert not full['candidate_promoted']
assert parts['diagnostic_medians_are_not_additive']
parts_handoff = D/'results/c512-quad-parts-v1-monitor-luna-v1/handoff.json'
assert sha(parts_handoff) == '108e3b2491984191f56ee53e7a15638b39aaff00a543f49c7be4f0bf6f1317f0'
assert js(parts_handoff)['passed']
prior_fixtures = {v['name']: v for v in parts['fixtures']}
for p in (Path(__file__), HERE/'audit_c512_upstream_parts_v1.py', HERE/'c512_upstream_jobs_v1.py',
          HERE/'Run-C512UpstreamPartsV1.cmd', HERE/'Run-C512UpstreamPartsV1Audit.cmd',
          HERE/'C512_UPSTREAM_PARTS_DESIGN_V1.md', handoff, parts_handoff):
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


def ffn_probe(name, x, outputs):
    assert name not in fixtures
    fixtures[name] = dict(module=modules[name], x=x.clone(), mlp=outputs['out'].clone())


def qkv_probe(attention, features, shift, outputs):
    name = attention_names[id(attention)]
    f = fixtures[name]
    assert raw(features) == raw(f['mlp']) and shift == f['module'].window_shift
    for key, value in zip(('q', 'k', 'v'), outputs):
        f[key] = value.clone()


def compile_work(work):
    for job in work['jobs']:
        kernel = job['jit'].warmup(*job['args'], grid=job['grid'], **job['options'])
        kernel._init_handles()
        meta = dict(hash=kernel.hash, spills=kernel.n_spills,
                    registers=kernel.n_regs, shared_bytes=kernel.metadata.shared)
        assert meta == report['source_resources'][job['label']] and kernel.n_spills == 0, (job['label'], meta)
        job['kernel'] = kernel
        report['resources'][job['label']] = meta


def emit(work):
    for j in work['jobs']:
        assert j['jit'][j['grid']](*j['args'], **j['options']) is j['kernel']


save()
try:
    assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
    torch.set_num_threads(2)
    stack = Stack(EXACT, hidden_scales=[arrays.load(r['hidden_scale']) for r in repair['calibration']['layers']],
                  calibration=calibration['calibration'])
    sources.update(stack.rewrite.fork.sources)
    modules = dict(stack.model.named_modules())
    attention_names = {id(modules[n].attention): n for n in stack.c512_int8.packed}
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
        phase('capture_real_upstream_operands')
        scope = stack.c512_int8
        assert scope.ffn_probe is None and scope.qkv_probe is None
        scope.ffn_probe, scope.qkv_probe = ffn_probe, qkv_probe
        try:
            with stack.installed(), body.installed(), use_arithmetic_backend('triton'):
                value = body.forward_front(model, **entry.inputs, sigmoid=model.sigmoid,
                                           blend_scale=model.blend_scale, return_float32=False)
            assert raw(value) == expected_body
        finally:
            scope.ffn_probe = scope.qkv_probe = None
        assert list(fixtures) == list(scope.packed) and len(fixtures) == 16
        for name, f in fixtures.items():
            f['hashes'] = {k: digest(f[k]) for k in ('x', 'mlp', 'q', 'k', 'v')}
            prior = prior_fixtures[name]
            assert all(f['hashes'][k] == prior['hashes'][k] for k in ('q', 'k', 'v'))
            assert f['hashes']['mlp'] == prior['hashes']['residual']
            report['fixtures'].append(dict(name=name, shift=list(f['module'].window_shift),
                hashes=f['hashes'], actual_body_provenance=True, prior_boundary_hashes_equal=True))
        report['source_resources'] = {'ffn:'+k: dict(v) for k, v in scope.ffn_resources.items()}
        for key, meta in scope.compact_resources.items():
            label, kernel_hash = key.rsplit(':', 1)
            assert 'qkv:'+label not in report['source_resources']
            report['source_resources']['qkv:'+label] = dict(hash=kernel_hash, **meta)
        assert len(report['source_resources']) == 10
        report['source_selections'] = dict(ffn=scope.ffn_selections, qkv=scope.compact_selections)
        phase('compile_and_capture_upstream')
        stream, pool = torch.xpu.Stream(), torch.xpu.graph_pool_handle()
        with stack.rewrite.fork.installed(), stack.rewrite.resource_gate():
            for stage in ('ffn', 'qkv', 'chain'):
                work = upstream.build(stack, fixtures, stage)
                row = dict(name=stage, kernels_per_replay=len(work['jobs']), samples_ms=[],
                           output_hashes=work['expected'])
                work['row'] = row
                workloads.append(work); report['workloads'].append(row)
                compile_work(work)
                with stream:
                    for _ in range(2):
                        emit(work)
                torch.xpu.synchronize()
                graph = torch.xpu.XPUGraph(); work['graph'] = graph
                with torch.xpu.graph(graph, stream=stream, pool=pool):
                    emit(work)
                graph.replay(); torch.xpu.synchronize()
                assert [digest(t) for t in work['outputs']] == work['expected']
                row['captured_output_byte_equal'] = True
            assert report['resources'] == report['source_resources']
            phase('live_upstream_inputs')
            for work in workloads:
                stage = work['stage']; key = 'mlp' if stage == 'qkv' else 'x'
                saved = [(f[key], f[key].clone()) for f in fixtures.values()]
                for t, _ in saved:
                    t.zero_()
                try:
                    work['graph'].replay()
                    changed = [digest(t) for t in work['outputs']]
                    assert changed != work['expected'], stage
                    emit(work)
                    assert [digest(t) for t in work['outputs']] == changed
                finally:
                    for t, value in saved:
                        t.copy_(value)
                    work['graph'].replay()
                    assert [digest(t) for t in work['outputs']] == work['expected']
                report['live_checks'].append(dict(stage=stage, changed=True, eager_byte_equal=True, restored=True))
        phase('rotating_upstream_timing')
        for repeat in range(12):
            offset = repeat % 3
            order = workloads[offset:]+workloads[:offset]
            if repeat%2:
                order = order[::-1]
            report['orders'].append([w['stage'] for w in order])
            for w in order:
                torch.xpu.synchronize(); started = time.perf_counter_ns()
                for _ in range(32):
                    w['graph'].replay()
                torch.xpu.synchronize()
                w['row']['samples_ms'].append((time.perf_counter_ns()-started)/32e6)
                assert [digest(t) for t in w['outputs']] == w['expected']
            save()
        for w in workloads:
            w['row']['median_ms'] = statistics.median(w['row']['samples_ms'])
        assert all(f['hashes'] == {k: digest(f[k]) for k in f['hashes']} for f in fixtures.values())
        assert original_inputs == {k: None if t is None else digest(t) for k, t in entry.inputs.items()}
        assert raw(entry.output) == expected_body and model._previous is previous and raw(previous) == history and model.next_seed == seed
        report['constant_hashes_after'] = constants()
        assert report['constant_hashes_before'] == report['constant_hashes_after']
        persistent = [t for f in fixtures.values() for k, t in f.items() if isinstance(t, torch.Tensor)]
        persistent += [t for w in workloads for j in w['jobs'] for t in j['args'] if isinstance(t, torch.Tensor)]
        segments = torch.xpu.memory_snapshot(pool)
        assert all(not any(v['address'] <= t.data_ptr() < v['address']+v['total_size'] for v in segments) for t in persistent)
        assert stack.rewrite.scopes == 7 and stack.graph.replays == 2 and stack.compact_queries.calls == 112
        assert scope.ffn_probe is None and scope.qkv_probe is None
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
