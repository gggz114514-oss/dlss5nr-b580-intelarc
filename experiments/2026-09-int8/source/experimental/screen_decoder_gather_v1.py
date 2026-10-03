"""Byte-preserving decoder materialization screen on accepted floor16 NR256.

Thirteen frozen car frames cover reset and temporal bodies. Five real merge
boundaries plus finite-half/stride/padding controls must match raw bytes. Timing
compares the five merges separately from the intact temporal body; no frame-rate
claim, new video, model promotion or approximate QKV arithmetic.
"""
from layout_crop_validation_env_v1 import *
from contextlib import nullcontext
import statistics, time, traceback
import numpy as np
import torch, triton
from PIL import Image
import decoder_gather_merge_v1 as kernels
from decoder_gather_scope_v1 import DecoderGather
from c512_quad_query_stack_v1 import Stack
import capture_body_v1 as body
import nr_backend.decoder as dec
from nr_backend.execution import use_arithmetic_backend
from residual_scale_v1 import ResidualScale
import compressed_arrays_v1 as arrays

OUT = D/'results/decoder-gather-v1'
assert not OUT.exists()
full = receipt(D/'results/c512-quad-full-v1/validation.json',
    'dc739e5351b38a091a5392fe535d1a04158e6c064428848367441b424e8153e0')
calibration = receipt(D/'results/c512-int8-range-v1/validation.json',
    '97756e1d6c4af2c7d4cf97120ed0778820b9cf594232167ffe5ea451f018fda4')
repair = receipt(D/'results/int8-ffn-range-repair-v1/validation.json',
    'bd53a2e2b818886e79fed05e836646bdd89814a434aa932ae5bdd4f7944ce2aa')
assert full['all_frozen_outputs_equal']
for name in ('screen_decoder_gather_v1.py', 'decoder_gather_merge_v1.py', 'decoder_gather_scope_v1.py',
             'audit_decoder_gather_v1.py', 'Run-DecoderGatherV1.cmd', 'Run-DecoderGatherV1Audit.cmd',
             'DECODER_GATHER_DESIGN_V1.md'):
    sources[str(HERE/name)] = sha(HERE/name)
OUT.mkdir()
report = dict(passed=False, phase='initializing', scope=__doc__, sources=sources,
    exact_gate=gate, queue_check=queue_check, arithmetic_changed=False, new_quantization=False,
    candidate_promoted=False, default_unchanged=True, guards_removed=False,
    nvidia_byte_parity_claimed=False, raw_tensor_files=False, new_video=False,
    end_to_end_speedup_claimed=False, diagnostic_medians_are_not_additive=True,
    frames=[], boundaries=[], controls=[], workloads=[], orders=[], live_checks=[], resources={})
stack, scope, workloads, fixtures = None, None, [], []
raw = lambda t: t.cpu().numpy().tobytes()
digest = lambda t: hashlib.sha256(raw(t)).hexdigest()


def save():
    p = OUT/'progress.tmp'
    p.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    p.replace(OUT/'validation.json')


def phase(name):
    report['phase'] = name
    save()
    print(json.dumps(dict(phase=name)), flush=True)


def constants():
    return {n: digest(t) for n, t in stack.model.named_buffers() if n != '_previous'}


def reference(p, s, scale, shift):
    h, w = s.shape[:2]
    expanded = p.repeat_interleave(2, 0).repeat_interleave(2, 1)[:h, :w]
    merged = dec.half_fma(s, scale, expanded)
    if shift is None:
        return dec.q(merged)
    sy, sx = shift
    return torch.nn.functional.pad(merged, (0, 0, sx, (-w-sx) % 8, sy, (-h-sy) % 8))


def resource(label, meta):
    assert meta['spills'] == 0
    key = label+':'+meta['hash']
    if key in report['resources']:
        assert report['resources'][key] == meta
    report['resources'][key] = meta


def probe(name, p, s, scale, shift, out):
    # The recorded operands and output are real propagated model boundaries.
    # Compare later outside the model's FP8 provenance scope: do not inject a
    # second old path into its body analysis or timing.
    assert name not in [f['name'] for f in fixtures]
    fixtures.append(dict(name=name, p=p.clone(), s=s.clone(), scale=scale.clone(),
                         shift=shift, out=out.clone()))


def run_body(inputs, candidate):
    with scope.installed() if candidate else nullcontext():
        with stack.installed(), body.installed(), use_arithmetic_backend('triton'):
            return body.forward_front(stack.model, **inputs, sigmoid=stack.model.sigmoid,
                blend_scale=stack.model.blend_scale, return_float32=False)


def capture(name, fn, expected, stream, pool):
    with stream:
        for _ in range(2):
            warm = fn()
    torch.xpu.synchronize()
    assert [raw(t) for t in warm] == expected, name
    public = tuple(torch.empty_like(t, memory_format=torch.contiguous_format) for t in warm)
    del warm
    graph = torch.xpu.XPUGraph()
    work = dict(name=name, graph=graph, outputs=public, fn=fn, expected=expected)
    workloads.append(work)
    with torch.xpu.graph(graph, stream=stream, pool=pool):
        temporary = fn()
        for target, source in zip(public, temporary):
            target.copy_(source)
    del temporary
    graph.replay()
    torch.xpu.synchronize()
    assert [raw(t) for t in public] == expected, name
    row = dict(name=name, output_hashes=[hashlib.sha256(v).hexdigest() for v in expected],
               captured_matches_eager=True, elapsed_ns=[], samples_ms=[])
    work['row'] = row
    report['workloads'].append(row)
    return work


save()
try:
    torch.set_num_threads(2)
    stack = Stack(EXACT, hidden_scales=[arrays.load(v['hidden_scale']) for v in repair['calibration']['layers']],
                  calibration=calibration['calibration'])
    sources.update(stack.rewrite.fork.sources)
    scope = DecoderGather(stack.model)
    report['runtime'] = dict(torch=torch.__version__, triton=triton.__version__, device=torch.xpu.get_device_name())
    report['constant_hashes_before'] = constants()
    assert report['constant_hashes_before'] == full['constant_guard']['compact']['before']
    inputs = R/'inputs/flow-full-1920x1080-v3'
    mp = inputs/'manifest.json'
    assert sha(mp) == full['sources'][str(mp)]
    sources[str(mp)] = sha(mp)
    manifest = js(mp)
    expected = {v['frame']: v for v in full['paired']['compact'] if v['round'] == 0}
    assert len(expected) == len(manifest['frames']) == 13
    model, scaler = stack.model, ResidualScale(256)
    with torch.inference_mode():
        phase('frozen_car_bodies_reset_and_temporal')
        for i, spec in enumerate(manifest['frames']):
            p, f = inputs/spec['file'], inputs/spec['motion_file']
            for path in (p, f):
                assert sha(path) == full['sources'][str(path)]
                sources[str(path)] = sha(path)
            a = np.asarray(Image.open(p).convert('RGB')).astype('f4')/255
            m = np.fromfile(f, '<f4').reshape(1080, 1920, 2)
            rgb, motion = torch.from_numpy(a).to('xpu'), torch.from_numpy(m).to('xpu')
            reset = expected[i]['reset']
            with stack.installed(), use_arithmetic_backend('triton'):
                canvas, flow = scaler.prepare(rgb, motion)
                low = model(canvas, flow, reset=reset)
                out = scaler.composite(rgb, canvas, low)
            assert digest(low) == expected[i]['low_raw_sha256'] and digest(out) == expected[i]['full_raw_sha256']
            entry = stack.graph.last_entry
            ib = {k: None if t is None else digest(t) for k, t in entry.inputs.items()}
            previous, history, seed = model._previous, raw(model._previous), model.next_seed
            try:
                if i == 1:
                    scope.probe = probe
                candidate = run_body(entry.inputs, True)
            finally:
                scope.probe = None
            assert raw(candidate) == raw(low)
            with use_arithmetic_backend('triton'):
                composed = scaler.composite(rgb, canvas, candidate)
            assert raw(composed) == raw(out)
            assert ib == {k: None if t is None else digest(t) for k, t in entry.inputs.items()}
            assert model._previous is previous and raw(previous) == history and model.next_seed == seed
            assert raw(rgb) == a.tobytes() and raw(motion) == m.tobytes()
            if i == 1:
                static = {k: None if t is None else t.clone() for k, t in entry.inputs.items()}
                body_expected = raw(candidate)
            report['frames'].append(dict(frame=i, reset=reset, byte_equal=True, inputs_unchanged=True,
                history_unchanged=True, low_raw_sha256=digest(candidate), full_raw_sha256=digest(composed)))
            save()
        assert [f['name'] for f in fixtures] == [name for name, _ in scope.modules]
        phase('real_merge_boundaries_and_finite_half_controls')
        with stack.rewrite.fork.installed(), stack.rewrite.resource_gate(), use_arithmetic_backend('triton'):
            for f in fixtures:
                old = reference(f['p'], f['s'], f['scale'], f['shift'])
                new, meta = kernels.forward(f['p'], f['s'], f['scale'], shift=f['shift'])
                resource(f['name'], meta)
                assert raw(old) == raw(new) == raw(f['out'])
                row = dict(name=f['name'], projected_shape=list(f['p'].shape), skip_shape=list(f['s'].shape),
                    output_shape=list(new.shape), shift=f['shift'], byte_equal=True, actual_body_provenance=True,
                    hashes={k: digest(f[k]) for k in ('p', 's', 'scale', 'out')})
                report['boundaries'].append(row)
            rng = np.random.default_rng(192837)
            # Every finite half bit-pattern is used as a projected operand.
            bits = np.arange(65536, dtype='u2')
            bits[(bits & 0x7fff) >= 0x7c00] = 0
            allhalf = bits.view('f2').reshape(32, 64, 32)
            palette = np.array([-448., -1., -.5, -0., 0., .5, 1., 448.], dtype='f2')
            for case in ('finite_patterns', 'strided_odd_crop'):
                p = torch.from_numpy(allhalf.copy()).to('xpu')
                skip = torch.from_numpy(rng.choice(palette, (64, 128, 32))).to('xpu')
                scale = torch.from_numpy(rng.choice(np.array([-2., -1., -2**-14, -0., 0., 2**-14, 1., 2.], dtype='f2'), 64)).to('xpu')[::2]
                if case == 'strided_odd_crop':
                    p, skip = p[1::2, ::2], skip[1:59:2, 1:123:2]
                for shift in (None, (0, 0), (0, 4), (4, 0), (4, 4)):
                    old = reference(p, skip, scale, shift)
                    new, meta = kernels.forward(p, skip, scale, shift=shift)
                    resource(case+str(shift), meta)
                    assert raw(new) == raw(old), (case, shift)
                    report['controls'].append(dict(case=case, shift=shift, byte_equal=True,
                        projected_shape=list(p.shape), projected_stride=list(p.stride()),
                        skip_shape=list(skip.shape), skip_stride=list(skip.stride()), output_shape=list(new.shape),
                        scale_stride=list(scale.stride()), output_sha256=digest(new)))
        phase('capture_paired_body_and_five_merge_graphs')
        stream, pool = torch.xpu.Stream(), torch.xpu.graph_pool_handle()
        for candidate in (False, True):
            before = len(stack.rewrite.builds)
            work = capture('body_gather' if candidate else 'body_original',
                lambda candidate=candidate: (run_body(static, candidate),), [body_expected], stream, pool)
            builds = stack.rewrite.builds[before:]
            assert len(builds) == 3
            work['row']['capture_counts'] = [{k: b[k] for k in ('triton_calls', 'standalone_fp8', 'quantization_calls', 'elided_fp8')} for b in builds]
        def merges(candidate):
            with stack.rewrite.fork.installed(), stack.rewrite.resource_gate(), use_arithmetic_backend('triton'):
                if candidate:
                    return tuple(kernels.forward(f['p'], f['s'], f['scale'], shift=f['shift'])[0] for f in fixtures)
                return tuple(reference(f['p'], f['s'], f['scale'], f['shift']) for f in fixtures)
        merge_expected = [raw(f['out']) for f in fixtures]
        for candidate in (False, True):
            capture('merges_gather' if candidate else 'merges_original',
                    lambda candidate=candidate: merges(candidate), merge_expected, stream, pool)
        phase('captured_live_input_controls')
        originals = [(t, t.clone()) for t in (static['rgb'], static['front'], *[f[k] for f in fixtures for k in ('p', 's')])]
        for case in ('zero', 'negated'):
            try:
                for t, backup in originals:
                    t.copy_(torch.zeros_like(backup) if case == 'zero' else -backup)
                got = {}
                for work in workloads:
                    for out in work['outputs']:
                        out.fill_(.125)
                    work['graph'].replay()
                    values = [raw(t) for t in work['outputs']]
                    assert values != work['expected']
                    eager = work['fn']()
                    assert values == [raw(t) for t in eager]
                    got[work['name']] = [hashlib.sha256(v).hexdigest() for v in values]
                assert got['body_original'] == got['body_gather'] and got['merges_original'] == got['merges_gather']
            finally:
                for t, backup in originals:
                    t.copy_(backup)
                for work in workloads:
                    work['graph'].replay()
                    assert [raw(t) for t in work['outputs']] == work['expected']
            report['live_checks'].append(dict(case=case, all_changed=True, byte_equal=True, eager_equal=True, restored=True, hashes=got))
        persistent = [t for t in static.values() if t is not None]
        persistent += [f[k] for f in fixtures for k in ('p', 's', 'scale', 'out')]
        persistent += [t for w in workloads for t in w['outputs']]
        segments = torch.xpu.memory_snapshot(pool)
        assert all(not any(v['address'] <= t.data_ptr() < v['address']+v['total_size'] for v in segments) for t in persistent)
        phase('paired_host_completion_timing')
        report.update(rounds=12, replays_per_sample=32, warmup_replays=16, clock='perf_counter_ns_with_device_completion')
        for work in workloads:
            for _ in range(16):
                work['graph'].replay()
        torch.xpu.synchronize()
        for repeat in range(12):
            offset = repeat % len(workloads)
            order = workloads[offset:]+workloads[:offset]
            if repeat % 2:
                order = order[::-1]
            report['orders'].append([w['name'] for w in order])
            for work in order:
                torch.xpu.synchronize()
                started = time.perf_counter_ns()
                for _ in range(32):
                    work['graph'].replay()
                torch.xpu.synchronize()
                ns = time.perf_counter_ns()-started
                work['row']['elapsed_ns'].append(ns)
                work['row']['samples_ms'].append(ns/32e6)
                assert [raw(t) for t in work['outputs']] == work['expected']
            save()
        for work in workloads:
            work['row']['median_ms'] = statistics.median(work['row']['samples_ms'])
        by_name = {v['name']: v for v in report['workloads']}
        report['comparisons'] = []
        for family in ('body', 'merges'):
            a, b = by_name[family+'_original'], by_name[family+'_gather']
            report['comparisons'].append(dict(family=family, baseline_ms=a['median_ms'], candidate_ms=b['median_ms'],
                saving_ms=a['median_ms']-b['median_ms'], saving_percent=(a['median_ms']-b['median_ms'])/a['median_ms']*100,
                faster_rounds=sum(y<x for x, y in zip(a['samples_ms'], b['samples_ms']))))
        assert model._previous is previous and raw(previous) == history and model.next_seed == seed
        report['constant_hashes_after'] = constants()
        assert report['constant_hashes_after'] == report['constant_hashes_before']
        assert all({k: digest(f[k]) for k in ('p', 's', 'scale', 'out')} == r['hashes'] for f, r in zip(fixtures, report['boundaries']))
        with stack.installed():
            stack.graph._validate()
        scope.verify_restored()
        stack.rewrite.verify_restored()
        stack.compact_queries.verify_restored()
        report.update(passed=True, phase='completed', all_frozen_outputs_equal=True, constants_unchanged=True,
            fixture_inputs_unchanged=True, history_unchanged=True, persistent_io_outside_pool=True,
            scopes_restored=True, decoder_resources=scope.resources, decoder_calls=scope.calls,
            public_graph_replays=stack.graph.replays)
except BaseException:
    report.update(passed=False, failure_phase=report['phase'], phase='failed', error=traceback.format_exc())
    raise
finally:
    try:
        for work in workloads:
            work['graph'].reset()
        if scope is not None:
            scope.verify_restored()
        if stack is not None:
            stack.close()
        finalize_sources()
    except BaseException:
        report.update(passed=False, finalization_error=traceback.format_exc())
        raise
    finally:
        save()
print(json.dumps(dict(passed=True, comparisons=report['comparisons'], diagnostic_only=True)), flush=True)
