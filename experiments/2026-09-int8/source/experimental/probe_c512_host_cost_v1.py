"""Locate C512 full-v1 host overhead without weakening or replacing any guard.

Four balanced scenarios: baseline/floor16 times plain/host-traced. Each has its
own evolving history. Frozen outputs must match for every frame and repeat.
Separate idle-device CPU checks are nonadditive diagnostics. No new video.
"""
from layout_crop_validation_env_v1 import *
import statistics, time, traceback
from contextlib import nullcontext
OUT = D / 'results/c512-host-cost-v1'
assert not OUT.exists()
full = receipt(D / 'results/c512-int8-full-v1/validation.json',
    'd1d2b8e21b1c9e95e7d99b7e0bf5735237781e2412301295e7472aeff92ba7f2')
range_report = receipt(D / 'results/c512-int8-range-v1/validation.json',
    '97756e1d6c4af2c7d4cf97120ed0778820b9cf594232167ffe5ea451f018fda4')
repair = receipt(D / 'results/int8-ffn-range-repair-v1/validation.json',
    'bd53a2e2b818886e79fed05e836646bdd89814a434aa932ae5bdd4f7944ce2aa')
paired = receipt(D / 'results/layout-crop-residual256-v1/validation.json',
    'b0d5b7e984ca530d36459b3c20651b3989f37719aae57bcee3bb6eb64e265fe8')
review_path = D / 'results/c512-int8-full-v1/user-review-v1.json'
assert sha(review_path) == '2f923c191f990960ea60ed28347d411057424adc57b38d99b4ac185042c6c9d6'
assert js(review_path)['status'] == 'accepted_face_quality_for_c512_int8_full_v1'
import numpy as np
import torch, triton
from PIL import Image
from nr256_selected_stack_v4 import Stack as DonorStack
from compact_c512_qkv_stack_v1 import Stack as BaselineStack
from c512_int8_full_stack_v1 import Stack as CandidateStack, NAMES as C512_KEYS
from serial_graph_workspace_v1 import share_before_capture
from nr_backend.execution import use_arithmetic_backend
from residual_scale_v1 import ResidualScale
from c512_host_spans_v1 import HostSpans, ANNOTATIONS
import compressed_arrays_v1 as arrays

assert Path(triton.__file__).is_relative_to(TOOLCHAIN / 'site')
ROUTES = ('baseline', 'floor16')
SCENARIOS = ('baseline_plain', 'baseline_traced', 'floor16_plain', 'floor16_traced')
inputs = R / 'inputs/flow-full-1920x1080-v3'
manifest_path = inputs / 'manifest.json'
assert sha(manifest_path) == paired['sources'][str(manifest_path)]
manifest = js(manifest_path)
expected = {name: {v['frame']: v for v in full['paired']['runs'][name] if v['round'] == 0} for name in ROUTES}
for p in (Path(__file__), HERE / 'c512_host_spans_v1.py', HERE / 'audit_c512_host_cost_v1.py',
          HERE / 'Run-C512HostCostV1.cmd', HERE / 'Run-C512HostCostV1Audit.cmd', review_path, manifest_path):
    sources[str(p)] = sha(p)
OUT.mkdir()
report = dict(scope=__doc__, passed=False, phase='initializing', sources=sources, exact_gate=gate,
    queue_check=queue_check, candidate_promoted=False, arithmetic_changed=False,
    guards_removed=False, new_video=False, default_unchanged=True, rows=[], cpu=[],
    trace_scope='Nested host submission/wait wall time, not exclusive GPU kernel duration. Added span overhead is measured against plain.',
    cpu_scope='Idle-device isolated guard/metadata calls; no model execution. Medians cannot be added or subtracted from pipeline spans.',
    pipeline_scope=full['speed_scope'], independent_history_per_scenario=True,
    trace_ast_identity_verified=True, annotations=ANNOTATIONS, extra_gpu_fences_in_trace=0,
    reused_approved_video=full['video'])
stacks, donor = {}, None
raw = lambda t: t.cpu().numpy().tobytes()
digest = lambda a: hashlib.sha256(a.tobytes()).hexdigest()


def save():
    p = OUT / 'progress.tmp'
    p.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    p.replace(OUT / 'validation.json')


def constant_hashes(s):
    tensors = {n: t for n, t in s.model.named_buffers() if n != '_previous'}
    assert set(tensors) == {v[0] for v in s.graph.constants}
    return {n: hashlib.sha256(raw(t)).hexdigest() for n, t in tensors.items()}


def car_pair(i):
    spec = manifest['frames'][i]
    p, f = inputs / spec['file'], inputs / spec['motion_file']
    for path in (p, f):
        assert sha(path) == paired['sources'][str(path)]
        sources[str(path)] = sha(path)
    a = np.asarray(Image.open(p).convert('RGB')).astype('f4') / 255
    m = np.fromfile(f, '<f4').reshape(1080, 1920, 2)
    return a, m, torch.from_numpy(a).to('xpu'), torch.from_numpy(m).to('xpu')


def run(name, rgb, motion, reset, traced=False):
    s = stacks[name]
    scopes, entries, replays = s.rewrite.scopes, len(s.graph.entries), s.graph.replays
    trace = HostSpans(s) if traced else None
    with s.installed(), use_arithmetic_backend('triton') as dispatch:
        with trace.installed() if trace is not None else nullcontext():
            torch.xpu.synchronize()
            if trace is None:
                start = time.perf_counter_ns()
                canvas, flow = scaler.prepare(rgb, motion)
                low = s.model(canvas, flow, reset=reset)
                out = scaler.composite(rgb, canvas, low)
                torch.xpu.synchronize()
                elapsed = (time.perf_counter_ns() - start)/1e6
            else:
                with trace.span('pipeline'):
                    with trace.span('prepare'):
                        canvas, flow = scaler.prepare(rgb, motion)
                    with trace.span('model'):
                        low = s.model(canvas, flow, reset=reset)
                    with trace.span('composite'):
                        out = scaler.composite(rgb, canvas, low)
                    with trace.span('final_wait'):
                        torch.xpu.synchronize()
    assert s.graph.replays == replays + 1 and dispatch.get('xpu_graph_replay') == 1
    assert len(s.graph.entries) > entries or s.rewrite.scopes == scopes
    s.rewrite.verify_restored()
    result = None if trace is None else trace.result()
    if result is not None:
        elapsed = result['host_ms']
    return out, low, elapsed, result


def cpu_checks(s):
    methods = dict(call_guard=s.call_guard.validate, vit_constants=s.int8_vit.validate_constants,
        graph_validate=s.graph._validate, graph_constants=s.graph._constants,
        buffer_enumeration=lambda: tuple(s.model.named_buffers()))
    if hasattr(s, 'c512_int8'):
        methods['c512_constants'] = s.c512_int8.validate_constants
        def registry():
            for row, values in zip(s.model._c512_int8_buffers, s.c512_int8.packed.values()):
                assert all(getattr(row, key) is t for key, t in zip(C512_KEYS, values))
        methods['c512_registry'] = registry
    return methods


save()
try:
    torch.set_num_threads(2)
    scales = [arrays.load(v['hidden_scale']) for v in repair['calibration']['layers']]
    donor = DonorStack(EXACT)
    donor_buffers = dict(donor.model.named_buffers(remove_duplicate=False))
    assert len(donor_buffers) == 779
    stacks['baseline'] = BaselineStack(EXACT, hidden_scales=scales, share_with=donor, bm=32)
    stacks['floor16'] = CandidateStack(EXACT, hidden_scales=scales,
        calibration=range_report['calibration'], share_with=donor)
    for name, s in stacks.items():
        own = dict(s.model.named_buffers(remove_duplicate=False))
        assert all(own[n] is t for n, t in donor_buffers.items())
        assert len(own) == (819 if name == 'baseline' else 1011)
        sources.update(s.rewrite.fork.sources)
    report['shared_pool'] = share_before_capture([s.graph for s in stacks.values()])
    report['constant_guard'] = {n: dict(before=constant_hashes(s)) for n, s in stacks.items()}
    scaler = ResidualScale(256)
    tables = [digest(t.cpu().numpy()) for row in scaler.tables.values() for t in row]
    report['runtime'] = dict(torch=torch.__version__, triton=triton.__version__, device=torch.xpu.get_device_name())
    with torch.inference_mode():
        resident = [car_pair(i) for i in range(13)]
        report['phase'] = 'warmup'
        save()
        for name, s in stacks.items():
            for i in (0, 1):
                _, _, rgb, motion = resident[i]
                out, low, _, _ = run(name, rgb, motion, i == 0)
                assert (digest(out.cpu().numpy()), digest(low.cpu().numpy())) == (
                    expected[name][i]['full_raw_sha256'], expected[name][i]['low_raw_sha256'])
            assert len(s.graph.entries) == 2 and s.rewrite.scopes == len(s.rewrite.builds) == 6
            assert len(s.warp.entries) == 1
        report['phase'] = 'plain_and_instrumented_pipeline'
        save()
        for repeat in range(4):
            states = {case: (None, 0) for case in SCENARIOS}
            for i, (a, m, rgb, motion) in enumerate(resident):
                shift = (repeat + i) % 4
                order = SCENARIOS[shift:] + SCENARIOS[:shift]
                for case in order:
                    name, mode = case.rsplit('_', 1)
                    s = stacks[name]
                    others = {k: None if v[0] is None else raw(v[0]) for k, v in states.items() if k != case}
                    previous, seed = states[case]
                    before_hash = None if previous is None else hashlib.sha256(raw(previous)).hexdigest()
                    s.model._previous, s.model._next_seed = previous, seed
                    reset = bool(manifest['frames'][i]['reset'])
                    out, low, elapsed, trace = run(name, rgb, motion, reset, mode == 'traced')
                    ob, lb = raw(out), raw(low)
                    ref = expected[name][i]
                    assert hashlib.sha256(ob).hexdigest() == ref['full_raw_sha256']
                    assert hashlib.sha256(lb).hexdigest() == ref['low_raw_sha256']
                    assert raw(s.model._previous) == lb and s.model.next_seed == ref['next_seed']
                    assert s.model._previous.data_ptr() != low.data_ptr()
                    states[case] = s.model._previous, s.model.next_seed
                    out.zero_()
                    low.zero_()
                    assert raw(s.model._previous) == lb
                    for key, old in others.items():
                        assert (None if states[key][0] is None else raw(states[key][0])) == old
                    assert raw(rgb) == a.tobytes() and raw(motion) == m.tobytes()
                    row = dict(round=repeat, frame=i, scenario=case, route=name, mode=mode,
                        reset=reset, order=list(order), host_ms=elapsed,
                        full_raw_sha256=ref['full_raw_sha256'], low_raw_sha256=ref['low_raw_sha256'],
                        previous_low_sha256=before_hash, seed_before=seed, next_seed=s.model.next_seed,
                        frozen_output_equal=True, private_history_matches_low=True,
                        other_histories_unchanged=True, inputs_unchanged=True, trace=trace)
                    report['rows'].append(row)
            save()
            print(json.dumps(dict(pipeline_round=repeat,scenarios=4,all_frozen_equal=True)), flush=True)
        assert len(report['rows']) == 208
        report['pipeline'] = {case: {mode: dict(
            mean_ms=statistics.mean(v['host_ms'] for v in report['rows'] if v['scenario'] == case and (mode == 'all' or not v['reset'])),
            round_mean_ms=[statistics.mean(v['host_ms'] for v in report['rows'] if v['scenario'] == case and v['round'] == k and
                (mode == 'all' or not v['reset'])) for k in range(4)]) for mode in ('all', 'temporal')} for case in SCENARIOS}
        report['span_summary'] = {}
        for name in ROUTES:
            groups = {}
            for row in report['rows']:
                if row['route'] != name or row['mode'] != 'traced' or row['reset']:
                    continue
                for span in row['trace']['spans']:
                    groups.setdefault(span['name'], []).append(span)
            report['span_summary'][name] = {key: dict(count=len(values),
                mean_inclusive_ms=statistics.mean(v['inclusive_ns'] for v in values)/1e6,
                mean_exclusive_ms=statistics.mean(v['exclusive_ns'] for v in values)/1e6)
                for key, values in groups.items()}
        report['instrumentation_overhead'] = {name: {mode:
            report['pipeline'][name+'_traced'][mode]['mean_ms'] - report['pipeline'][name+'_plain'][mode]['mean_ms']
            for mode in ('all', 'temporal')} for name in ROUTES}
        report['phase'] = 'isolated_idle_cpu_checks'
        save()
        for repeat in range(8):
            for name in (ROUTES if repeat % 2 == 0 else ROUTES[::-1]):
                s = stacks[name]
                previous, history, seed = s.model._previous, raw(s.model._previous), s.model.next_seed
                replays = s.graph.replays, s.warp.replays, s.rewrite.scopes
                with s.installed(), use_arithmetic_backend('triton') as dispatch:
                    torch.xpu.synchronize()
                    methods = cpu_checks(s)
                    keys = list(methods)
                    shift = repeat % len(keys)
                    order = keys[shift:] + keys[:shift]
                    for label in order:
                        fn = methods[label]
                        fn()
                        start = time.perf_counter_ns()
                        for _ in range(32):
                            result = fn()
                        elapsed = (time.perf_counter_ns() - start)/32e6
                        if label == 'graph_constants':
                            assert result == s.graph.constants
                        elif label == 'buffer_enumeration':
                            assert len(result) == (820 if name == 'baseline' else 1012)
                        else:
                            assert result is None
                        report['cpu'].append(dict(round=repeat, route=name, label=label,
                            iterations=32, per_call_ms=elapsed, no_model_execution=True))
                    assert dispatch == dict(backend='triton', dense=0, batched=0)
                assert (s.graph.replays, s.warp.replays, s.rewrite.scopes) == replays
                assert s.model._previous is previous and raw(previous) == history and s.model.next_seed == seed
            save()
        report['cpu_summary'] = {name: {label: dict(samples_ms=[v['per_call_ms'] for v in report['cpu'] if v['route'] == name and v['label'] == label],
            median_ms=statistics.median(v['per_call_ms'] for v in report['cpu'] if v['route'] == name and v['label'] == label))
            for label in cpu_checks(s)} for name, s in stacks.items()}
        report['cpu_preserved_history_and_replay_counts'] = True
        for name, s in stacks.items():
            _, _, rgb, motion = resident[0]
            out, low, _, _ = run(name, rgb, motion, True)
            assert (digest(out.cpu().numpy()), digest(low.cpu().numpy())) == (
                expected[name][0]['full_raw_sha256'], expected[name][0]['low_raw_sha256'])
            assert s.graph.replays == 107 and len(s.graph.entries) == 2 and s.rewrite.scopes == 6
            report['constant_guard'][name]['after'] = constant_hashes(s)
            assert report['constant_guard'][name]['before'] == report['constant_guard'][name]['after']
            with s.installed():
                s.graph._validate()
            assert '_constants' not in s.graph.__dict__ and '_validate' not in s.graph.__dict__
            assert '_complete' not in s.graph.__dict__ and 'validate' not in s.call_guard.__dict__
            assert 'validate_constants' not in s.int8_vit.__dict__
        candidate = stacks['floor16']
        assert 'validate_constants' not in candidate.c512_int8.__dict__
        assert candidate.c512_int8.ffn_calls == 96
        assert candidate.c512_int8.ffn_resources == full['candidate']['c512_int8']['resources']
        assert candidate.c512_int8.ffn_selections == full['candidate']['c512_int8']['selections']
        assert tables == [digest(t.cpu().numpy()) for row in scaler.tables.values() for t in row]
        report['graphs'] = {n: s.graph.metadata() for n, s in stacks.items()}
        report['capture_counts'] = {n: [{k: v[k] for k in ('triton_calls', 'standalone_fp8', 'quantization_calls', 'elided_fp8')}
            for v in s.rewrite.builds] for n, s in stacks.items()}
        assert report['capture_counts'] == full['capture_build_counts']
        assert donor.model._previous is None and donor.model.next_seed == 0 and donor.graph.replays == 0
        assert not donor.graph.entries
    report.update(passed=True, phase='completed', all_plain_and_traced_outputs_frozen_equal=True,
        trace_bindings_restored=True, steady_no_recapture=True, guards_constants_histories_preserved=True)
except BaseException:
    report.update(passed=False, phase='failed', error=traceback.format_exc())
    raise
finally:
    try:
        for s in stacks.values():
            s.close()
        if donor is not None:
            donor.close()
        finalize_sources()
    except BaseException:
        report.update(passed=False, phase='failed', finalization_error=traceback.format_exc())
        raise
    finally:
        save()
print(json.dumps(dict(passed=True, pipeline_rows=208, cpu_cases=len(report['cpu']),
    pipeline=report['pipeline'], cpu_summary=report['cpu_summary'], no_arithmetic_change=True)), flush=True)
