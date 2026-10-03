"""Calibrated whole C512 FFN screen on accepted repaired/compact32 activations.

Fit six fixed samples: face48/144/242 and temporal car0/5/10. Hold out face96/192
and car1. All16 blocks and nine samples are checked against a CPU INT32 oracle.
Compare intact selected FFNs against 5-kernel split and 4-kernel fused INT8.
This isolated screen does not run a full approximate model or approve quality.

V2 fixes only the final constant guard: registered _previous is temporal state,
initially None, not an immutable model constant. Hash all graph-protected model
constants and check history separately. V1 failure and arithmetic stay frozen.
"""
from layout_crop_validation_env_v1 import *
from contextlib import contextmanager
import statistics, time, traceback

OUT = D / 'results/c512-int8-ffn-screen-v2'
assert not OUT.exists()
compact = receipt(D / 'results/compact-c512-qkv-v1/validation.json',
    '42f72b137c566343ce780c1ce1d7408e102b2f630cde45444ad1b49c16398348')
repair = receipt(D / 'results/int8-ffn-range-repair-v1/validation.json',
    'bd53a2e2b818886e79fed05e836646bdd89814a434aa932ae5bdd4f7944ce2aa')
face = receipt(D / 'results/int8-ffn-face480-finish-v1/validation.json',
    'b9590f1d70607ca0e3a20869f85ee7149af08ca0a9d4c8d005138e4bb1751e24')
paired = receipt(D / 'results/layout-crop-residual256-v1/validation.json',
    'b0d5b7e984ca530d36459b3c20651b3989f37719aae57bcee3bb6eb64e265fe8')
breakdown = receipt(D / 'results/c512-breakdown-v1/validation.json',
    '707ca3af8f6542c6eb9dad5f739906b1719968987202e005052815084d7a1b57')
review_path = D / 'results/int8-ffn-range-repair-v1/user-review-v1.json'
assert sha(review_path) == '8629fdc089e24eeb9b968e65d6a85d80f89268437a488a17d0c11497360ce214'
assert js(review_path)['status'] == 'accepted_face_color_for_range_repair_v1'
failure_paths = {}
for relative, expected in (
    ('c512-int8-ffn-screen-v1/validation.json', '46b46e5e660816ea7740fdb86a028e76bbdfa070d47b3f2cc80d9e890cbcc2d9'),
    ('c512-int8-ffn-screen-v1.log', 'e9608aafb049c9bd6277b42a44708b1edcd7934233cc722ce58f7216a941bfab'),
    ('c512-int8-ffn-screen-v1.log.lease.json', '52bc7bfd831bcc8b2b70fd416b816618a476a7d658651bfaab52d30e94ffa0ac')):
    p = D / 'results' / relative
    assert sha(p) == expected
    sources[str(p)] = expected
    failure_paths[relative] = dict(path=str(p), sha256=expected)
failed_v1 = js(D / 'results/c512-int8-ffn-screen-v1/validation.json')
assert failed_v1['phase'] == 'failed' and not failed_v1['passed']
assert len(failed_v1['cases']) == 144 and len(failed_v1['timing']) == 16
assert 'assert constant_hashes == ' in failed_v1['error'] and not failed_v1.get('finalization_error')
for name in ('c512_int8_ffn_oracle_v1.py', 'c512_int8_ffn_gpu_v1.py', 'c512_int8_ffn_capture_v1.py'):
    p = HERE / name
    assert sha(p) == failed_v1['sources'][str(p)]
    sources[str(p)] = sha(p)

import numpy as np
import torch, triton
from PIL import Image
from compact_c512_qkv_stack_v1 import Stack
from residual_scale_v1 import ResidualScale
from face480_residual_scale_v1 import Face480Scale
from nr_backend.execution import use_arithmetic_backend
from full_body_dataflow_v1 import DetailedDataflow
from c512_int8_ffn_capture_v1 import Recorder, record_ffns, own_inputs, run_seeded, signature
from c512_int8_ffn_gpu_v1 import Segment
import c512_int8_ffn_oracle_v1 as oracle
import compressed_arrays_v1 as arrays

assert Path(triton.__file__).is_relative_to(TOOLCHAIN / 'site')
assert triton.__version__.startswith('3.8.0')
NAMES = [f'{side}512.{i}.ffn' for side in ('encoder', 'decoder') for i in range(8)]
CAL = ['car0', 'car5', 'car10', 'face48', 'face144', 'face242']
HELD = ['car1', 'face96', 'face192']
LABELS = ['selected', 'split', 'fused']
COUNT_KEYS = ('triton_calls', 'standalone_fp8', 'quantization_calls', 'elided_fp8')
inputs = R / 'inputs/flow-full-1920x1080-v3'
manifest_path = inputs / 'manifest.json'
assert sha(manifest_path) == paired['sources'][str(manifest_path)]
manifest = js(manifest_path)
expected_car = {r['frame']: r for r in repair['paired']['runs']['repaired_int8'] if r['round'] == 0}
for p in (Path(__file__), HERE / 'c512_int8_ffn_oracle_v1.py', HERE / 'c512_int8_ffn_gpu_v1.py',
          HERE / 'c512_int8_ffn_capture_v1.py', HERE / 'audit_c512_int8_ffn_screen_v2.py',
          HERE / 'Run-C512Int8FfnScreenV2.cmd', manifest_path, review_path):
    sources[str(p)] = sha(p)

OUT.mkdir()
report = dict(scope=__doc__, passed=False, phase='initializing', sources=sources,
    exact_gate=gate, queue_check=queue_check, candidate_promoted=False, new_quantization=True,
    full_candidate_model_test=False, new_video=False, visual_approval=False, default_unchanged=True,
    calibration_samples=CAL, heldout_samples=HELD, calibration_margin=1.25,
    sample_scope='Frozen baseline activations. Heldout frames are correlated with calibration clips; no unseen-video claim.',
    timing_scope='Isolated complete C512 FFN, entry quantization through exit FP8, plus equal output copies. '
                 '8 independent repeated segments per graph, 7 rounds x20 replays. Excludes calibration, packing, JIT, rest of NR and IO.',
    diagnostic_medians_are_not_additive=True, samples=[], calibration=[], cases=[], timing=[], resources={})
report.update(frozen_failed_v1=failure_paths, arithmetic_unchanged_from_v1=True,
              fix_scope='Model constant/history guard and explicit diagnostic receipts only')
stack = None
graphs = []
fixtures, timed_items = {}, {}
raw = lambda t: t.cpu().numpy().tobytes()
digest = lambda a: hashlib.sha256(a.tobytes()).hexdigest()
tdigest = lambda t: hashlib.sha256(raw(t)).hexdigest()
to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to('xpu')


def save():
    tmp = OUT / 'progress.tmp'
    tmp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(OUT / 'validation.json')


def meta(a):
    return dict(shape=list(a.shape), dtype=a.dtype.str, bytes=a.nbytes, raw_sha256=digest(a))


def immutable_constants():
    # Same exact exclusion as GraphFront._constants, never a prefix/persistent
    # filter: the forty _int8_ffn_buffers remain protected too.
    assert '_previous' in stack.model._buffers
    assert stack.model._buffers['_previous'] is stack.model._previous
    buffers = {name: tensor for name, tensor in stack.model.named_buffers() if name != '_previous'}
    expected_names = [row[0] for row in stack.graph.constants]
    assert '_previous' not in expected_names and len(expected_names) == len(set(expected_names))
    assert set(buffers) == set(expected_names), 'Constant names differ from graph signature'
    return {name: dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                      bytes=t.numel()*t.element_size(), raw_sha256=tdigest(t)) for name, t in buffers.items()}


def constant_difference(before, after):
    return dict(added=sorted(after.keys()-before.keys()), removed=sorted(before.keys()-after.keys()),
                changed=sorted(k for k in before.keys() & after.keys() if before[k] != after[k]))


@contextmanager
def body_scopes():
    with stack.rewrite.layout.installed(), stack.rewrite.vit_layout.installed(), \
            stack.rewrite.post_region.installed(), stack.rewrite.fork.installed(), \
            stack.rewrite.resource_gate():
        yield


def captured(label, low):
    entry = stack.graph.last_entry
    static = {k: None if t is None else t.clone() for k, t in entry.inputs.items()}
    before = {k: None if t is None else raw(t) for k, t in static.items()}
    original_before = {k: None if t is None else raw(t) for k, t in entry.inputs.items()}
    previous, history, seed = stack.model._previous, raw(stack.model._previous), stack.model.next_seed
    options = dict(sigmoid=stack.model.sigmoid, blend_scale=stack.model.blend_scale, return_float32=False)
    with stack.installed(), use_arithmetic_backend('triton'), body_scopes():
        control = DetailedDataflow()
        with control.installed():
            value = stack.rewrite.original(stack.model, **static, **options)
        assert raw(value) == raw(low)
        analysis = DetailedDataflow()
        recorder = Recorder(analysis)
        with analysis.installed(), record_ffns(stack.rewrite.layout, recorder):
            value = stack.rewrite.original(stack.model, **static, **options)
        assert raw(value) == raw(low)
        assert signature(control.events) == signature(analysis.events)
        for trace in (control, analysis):
            counts = {k: trace.rewrite_summary()[k] for k in COUNT_KEYS}
            assert list(counts.values()) == [643, 172, 395, 223]
        assert [s['name'] for s in recorder.stages] == NAMES
        values = {}
        boundaries = []
        for item in recorder.stages:
            assert len(item['args']) == len(item['outputs']) == 1
            x = item['args'][0].cpu().numpy().reshape(144, 512).copy()
            y = item['outputs'][0].cpu().numpy().reshape(144, 512).copy()
            assert np.isfinite(x).all() and np.isfinite(y).all()
            values[item['name']] = (x, y)
            boundaries.append(dict(name=item['name'], input=meta(x), output=meta(y),
                input_provenance=item['input_provenance'], triton_calls=len(signature(item['events'])),
                quantization_calls=item['quantization_calls'], elided_fp8=item['elided_fp8']))
            if label == 'car1':
                args, roots = own_inputs(item)
                assert args[0].is_contiguous() and tuple(args[0].shape) == (12, 12, 512)
                timed_items[item['name']] = (item, args, roots)
                old = next(s for s in breakdown['stages'] if s['name'] == item['name'])
                assert old['inputs'][0]['raw_sha256'] == digest(x)
                assert old['outputs'][0]['raw_sha256'] == digest(y)
        fixtures[label] = values
    assert before == {k: None if t is None else raw(t) for k, t in static.items()}
    assert original_before == {k: None if t is None else raw(t) for k, t in entry.inputs.items()}
    assert stack.model._previous is previous and raw(previous) == history and stack.model.next_seed == seed
    assert raw(entry.output) == raw(low)
    stack.rewrite.verify_restored()
    return dict(eager_matches_frozen_graph=True, physical_sequence_unchanged=True,
                history_seed_and_inputs_unchanged=True, counts=counts, boundaries=boundaries)


def run_frame(scene, index, a, motion_cpu, scaler, reset):
    rgb, motion = to(a), to(motion_cpu)
    with stack.installed(), use_arithmetic_backend('triton') as dispatch:
        canvas, flow = scaler.prepare(rgb, motion)
        low = stack.model(canvas, flow, reset=reset)
        full = scaler.composite(rgb, canvas, low)
        torch.xpu.synchronize()
    expected = expected_car[index] if scene == 'car' else repair['frames'][index]
    expected_low = expected['low_raw_sha256'] if scene == 'car' else expected['low']['raw_sha256']
    assert tdigest(low) == expected_low and tdigest(full) == expected['full_raw_sha256']
    assert stack.model.next_seed == expected['next_seed']
    assert raw(stack.model._previous) == raw(low)
    assert raw(rgb) == a.tobytes() and raw(motion) == motion_cpu.tobytes()
    assert dispatch.get('xpu_graph_replay') == 1
    label = scene + str(index)
    if label in CAL + HELD:
        row = dict(label=label, scene=scene, frame=index, reset=reset, calibration=label in CAL,
            heldout=label in HELD, input_rgb_float_sha256=digest(a), motion=meta(motion_cpu),
            low_raw_sha256=tdigest(low), full_raw_sha256=tdigest(full), next_seed=stack.model.next_seed,
            baseline_reference_byte_equal=True, **captured(label, low))
        report['samples'].append(row)
        save()
        print(json.dumps(dict(phase='baseline_capture', sample=label, blocks=16)), flush=True)


def assert_array(tensor, expected, label):
    actual = tensor.cpu().numpy()
    assert actual.shape == expected.shape and actual.dtype == expected.dtype, (label, 'shape/dtype')
    assert actual.tobytes() == expected.tobytes(), (label, 'CPU/GPU mismatch', int(np.count_nonzero(actual != expected)))


def check_gpu(segment, x, candidate, label):
    b = segment.buffers
    for t in b.values():
        t.fill_(-91 if t.dtype == torch.int8 else float('nan'))
    segment.run('split', True)
    torch.xpu.synchronize()
    for key in b:
        assert_array(b[key], candidate[key], label + '.split.' + key)
    # Fused path has no global QH/half-hidden writes, even in its debug variant.
    b['qh'].fill_(91)
    b['hraw'].fill_(float('nan'))
    untouched = {key: raw(b[key]) for key in ('qh', 'hraw')}
    segment.run('fused', True)
    torch.xpu.synchronize()
    for key in set(b) - set(untouched):
        assert_array(b[key], candidate[key], label + '.fused.' + key)
    assert untouched == {key: raw(b[key]) for key in untouched}
    for variant in ('split', 'fused'):
        debug_keys = ('zraw', 'hraw', 'graw', 'raw')
        for key in debug_keys:
            b[key].fill_(float('nan'))
        untouched = {key: raw(b[key]) for key in debug_keys}
        if variant == 'fused':
            b['qh'].fill_(91)
            untouched['qh'] = raw(b['qh'])
        segment.run(variant)
        torch.xpu.synchronize()
        assert_array(b['out'], candidate['out'], label + '.' + variant + '.nondebug')
        assert_array(b['qg'], candidate['qg'], label + '.' + variant + '.qg')
        assert untouched == {key: raw(b[key]) for key in untouched}


def time_block(name, segment, candidate):
    item, args, roots = timed_items[name]
    original_roots = [raw(r['tensor']) for r in roots]
    reference = fixtures['car1'][name][1].tobytes()
    entries = {}
    row = dict(name=name, samples_ms={label: [] for label in LABELS}, orders=[],
               captured_complete_segments=8, replays_per_round=20, equal_output_copy=True)
    pool, stream = torch.xpu.graph_pool_handle(), torch.xpu.Stream()
    with stack.installed(), use_arithmetic_backend('triton'), body_scopes():
        baseline = lambda: run_seeded(item, args, roots)[0]
        variants = dict(selected=baseline, split=lambda: segment.run('split'), fused=lambda: segment.run('fused'))
        saved = dict(selected=reference, split=candidate['out'].tobytes(), fused=candidate['out'].tobytes())
        for label in LABELS:
            fn = variants[label]
            with stream:
                for _ in range(2):
                    warm = fn()
            torch.xpu.synchronize()
            assert raw(warm) == saved[label]
            public = torch.empty((144, 512), dtype=torch.float16, device='xpu')
            graph = torch.xpu.XPUGraph()
            graphs.append(graph)
            with torch.xpu.graph(graph, stream=stream, pool=pool):
                for _ in range(8):
                    value = fn()
                    public.copy_(value.reshape(144, 512))
            graph.replay()
            torch.xpu.synchronize()
            assert raw(public) == saved[label]
            entries[label] = (graph, public)
        segments = torch.xpu.memory_snapshot(pool)
        persistent = [r['tensor'] for r in roots] + list(segment.buffers.values())
        persistent += [public for _, public in entries.values()]
        assert all(not any(s['address'] <= t.data_ptr() < s['address'] + s['total_size'] for s in segments) for t in persistent)
        for repetition in range(7):
            order = LABELS[repetition % 3:] + LABELS[:repetition % 3]
            if repetition % 2:
                order = order[::-1]
            row['orders'].append(order)
            for label in order:
                graph, public = entries[label]
                torch.xpu.synchronize()
                start = time.perf_counter()
                for _ in range(20):
                    graph.replay()
                torch.xpu.synchronize()
                row['samples_ms'][label].append((time.perf_counter() - start)*1000/(20*8))
                assert raw(public) == saved[label]
        # A changed live input must be consumed by all captured graphs.
        args[0].zero_()
        changed = {}
        for label, (graph, public) in entries.items():
            graph.replay()
            torch.xpu.synchronize()
            eager = variants[label]()
            torch.xpu.synchronize()
            assert raw(public) == raw(eager)
            assert raw(public) != saved[label]
            changed[label] = raw(public)
        assert changed['split'] == changed['fused']
        for root, saved_root in zip(roots, original_roots):
            restored = np.frombuffer(saved_root, dtype=root['tensor'].cpu().numpy().dtype).copy()
            root['tensor'].copy_(to(restored))
        for label, (graph, public) in entries.items():
            graph.replay()
            torch.xpu.synchronize()
            assert raw(public) == saved[label]
            graph.reset()
            graphs.remove(graph)
    assert original_roots == [raw(r['tensor']) for r in roots]
    row['median_ms'] = {label: statistics.median(values) for label, values in row['samples_ms'].items()}
    row['change_percent'] = {label: (row['median_ms'][label]/row['median_ms']['selected'] - 1)*100 for label in ('split', 'fused')}
    row.update(live_input_graph_checks=True, baseline_physical_sequence_and_fp8_elisions_equal=True,
               outputs_match_checked_eager=True, persistent_io_outside_pool=True, operands_unchanged=True)
    return row


save()
try:
    torch.set_num_threads(2)
    scales = [arrays.load(row['hidden_scale']) for row in repair['calibration']['layers']]
    stack = Stack(EXACT, hidden_scales=scales, bm=32)
    sources.update(stack.rewrite.fork.sources)
    report['runtime'] = dict(torch=torch.__version__, triton=triton.__version__, device=torch.xpu.get_device_name())
    assert stack.model._previous is None and stack.model.next_seed == 0
    constant_hashes = immutable_constants()
    report['constant_guard'] = dict(excluded_state=['_previous'],
        graph_signature_names=[row[0] for row in stack.graph.constants], before=constant_hashes,
        initial_history_none=True, initial_history_seed=0)
    save()
    car_scaler, face_scaler = ResidualScale(256), Face480Scale()
    with torch.inference_mode():
        report['phase'] = 'collecting_baseline'; save()
        for i in range(11):
            spec = manifest['frames'][i]
            p, f = inputs / spec['file'], inputs / spec['motion_file']
            for path in (p, f):
                assert sha(path) == paired['sources'][str(path)]
                sources[str(path)] = sha(path)
            a = np.asarray(Image.open(p).convert('RGB')).astype('f4') / 255
            motion = np.fromfile(f, '<f4').reshape(1080, 1920, 2)
            run_frame('car', i, a, motion, car_scaler, bool(spec['reset']))
        for i in (48, 144, 242, 96, 192):
            p = D / 'results/int8-ffn-face480-review-v1' / f'frame{i:03d}-full-and-face.png'
            assert sha(p) == face['images'][str(p)]
            sources[str(p)] = sha(p)
            pixels = np.asarray(Image.open(p).convert('RGB'))[56:536, :864].copy()
            assert digest(pixels) == repair['frames'][i]['input_rgb8_sha256']
            motion = arrays.load(repair['frames'][i]['motion'])
            previous = arrays.load(repair['frames'][i-1]['low'])
            stack.model.reset()
            stack.model._previous = to(previous)
            stack.model._next_seed = i
            run_frame('face', i, pixels.astype('f4') / 255, motion, face_scaler, False)
        assert set(fixtures) == set(CAL + HELD) and set(timed_items) == set(NAMES)
        assert len(stack.graph.entries) == 2 and stack.graph.replays == 16 and stack.rewrite.scopes == 6
        history, hbytes, seed = stack.model._previous, raw(stack.model._previous), stack.model.next_seed
        report['history_guard'] = dict(before=meta(history.cpu().numpy()), seed_before=seed,
            frozen_reference_frame=192, separately_checked_after_baseline=True)
        assert report['history_guard']['before']['raw_sha256'] == repair['frames'][192]['low']['raw_sha256']
        modules = list(stack.model.encoder512) + list(stack.model.decoder512)
        for name, module in zip(NAMES, modules):
            report.update(phase='calibration_and_gpu_screen', active_block=name); save()
            weights = [t.cpu().numpy().copy() for t in (module.ffwd.linear, module.ffwd.expand,
                module.ffwd.reduce, module.ffwd_projection.weight, module.ffwd_projection.skip_scale)]
            packed = oracle.calibrate([fixtures[label][name][0] for label in CAL], weights)
            report['calibration'].append(dict(name=name, fitted_only=CAL,
                scales={key: arrays.save(packed[key]) for key in ('sz', 'sh', 'sg')},
                packed_metadata={key: meta(value) for key, value in packed.items()},
                no_calibration_clipping=True, staged_candidate_activations=True))
            item, args, roots = timed_items[name]
            x_gpu = args[0].reshape(144, 512)
            constant_arrays = oracle.gpu_constants(packed)
            constants = [to(a) for a in constant_arrays]
            constant_bytes = [raw(t) for t in constants]
            segment = Segment(x_gpu, *constants)
            report['resources'][name] = dict(kernels=segment.resources, selections=segment.selections)
            assert len(segment.plans['split', False]) == 5 and len(segment.plans['fused', False]) == 4
            for label in CAL + HELD:
                x, reference = fixtures[label][name]
                candidate = oracle.run(x, packed)
                x_gpu.copy_(to(x))
                check_gpu(segment, x_gpu, candidate, name + '.' + label)
                assert raw(x_gpu) == x.tobytes()
                clip = oracle.clipping(candidate, packed)
                if label in CAL:
                    assert all(v['fraction'] == 0 for v in clip.values())
                report['cases'].append(dict(name=name, sample=label, calibration=label in CAL, heldout=label in HELD,
                    input=meta(x), reference=meta(reference), cpu={key: meta(v) for key, v in candidate.items()},
                    split_all_boundaries_match_cpu=True, fused_boundaries_match_cpu=True,
                    nondebug_matches_cpu=True, fused_no_hidden_writes=True, debug_stores_disabled=True,
                    clipping=clip, error_vs_selected=oracle.error_metrics(candidate['out'], reference)))
            zero = np.zeros((144, 512), dtype='f2')
            x_gpu.zero_()
            check_gpu(segment, x_gpu, oracle.run(zero, packed), name + '.zero')
            assert raw(segment.buffers['sx']) == np.ones(144, dtype='f4').tobytes()
            x_gpu.copy_(to(fixtures['car1'][name][0]))
            candidate = oracle.run(fixtures['car1'][name][0], packed)
            report.update(phase='paired_segment_timing', active_block=name); save()
            timing = time_block(name, segment, candidate)
            timing['zero_entry_cpu_gpu_control'] = True
            report['timing'].append(timing)
            assert [raw(t) for t in constants] == constant_bytes
            print(json.dumps(dict(block=name, median_ms=timing['median_ms'], change_percent=timing['change_percent'])), flush=True)
            save()
        assert len(report['cases']) == 144 and len(report['timing']) == 16
        final_constants = immutable_constants()
        difference = constant_difference(constant_hashes, final_constants)
        report['constant_guard'].update(after=final_constants, difference=difference)
        report['history_guard'].update(after=meta(stack.model._previous.cpu().numpy()),
            seed_after=stack.model.next_seed, same_object=stack.model._previous is history)
        save()
        assert not any(difference.values()), ('Immutable model constants changed', difference)
        assert stack.model._previous is history and raw(history) == hbytes and stack.model.next_seed == seed
        assert report['history_guard']['before'] == report['history_guard']['after']
        assert stack.graph.replays == 16 and stack.rewrite.scopes == 6
        with stack.installed():
            stack.graph._validate()
        stack.rewrite.verify_restored()
        report['baseline_history_seed_constants_and_graphs_unchanged'] = True
        report['graphs'] = stack.graph.metadata()
        report['calibration_constants_unchanged'] = True
        report['median_across_blocks_ms'] = {label: statistics.median(t['median_ms'][label] for t in report['timing']) for label in LABELS}
    report.update(passed=True, phase='completed')
except BaseException:
    report.update(passed=False, phase='failed', error=traceback.format_exc())
    raise
finally:
    try:
        for graph in graphs:
            graph.reset()
        if stack is not None:
            stack.close()
        finalize_sources()
    except BaseException:
        report.update(passed=False, phase='failed', finalization_error=traceback.format_exc())
        raise
    finally:
        save()
print(json.dumps(dict(passed=True, cases=144, timed_blocks=16, median_across_blocks_ms=report['median_across_blocks_ms'],
    full_candidate_model_test=False)), flush=True)
