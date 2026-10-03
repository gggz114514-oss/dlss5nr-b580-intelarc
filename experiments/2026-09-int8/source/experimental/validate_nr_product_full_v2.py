"""Consolidated product Session validation against the accepted floor16 video.

Identical arithmetic/weights/guards. Compare real propagated attention and full
projection boundaries, all shift/padding controls, frozen car/243-face bytes,
and separate paired body/pipeline timings. No new video or raw tensor dumps.
"""
from layout_crop_validation_env_v1 import *
import statistics, subprocess, time, traceback
OUT = D / 'results/nr-product-full-v2'
assert not OUT.exists()
screen = receipt(D/'results/c512-quad-queries-v2/validation.json',
    '546d2ee7b20478fdacaa7a6bde4bc547b02e4dd8b490685669578b405ce5bb89')
screen_handoff = D/'results/c512-quad-queries-v2-monitor-luna-v1/handoff.json'
assert sha(screen_handoff) == '0e4d4f716bc39424de8838338248171da59603aa908ef2422d611b0525698269'
assert js(screen_handoff)['passed'] and js(screen_handoff)['local_screen_only']
sources[str(screen_handoff)] = sha(screen_handoff)
assert not screen['full_model_run'] and not screen['performance_claimed']
full = receipt(D/'results/c512-int8-full-v1/validation.json',
    'd1d2b8e21b1c9e95e7d99b7e0bf5735237781e2412301295e7472aeff92ba7f2')
range_report = receipt(D/'results/c512-int8-range-v1/validation.json',
    '97756e1d6c4af2c7d4cf97120ed0778820b9cf594232167ffe5ea451f018fda4')
repair = receipt(D/'results/int8-ffn-range-repair-v1/validation.json',
    'bd53a2e2b818886e79fed05e836646bdd89814a434aa932ae5bdd4f7944ce2aa')
media = receipt(D/'results/long-precision-480-v1/validation.json',
    '858b4dd646c3be61263bc3fac6fabe400912686af672861638619c63be3be8a3')
paired = receipt(D/'results/layout-crop-residual256-v1/validation.json',
    'b0d5b7e984ca530d36459b3c20651b3989f37719aae57bcee3bb6eb64e265fe8')
review = D/'results/c512-int8-full-v1/user-review-v1.json'
assert sha(review) == '2f923c191f990960ea60ed28347d411057424adc57b38d99b4ac185042c6c9d6'
assert js(review)['status'] == 'accepted_face_quality_for_c512_int8_full_v1'
import numpy as np
import torch, triton
from PIL import Image
from nr256_selected_stack_v4 import Stack as DonorStack
from c512_int8_full_stack_v1 import Stack as BaselineStack
sys.path.insert(0, str(ROOT/'product'))
from nr_runtime_v1 import Session
from nr256_product_stack_v1 import Stack as CandidateStack
CONFIG = Path('D:/Codex-NR-Experiments/nr-b580/product-v1/local-runtime-v1.json')
config = js(CONFIG)
assert config['profile_sha256'] == '8c0994c72c95404caf8116ef0a2ac0580fd8330c8b8a0c0a5f856489b1a9bb21'
for p,h in config['asset_files'].items():
    assert sha(p) == h
    sources[p] = h
sources[str(CONFIG)] = sha(CONFIG)
for p in (ROOT/'product/nr_runtime_v1.py', ROOT/'product/README.md', ROOT/'BODY_OPTIMIZATION_SUMMARY.md', HERE/'nr256_product_stack_v1.py', HERE/'decoder_gather_scope_v1.py', HERE/'decoder_gather_merge_v1.py'):
    sources[str(p)] = sha(p)
product_session = None
import c512_quad_dispatch_v1 as kernels
from serial_graph_workspace_v1 import share_before_capture
from residual_scale_v1 import ResidualScale
from face480_residual_scale_v1 import Face480Scale
from nr_backend.execution import use_arithmetic_backend
from nr_review_video_finalize_v1 import read_exact
import compressed_arrays_v1 as arrays
import capture_body_v1 as body

assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
ROUTES = ('floor16', 'compact')
assert kernels.SCREEN_HASHES == {r['label'].removeprefix('candidate_'): r['hash']
    for r in screen['resources'] if r['label'].startswith('candidate_')}
car_root = R/'inputs/flow-full-1920x1080-v3'
manifest_path = car_root/'manifest.json'
assert sha(manifest_path) == paired['sources'][str(manifest_path)]
manifest = js(manifest_path)
expected = {v['frame']: v for v in full['paired']['runs']['floor16'] if v['round'] == 0}
assert len(expected) == len(manifest['frames']) == 13
source = Path(media['source'])
ffmpeg = Path(next(p for p in media['sources'] if Path(p).name == 'ffmpeg.exe'))
for p in (Path(__file__), HERE/'c512_quad_dispatch_v1.py', HERE/'c512_quad_query_stack_v1.py',
          HERE/'audit_nr_product_full_v2.py', HERE/'Run-NRProductFullV2.cmd',
          HERE/'Run-NRProductFullV2Audit.cmd', HERE/'NR_PRODUCT_FULL_DESIGN_V2.md',
          HERE/'c512_quad_queries_v2.py', review, source, ffmpeg, manifest_path):
    sources[str(p)] = sha(p)
OUT.mkdir()
report = dict(scope=__doc__, passed=False, phase='initializing', sources=sources, exact_gate=gate,
    queue_check=queue_check, candidate_promoted=False, default_unchanged=True,
    arithmetic_changed=False, guards_removed=False, recalibration=False, new_video=False,
    exact_backend_changed=False, nvidia_byte_parity_claimed=False,
    screened_kernels=kernels.SCREEN_HASHES, local_screen_sha256=sha(D/'results/c512-quad-queries-v2/validation.json'),
    body_scope='Actual captured car1 temporal NR256 body,8 rounds x10 replays; separate nonadditive timing.',
    pipeline_scope=full['speed_scope'], reused_approved_video=full['video'],
    body={n: dict(samples_ms=[]) for n in ROUTES}, paired={n: [] for n in ROUTES},
    controls=[], boundaries=[], eager_checks=[], frames=[], saved_raw_tensors=False)
stacks, held, donor, decoder, stderr = {}, {}, None, None, None
raw = lambda t: t.cpu().numpy().tobytes()
digest = lambda a: hashlib.sha256(a.tobytes()).hexdigest()


def save():
    p = OUT/'progress.tmp'
    p.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    p.replace(OUT/'validation.json')


def phase(value):
    report['phase'] = value
    save()
    print(json.dumps(dict(phase=value)), flush=True)


def constants(s):
    values = {n: t for n, t in s.model.named_buffers() if n != '_previous'}
    assert set(values) == {v[0] for v in s.graph.constants}
    return {n: hashlib.sha256(raw(t)).hexdigest() for n, t in values.items()}


def compact_reference(t, inverse, shift):
    a, inv = t.cpu().numpy(), inverse.cpu().numpy()
    assert a.shape == (16, 2, 2, 64, 32)
    hw = a[..., inv, :].reshape(16, 2, 2, 8, 8, 32).transpose(0, 1, 3, 2, 4, 5).reshape(16, 16, 16, 32)
    sy, sx = shift
    crop = hw[:, sy:sy+12, sx:sx+12]
    return np.ascontiguousarray(crop.reshape(16, 3, 4, 3, 4, 32)
        .transpose(0, 1, 3, 2, 4, 5).reshape(16, 9, 16, 32))


def geometry_checks(inverse):
    inv = inverse.cpu().numpy()
    order = [base+y*8+x for base in (0, 4, 32, 36) for y in range(4) for x in range(4)]
    assert inv.tolist() == [order.index(i) for i in range(64)]
    rows = []
    for shift in ((0, 0), (0, 4), (4, 0), (4, 4)):
        tiles, coordinates = kernels.tile_geometry(shift), []
        for g in tiles:
            for row in range(16):
                local = order[g['query_start']+row]
                y, x = g['window']//2*8+local//8-shift[0], g['window']%2*8+local%8-shift[1]
                assert (y, x) == (g['tile']//3*4+row//4, g['tile']%3*4+row%4)
                assert ((y//4)*3+x//4, (y%4)*4+x%4) == (g['tile'], row)
                coordinates.append((y, x))
        assert len(coordinates) == len(set(coordinates)) == 144
        assert set(coordinates) == {(y, x) for y in range(12) for x in range(12)}
        rows.append(dict(shift=list(shift), bm=16, tiles=9,
            valid_pixels=144, coverage_exact=True, geometry=tiles))
    return rows


def run(name, scaler, rgb, motion, reset):
    s = stacks[name]
    scopes, entries, replays = s.rewrite.scopes, len(s.graph.entries), s.graph.replays
    torch.xpu.synchronize()
    started = time.perf_counter_ns()
    if name == 'compact':
        frame = product_session.process(rgb, motion, reset=reset)
        out, low = frame.color, frame.low_color
        assert frame.sequence == s.model.next_seed and frame.reset_applied == bool(reset)
    else:
        with s.installed(), use_arithmetic_backend('triton') as dispatch:
            canvas, flow = scaler.prepare(rgb, motion)
            low = s.model(canvas, flow, reset=reset)
            out = scaler.composite(rgb, canvas, low)
            torch.xpu.synchronize()
        assert dispatch.get('xpu_graph_replay') == 1
    ms = (time.perf_counter_ns()-started)/1e6
    assert s.graph.replays == replays+1
    assert len(s.graph.entries) > entries or s.rewrite.scopes == scopes
    s.rewrite.verify_restored()
    if name == 'compact':
        s.compact_queries.verify_restored()
    return out, low, ms


def control_checks(s):
    scope = s.compact_queries
    module = s.model.encoder512[0]
    inv = module.attention.pixel_inverse
    report['geometry'] = geometry_checks(inv)
    rng = np.random.default_rng(481039)
    shape = (16, 2, 2, 64, 32)
    q, k, v = [rng.choice(np.array([-1., -.5, .25, .5, 1.], dtype='f2'), shape) for _ in range(3)]
    residual = torch.from_numpy(rng.choice(np.array([-.5, 0., .5], dtype='f2'), (12, 12, 512))).to('xpu')
    inverse = inv.cpu().numpy()
    # Run with the same ShortFP8 dependency fork as captured model arithmetic.
    with s.installed(), s.rewrite.fork.installed(), s.rewrite.resource_gate(), use_arithmetic_backend('triton'):
        phase('compile_screened_candidate_kernels')
        sample = [torch.from_numpy(np.ascontiguousarray(a)).to('xpu') for a in (q, k, v)]
        report['preflight'] = kernels.preflight(*sample, module.attention.bias, inv,
            module.projection.weight, residual, module.projection.skip_scale)
        phase('shift_and_padding_controls')
        for shift in ((0, 0), (0, 4), (4, 0), (4, 4)):
            padding = np.ones((2, 2, 64), dtype=bool)
            sy, sx = shift
            for y in range(12):
                for x in range(12):
                    yy, xx = y+sy, x+sx
                    padding[yy//8, xx//8, inverse[(yy%8)*8+xx%8]] = False
            changed_k, changed_v, changed_q = k.copy(), v.copy(), q.copy()
            changed_k[:, padding, :] = 2
            changed_v[:, padding, :] = 4
            changed_q[:, padding, :] = 8
            cases = [('zero', [np.zeros_like(q) for _ in range(3)]),
                     ('distinct', [q, k, v]), ('padded_kv_changed', [q, changed_k, changed_v]),
                     ('ignored_q_changed', [changed_q, k, v])]
            outputs = {}
            for label, values in cases:
                tensors = [torch.from_numpy(np.ascontiguousarray(a)).to('xpu') for a in values]
                reference, _, _ = scope.original_attend(*tensors, module.attention.bias)
                actual, ak, ac = kernels.attend(*tensors, module.attention.bias, inv, shift=shift)
                want = compact_reference(reference, inv, shift)
                assert raw(actual) == want.tobytes(), (shift, label, 'attention')
                outputs[label] = digest(want)
                ref_full, _, _ = scope.original_project(reference, module.projection.weight, residual,
                    module.projection.skip_scale, inv, shift=shift)
                got_full, pk, pc = kernels.project(actual, module.projection.weight, residual, module.projection.skip_scale)
                assert raw(ref_full) == raw(got_full), (shift, label, 'projection')
                assert all(raw(t) == a.tobytes() for t, a in zip(tensors, values))
                report['controls'].append(dict(shift=list(shift), case=label, attention_byte_equal=True,
                    full_projection_byte_equal=True, inputs_unchanged=True, compact_sha256=digest(want),
                    projection_sha256=hashlib.sha256(raw(got_full)).hexdigest(),
                    resources={n: dict(hash=ker.hash, spills=ker.n_spills, selection=sel)
                        for n, ker, sel in [('attention', ak, ac), ('projection', pk, pc)]}))
            assert outputs['distinct'] == outputs['ignored_q_changed']
            assert outputs['distinct'] != outputs['padded_kv_changed'], 'Missing padded K/V influence'
    report['padded_kv_influence_preserved'] = report['ignored_queries_have_no_influence'] = True


def check_body(name, entry, label, live=False, probe=False):
    s = stacks[name]
    prev, history, seed = s.model._previous, raw(s.model._previous), s.model.next_seed
    saved = {k: None if t is None else t.clone() for k, t in entry.inputs.items()}
    old_output = raw(entry.output)
    pending, seen = {}, []
    def boundary(kind, module, operands, result):
        block = s.c512_int8.modules[id(module)]
        before = [raw(t) for t in operands]
        if kind == 'attention':
            reference, _, _ = s.compact_queries.original_attend(*operands)
            want = compact_reference(reference, module.attention.pixel_inverse, module.window_shift)
            assert raw(result) == want.tobytes(), (label, block, 'attention boundary')
            assert block not in pending
            pending[block] = reference
        else:
            assert kind == 'projection'
            _, weight, residual, scale, inverse = operands
            reference, _, _ = s.compact_queries.original_project(pending.pop(block), weight, residual,
                scale, inverse, shift=module.window_shift)
            assert raw(result) == raw(reference), (label, block, 'unquantized projection boundary')
            seen.append(block)
        assert before == [raw(t) for t in operands]
        report['boundaries'].append(dict(label=label, block=block, kind=kind,
            shift=list(module.window_shift), shape=list(result.shape),
            raw_sha256=hashlib.sha256(raw(result)).hexdigest(), byte_equal=True, inputs_unchanged=True))
    if probe:
        assert name == 'compact' and s.compact_queries.probe is None
        s.compact_queries.probe = boundary
    try:
        if live:
            entry.inputs['rgb'].zero_()
            entry.inputs['front'].zero_()
        ib = {k: None if t is None else raw(t) for k, t in entry.inputs.items()}
        with s.installed(), body.installed(), use_arithmetic_backend('triton'):
            value = body.forward_front(s.model, **entry.inputs, sigmoid=s.model.sigmoid,
                blend_scale=s.model.blend_scale, return_float32=False)
            torch.xpu.synchronize()
        eb = raw(value)
        entry.graph.replay()
        torch.xpu.synchronize()
        assert raw(entry.output) == eb
        assert (eb != old_output) if live else (eb == old_output)
        assert ib == {k: None if t is None else raw(t) for k, t in entry.inputs.items()}
        if probe:
            assert not pending and seen == list(s.c512_int8.packed)
    finally:
        if probe:
            s.compact_queries.probe = None
        for k, t in saved.items():
            if t is not None:
                entry.inputs[k].copy_(t)
        entry.graph.replay()
        torch.xpu.synchronize()
    assert raw(entry.output) == old_output
    assert s.model._previous is prev and raw(prev) == history and s.model.next_seed == seed
    report['eager_checks'].append(dict(route=name, label=label, live=live, probed_blocks=len(seen),
        graph_eager_byte_equal=True, live_input_consumed=live, inputs_restored=True, history_unchanged=True))


save()
try:
    torch.set_num_threads(2)
    scales = [arrays.load(v['hidden_scale']) for v in repair['calibration']['layers']]
    donor = DonorStack(EXACT)
    donor_buffers = dict(donor.model.named_buffers(remove_duplicate=False))
    assert len(donor_buffers) == 779
    stacks['floor16'] = BaselineStack(EXACT, hidden_scales=scales,
        calibration=range_report['calibration'], share_with=donor)
    product_session = Session.create(EXACT, config['profile_path'], config['profile_sha256'], share_with=donor)
    stacks['compact'] = product_session._stack
    addresses = set()
    for name, s in stacks.items():
        own = dict(s.model.named_buffers(remove_duplicate=False))
        assert len(own) == len(s.graph.constants) == 1011
        assert all(own[n] is t for n, t in donor_buffers.items())
        extra = {t.data_ptr() for n, t in own.items() if n not in donor_buffers}
        assert len(extra) == 232 and addresses.isdisjoint(extra)
        addresses.update(extra)
        sources.update(s.rewrite.fork.sources)
    report['shared_pool'] = share_before_capture([s.graph for s in stacks.values()])
    report['constant_guard'] = {n: dict(before=constants(s)) for n, s in stacks.items()}
    original_hashes = {n: v['raw_sha256'] for n, v in full['constant_guard']['floor16']['before'].items()}
    assert all(v['before'] == original_hashes for v in report['constant_guard'].values())
    car_scaler, face_scaler = ResidualScale(256), Face480Scale()
    tables = {n: [digest(t.cpu().numpy()) for row in sc.tables.values() for t in row]
              for n, sc in [('car', car_scaler), ('face', face_scaler)]}
    report['runtime'] = dict(torch=torch.__version__, triton=triton.__version__, device=torch.xpu.get_device_name())
    with torch.inference_mode():
        phase('shift_and_padding_controls')
        control_checks(stacks['compact'])
        phase('warmup')
        resident = []
        for spec in manifest['frames']:
            p, f = car_root/spec['file'], car_root/spec['motion_file']
            for path in (p, f):
                assert sha(path) == paired['sources'][str(path)]
                sources[str(path)] = sha(path)
            a = np.asarray(Image.open(p).convert('RGB')).astype('f4')/255
            m = np.fromfile(f, '<f4').reshape(1080, 1920, 2)
            resident.append((a, m, torch.from_numpy(a).to('xpu'), torch.from_numpy(m).to('xpu')))
        for name, s in stacks.items():
            for i in (0, 1):
                _, _, rgb, motion = resident[i]
                out, low, _ = run(name, car_scaler, rgb, motion, i == 0)
                assert (digest(out.cpu().numpy()), digest(low.cpu().numpy())) == (
                    expected[i]['full_raw_sha256'], expected[i]['low_raw_sha256'])
            assert s.rewrite.scopes == len(s.rewrite.builds) == 6 and len(s.graph.entries) == 2
            assert s.c512_int8.ffn_resources == full['candidate']['c512_int8']['resources']
            assert s.c512_int8.ffn_selections == full['candidate']['c512_int8']['selections']
        report['capture_counts'] = {n: [{k: v[k] for k in ('triton_calls', 'standalone_fp8', 'quantization_calls', 'elided_fp8')}
            for v in s.rewrite.builds] for n, s in stacks.items()}
        assert report['capture_counts']['floor16'] == full['capture_build_counts']['floor16']
        assert report['capture_counts']['compact'] == [dict(triton_calls=639, standalone_fp8=136, quantization_calls=311, elided_fp8=175)]*6
        phase('product_api_input_contracts')
        _, _, good_rgb, good_motion = resident[1]
        old_state = stacks['compact'].model._previous, raw(stacks['compact'].model._previous), stacks['compact'].model.next_seed
        report['api_rejections'] = []
        for label, color, motion in [('unsupported_size', good_rgb[:100,:100], good_motion[:100,:100]),
                ('wrong_motion_channels', good_rgb, good_motion[..., :1]), ('not_tensor', None, good_motion)]:
            try:
                product_session.process(color, motion)
            except (TypeError, ValueError):
                report['api_rejections'].append(label)
            else:
                raise AssertionError('API failed to reject '+label)
        assert not product_session._failed
        prev, old_bytes, old_seed = old_state
        assert stacks['compact'].model._previous is prev and raw(prev)==old_bytes and stacks['compact'].model.next_seed==old_seed
        phase('propagated_boundaries_and_live_inputs')
        for name, s in stacks.items():
            for entry in s.graph.entries.values():
                mode = 'temporal' if entry.inputs['previous'] is not None else 'reset'
                check_body(name, entry, mode, probe=name == 'compact')
                check_body(name, entry, 'live_'+mode, live=True)
        phase('paired_body_timing')
        histories = {n: (s.model._previous, raw(s.model._previous), s.model.next_seed) for n, s in stacks.items()}
        body_inputs = {n: {k: None if t is None else raw(t) for k, t in s.graph.last_entry.inputs.items()} for n, s in stacks.items()}
        for repeat in range(8):
            for name in (ROUTES if repeat%2 == 0 else ROUTES[::-1]):
                entry = stacks[name].graph.last_entry
                before = raw(entry.output)
                torch.xpu.synchronize()
                start = time.perf_counter_ns()
                for _ in range(10):
                    entry.graph.replay()
                torch.xpu.synchronize()
                report['body'][name]['samples_ms'].append((time.perf_counter_ns()-start)/1e7)
                assert raw(entry.output) == before
        for name, s in stacks.items():
            report['body'][name]['median_ms'] = statistics.median(report['body'][name]['samples_ms'])
            prev, b, seed = histories[name]
            assert s.model._previous is prev and raw(prev) == b and s.model.next_seed == seed
            assert body_inputs[name] == {k: None if t is None else raw(t) for k, t in s.graph.last_entry.inputs.items()}
            s.model.reset()
        phase('resident_pipeline_timing')
        for repeat in range(4):
            for i, (a, m, rgb, motion) in enumerate(resident):
                order = ROUTES if (repeat+i)%2 == 0 else ROUTES[::-1]
                reset = bool(manifest['frames'][i]['reset'])
                for name in order:
                    s = stacks[name]
                    before_seed = s.model.next_seed
                    before_hash = None if s.model._previous is None else hashlib.sha256(raw(s.model._previous)).hexdigest()
                    other = stacks[ROUTES[1] if name == ROUTES[0] else ROUTES[0]]
                    other_state = other.model._previous, None if other.model._previous is None else raw(other.model._previous), other.model.next_seed
                    out, low, ms = run(name, car_scaler, rgb, motion, reset)
                    ob, lb = raw(out), raw(low)
                    ref = expected[i]
                    assert (hashlib.sha256(ob).hexdigest(), hashlib.sha256(lb).hexdigest()) == (ref['full_raw_sha256'], ref['low_raw_sha256'])
                    assert raw(s.model._previous) == lb and s.model.next_seed == ref['next_seed']
                    assert s.model._previous.data_ptr() != low.data_ptr()
                    out.zero_(); low.zero_()
                    assert raw(s.model._previous) == lb
                    op, oh, other_seed = other_state
                    assert other.model._previous is op and (None if op is None else raw(op)) == oh and other.model.next_seed == other_seed
                    report['paired'][name].append(dict(round=repeat, frame=i, order=list(order), reset=reset, host_ms=ms,
                        full_raw_sha256=ref['full_raw_sha256'], low_raw_sha256=ref['low_raw_sha256'],
                        seed_before=before_seed, previous_low_sha256=before_hash, next_seed=s.model.next_seed,
                        frozen_equal=True, history_private=True, other_history_unchanged=True))
                assert raw(rgb) == a.tobytes() and raw(motion) == m.tobytes()
            save()
            print(json.dumps(dict(pipeline_round=repeat, byte_equal=True)), flush=True)
        report['pipeline'] = {n: {part: dict(mean_ms=statistics.mean(v['host_ms'] for v in rows if part == 'all' or not v['reset']),
            round_mean_ms=[statistics.mean(v['host_ms'] for v in rows if v['round'] == k and (part == 'all' or not v['reset'])) for k in range(4)])
            for part in ('all', 'temporal')} for n, rows in report['paired'].items()}
        for s in stacks.values():
            s.model.reset()
        del resident
        product_session.reset()
        assert product_session._source_size is None and stacks['compact'].model._previous is None and stacks['compact'].model.next_seed == 0
        report['api_reset_verified'] = True
        phase('continuous_face_frozen_byte_validation')
        stderr = (OUT/'face-decode.stderr.txt').open('xb')
        decoder = subprocess.Popen([str(ffmpeg), '-v', 'error', '-threads', '2', '-i', str(source), '-vf', media['decode_filter'],
            '-frames:v', '243', '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
            stdout=subprocess.PIPE, stderr=stderr, creationflags=0x08000000)
        first = None
        for i in range(243):
            pixels = np.frombuffer(read_exact(decoder.stdout, 480*864*3), dtype='u1').reshape(480, 864, 3).copy()
            ref = full['frames'][i]
            assert digest(pixels) == ref['input_rgb8_sha256']
            a = pixels.astype('f4')/255
            stored_motion = arrays.load(ref['motion'])
            assert stored_motion.dtype == np.float16 and np.isfinite(stored_motion).all()
            m = stored_motion.astype('<f4')
            assert m.astype('<f2').tobytes() == stored_motion.tobytes()
            rgb, motion = torch.from_numpy(a).to('xpu'), torch.from_numpy(m).to('xpu')
            row = dict(frame=i, reset=i == 0, input_rgb8_sha256=digest(pixels), motion=ref['motion'], runs={})
            for name in (ROUTES if i%2 == 0 else ROUTES[::-1]):
                s = stacks[name]
                out, low, _ = run(name, face_scaler, rgb, motion, i == 0)
                ob, lb = raw(out), raw(low)
                want = ref['runs']['floor16']
                assert (hashlib.sha256(ob).hexdigest(), hashlib.sha256(lb).hexdigest()) == (want['full_raw_sha256'], want['low_raw_sha256'])
                assert raw(s.model._previous) == lb and s.model.next_seed == i+1
                assert s.model._previous.data_ptr() != low.data_ptr()
                row['runs'][name] = dict(full_raw_sha256=want['full_raw_sha256'], low_raw_sha256=want['low_raw_sha256'],
                    next_seed=i+1, frozen_equal=True, private_history_matches_low=True)
                if i in (0, 120):
                    held[name, i] = out, low, ob, lb
                else:
                    out.zero_(); low.zero_()
                    assert raw(s.model._previous) == lb
            assert raw(rgb) == a.tobytes() and raw(motion) == m.tobytes()
            for out, low, ob, lb in held.values():
                assert raw(out) == ob and raw(low) == lb
            if i == 0:
                first = pixels.copy()
            if i in (96, 192):
                check_body('compact', stacks['compact'].graph.last_entry, 'face'+str(i), probe=True)
            row.update(inputs_unchanged=True, held_outputs_unchanged=True)
            report['frames'].append(row)
            if i%48 == 0 or i == 242:
                save(); print(json.dumps(dict(face_frame=i, byte_equal=True)), flush=True)
        assert decoder.stdout.read(1) == b'' and decoder.wait(timeout=30) == 0
        rgb = torch.from_numpy(first.astype('f4')/255).to('xpu')
        motion = torch.from_numpy(arrays.load(full['frames'][0]['motion']).astype('<f4')).to('xpu')
        for name, s in stacks.items():
            out, low, _ = run(name, face_scaler, rgb, motion, True)
            want = full['frames'][0]['runs']['floor16']
            assert (digest(out.cpu().numpy()), digest(low.cpu().numpy())) == (want['full_raw_sha256'], want['low_raw_sha256'])
            assert raw(s.model._previous) == raw(low) and s.model.next_seed == 1
            assert s.graph.replays == 298 and len(s.graph.entries) == 2
            assert s.rewrite.scopes == (12 if name == 'compact' else 10)
            assert s.c512_int8.ffn_calls == 16*s.rewrite.scopes
            report['constant_guard'][name]['after'] = constants(s)
            assert report['constant_guard'][name]['before'] == report['constant_guard'][name]['after']
            with s.installed():
                s.graph._validate()
            segments = torch.xpu.memory_snapshot(s.graph.capture_pool)
            for _, t in s.model.named_buffers():
                assert not any(v['address'] <= t.data_ptr() < v['address']+v['total_size'] for v in segments)
        candidate = stacks['compact']
        cq = candidate.compact_queries
        cq.verify_restored()
        assert cq.calls == candidate.c512_int8.calls == 192 and set(cq.blocks.values()) == {12}
        assert len(report['boundaries']) == 128 and len(report['eager_checks']) == 10
        report['candidate'] = dict(calls=cq.calls, blocks=cq.blocks, resources=cq.resources,
            permutation_checks=cq.permutations, native_tile_shape=[16, 9, 16, 32],
            selections=cq.selections, constants_added=0, per_block_output_bytes=16*144*32*2,
            old_per_block_output_bytes=16*256*32*2, full_kv_positions=64,
            quantization_unchanged=True, unquantized_pool_input_preserved=True,
            public_boundary_api_unchanged=True, capture_only_layout_rewrite=True)
        assert cq.resources and all(v['spills'] == 0 for v in cq.resources.values())
        for s in stacks.values():
            assert s.c512_int8.ffn_resources == full['candidate']['c512_int8']['resources']
            assert s.c512_int8.ffn_selections == full['candidate']['c512_int8']['selections']
        for n, sc in [('car', car_scaler), ('face', face_scaler)]:
            assert tables[n] == [digest(t.cpu().numpy()) for row in sc.tables.values() for t in row]
        report['product_gather'] = dict(calls=dict(candidate.decoder_gather.calls), resources=dict(candidate.decoder_gather.resources))
        assert set(candidate.decoder_gather.calls.values()) == {12} and len(candidate.decoder_gather.calls)==5
        assert candidate.decoder_gather.resources and all(v['spills']==0 for v in candidate.decoder_gather.resources.values())
        candidate.decoder_gather.verify_restored()
        report['profile'] = dict(id='nr256-reviewed-v1', path=config['profile_path'], sha256=config['profile_sha256'], arrays_relocated_without_change=True)
        assert len(product_session._scalers) == 2
        for dims, scaler in product_session._scalers.items():
            reference_scaler = car_scaler if dims == (1080,1920) else face_scaler
            assert {k: [raw(t) for t in vv] for k,vv in scaler.tables.items()} == {k: [raw(t) for t in vv] for k,vv in reference_scaler.tables.items()}
        report['api_scalers_equal_frozen'] = True
        report['graphs'] = {n: s.graph.metadata() for n, s in stacks.items()}
        assert donor.model._previous is None and donor.model.next_seed == 0 and not donor.graph.entries and donor.graph.replays == 0
        assert dict(donor.model.named_buffers(remove_duplicate=False)).keys() == donor_buffers.keys()
        assert all(dict(donor.model.named_buffers(remove_duplicate=False))[n] is t for n, t in donor_buffers.items())
    with torch.inference_mode():
        phase('product_api_native256_and_close')
        color = torch.linspace(0.,1.,256*256*3,device='xpu',dtype=torch.float32).reshape(256,256,3)
        motion = torch.zeros((256,256,2),device='xpu',dtype=torch.float32)
        try:
            product_session.process(color,motion)
        except ValueError as error:
            assert 'size change' in str(error)
        else:
            raise AssertionError('Source size change did not require reset')
        with stacks['floor16'].installed(), use_arithmetic_backend('triton'):
            reference_low = stacks['floor16'].model(color,motion,reset=True)
        frame = product_session.process(color,motion,reset=True)
        assert raw(frame.low_color)==raw(reference_low) and raw(frame.color)==raw(reference_low.float())
        assert frame.sequence==1 and frame.reset_applied
        product_session.close()
        product_session.close()
        try:
            product_session.process(color,motion,reset=True)
        except RuntimeError as error:
            assert 'closed' in str(error)
        else:
            raise AssertionError('Closed session executed')
        report['api_native256_and_close'] = dict(byte_equal=True, size_change_rejected=True, close_idempotent=True, closed_execution_rejected=True)
    report.update(passed=True, phase='completed', frames_completed=243, all_frozen_outputs_equal=True,
        face_motion_api_dtype='float32', face_motion_roundtrip_byte_equal=True,
        same_cpu_guards=True, inputs_constants_histories_and_pool_verified=True,
        reset_reproduced=True, scopes_restored=True, donor_never_executed=True)
except BaseException:
    report.update(passed=False, failure_phase=report['phase'], phase='failed', error=traceback.format_exc())
    raise
finally:
    try:
        if decoder is not None and decoder.poll() is None:
            decoder.kill(); decoder.wait(timeout=30)
        if stderr is not None:
            stderr.close()
        if product_session is not None:
            product_session.close()
        for name, s in stacks.items():
            if name != 'compact':
                s.close()
        if donor is not None:
            donor.close()
        finalize_sources()
    except BaseException:
        report.update(passed=False, phase='failed', finalization_error=traceback.format_exc())
        raise
    finally:
        save()
print(json.dumps(dict(passed=True, body=report['body'], pipeline=report['pipeline'],
    frames=243, boundary_checks=128, controls=16, all_frozen_outputs_equal=True)), flush=True)
