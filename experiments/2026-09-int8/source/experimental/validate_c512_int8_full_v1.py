"""Frozen floor16 full NR256: propagated CPU checks, paired timing, face video.

Compare against accepted CompactStack BM32 with repaired ViT INT8 and FP16 C512.
Use existing motion inputs. Decode/upload/flow/encoding are outside GPU timing.
All video frames run through their own recurrent histories, never frozen teacher
histories. No new calibration, weights, full frame or hidden dumps are saved.
"""
from layout_crop_validation_env_v1 import *
import statistics, subprocess, time, traceback
OUT = D / 'results/c512-int8-full-v1'
assert not OUT.exists()
range_report = receipt(D / 'results/c512-int8-range-v1/validation.json',
    '97756e1d6c4af2c7d4cf97120ed0778820b9cf594232167ffe5ea451f018fda4')
compact = receipt(D / 'results/compact-c512-qkv-v1/validation.json',
    '42f72b137c566343ce780c1ce1d7408e102b2f630cde45444ad1b49c16398348')
repair = receipt(D / 'results/int8-ffn-range-repair-v1/validation.json',
    'bd53a2e2b818886e79fed05e836646bdd89814a434aa932ae5bdd4f7944ce2aa')
full = receipt(D / 'results/long-precision-480-v1/validation.json',
    '858b4dd646c3be61263bc3fac6fabe400912686af672861638619c63be3be8a3')
paired = receipt(D / 'results/layout-crop-residual256-v1/validation.json',
    'b0d5b7e984ca530d36459b3c20651b3989f37719aae57bcee3bb6eb64e265fe8')
review_path = D / 'results/int8-ffn-range-repair-v1/user-review-v1.json'
assert sha(review_path) == '8629fdc089e24eeb9b968e65d6a85d80f89268437a488a17d0c11497360ce214'
assert js(review_path)['status'] == 'accepted_face_color_for_range_repair_v1'
import numpy as np
import torch, triton, cv2, imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from nr256_selected_stack_v4 import Stack as DonorStack
from compact_c512_qkv_stack_v1 import Stack as BaselineStack
from c512_int8_full_stack_v1 import Stack as CandidateStack, reconstruct, metadata
from serial_graph_workspace_v1 import share_before_capture
from nr_backend.execution import use_arithmetic_backend
from residual_scale_v1 import ResidualScale
from face480_residual_scale_v1 import Face480Scale
from nr_review_video_finalize_v1 import read_exact, finalize
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
import c512_int8_ffn_oracle_v1 as oracle

assert Path(triton.__file__).is_relative_to(TOOLCHAIN / 'site')
ROUTES = ('baseline', 'floor16')
car_inputs = R / 'inputs/flow-full-1920x1080-v3'
manifest_path = car_inputs / 'manifest.json'
assert sha(manifest_path) == paired['sources'][str(manifest_path)]
manifest = js(manifest_path)
expected = {r['frame']: r for r in repair['paired']['runs']['repaired_int8'] if r['round'] == 0}
assert len(expected) == len(manifest['frames']) == 13
source = Path(full['source'])
ffmpeg = Path(next(p for p in full['sources'] if Path(p).name == 'ffmpeg.exe'))
encoder = Path(imageio_ffmpeg.get_ffmpeg_exe())
font_path = Path('C:/Windows/Fonts/msyh.ttc')
for p in (Path(__file__), HERE / 'c512_int8_full_stack_v1.py', HERE / 'audit_c512_int8_full_v1.py',
          HERE / 'Run-C512Int8FullV1.cmd', HERE / 'Run-C512Int8FullV1Audit.cmd',
          HERE / 'nr_review_video_finalize_v1.py', review_path, manifest_path, source,
          ffmpeg, ffmpeg.with_name('ffprobe.exe'), encoder, font_path):
    sources[str(p)] = sha(p)
OUT.mkdir()
report = dict(scope=__doc__, passed=False, phase='initializing', sources=sources, exact_gate=gate,
    queue_check=queue_check, candidate_promoted=False, new_quality_approved=False, human_review='pending',
    default_unchanged=True, recalibration=False, arithmetic_kernels_unchanged=True,
    range_policy='frozen_floor16_split', frames=[], images={}, cpu_checks=[], eager_graph_checks=[],
    body={n: dict(samples_ms=[]) for n in ROUTES}, paired=dict(runs={n: [] for n in ROUTES}),
    body_scope='Captured car1 temporal body only; no front/warp/history commit/scaler. Nonadditive diagnostic.',
    speed_scope='Resident1080p prepare+NR256+private history+residual+host completion; excludes decode,flow,upload,encode,JIT.',
    roi_scope='Fixed384x384 display crop includes background; not skin segmentation.',
    temporal_scope='In-bounds DIS-warped change in error vs accepted baseline, no occlusion ground truth.',
    source=str(source), source_geometry=[864, 480], display_geometry=[2592, 1352],
    panel_order=['input', 'baseline', 'floor16'], own_history_for_every_frame=True,
    saved_raw_frames_or_weights=False)
stacks, held, processes, files = {}, {}, [], []
donor = None
digest = lambda a: hashlib.sha256(a.tobytes()).hexdigest()
raw = lambda t: t.cpu().numpy().tobytes()


def save():
    p = OUT / 'progress.tmp'
    p.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    p.replace(OUT / 'validation.json')


def child(label, command, **options):
    f = (OUT / (label + '.stderr.txt')).open('xb')
    files.append(f)
    p = subprocess.Popen(command, stderr=f, creationflags=0x08000000, **options)
    processes.append(p)
    return p


def constants(s):
    values = {n: t for n, t in s.model.named_buffers() if n != '_previous'}
    assert set(values) == {v[0] for v in s.graph.constants}
    return {n: dict(shape=list(t.shape), stride=list(t.stride()), dtype=str(t.dtype),
                    bytes=t.numel()*t.element_size(), raw_sha256=hashlib.sha256(raw(t)).hexdigest())
            for n, t in values.items()}


def metrics(a, b, roi=False):
    if roi:
        a, b = a[16:400, 256:640], b[16:400, 256:640]
    delta = a.astype('f8') - b.astype('f8')
    return dict(rgb_rmse_8bit=float(np.sqrt(np.mean(delta*delta))*255),
        mae_8bit=float(np.abs(delta).mean()*255), mean_delta_RGB_8bit=(delta.mean((0, 1))*255).tolist())


def car_pair(i):
    spec = manifest['frames'][i]
    p, f = car_inputs / spec['file'], car_inputs / spec['motion_file']
    for path in (p, f):
        assert sha(path) == paired['sources'][str(path)]
        sources[str(path)] = sha(path)
    a = np.asarray(Image.open(p).convert('RGB')).astype('f4') / 255
    m = np.fromfile(f, '<f4').reshape(1080, 1920, 2)
    return a, m, torch.from_numpy(a).to('xpu'), torch.from_numpy(m).to('xpu')


def run(name, scaler, rgb, motion, reset):
    s = stacks[name]
    before, entries = s.rewrite.scopes, len(s.graph.entries)
    with s.installed(), use_arithmetic_backend('triton') as dispatch:
        torch.xpu.synchronize()
        start = time.perf_counter()
        canvas, flow = scaler.prepare(rgb, motion)
        low = s.model(canvas, flow, reset=reset)
        out = scaler.composite(rgb, canvas, low)
        torch.xpu.synchronize()
        ms = (time.perf_counter() - start)*1000
    assert dispatch.get('xpu_graph_replay') == 1
    assert len(s.graph.entries) > entries or s.rewrite.scopes == before
    s.rewrite.verify_restored()
    return out, low, ms


def check_body(name, entry, label, cpu=False, live=False):
    s = stacks[name]
    previous, history, seed = s.model._previous, raw(s.model._previous), s.model.next_seed
    saved = {k: None if t is None else t.clone() for k, t in entry.inputs.items()}
    original_output = raw(entry.output)
    if live:
        entry.inputs['rgb'].zero_()
        entry.inputs['front'].zero_()
    inputs_before = {k: None if t is None else raw(t) for k, t in entry.inputs.items()}
    seen = []
    def probe(block, x, buffers):
        original = raw(x)
        detail = oracle.run(x.cpu().numpy(), cpu_packed[block])
        observed = {k: metadata(v.cpu().numpy()) for k, v in buffers.items()}
        cpu_meta = {k: metadata(detail[k]) for k in buffers}
        assert observed == cpu_meta, (label, block, 'propagated CPU/GPU mismatch')
        assert raw(x) == original
        report['cpu_checks'].append(dict(label=label, block=block, input=metadata(x.cpu().numpy()),
            buffers=observed, cpu=cpu_meta, all_equal=True, input_unchanged=True,
            clipping=oracle.clipping(detail, cpu_packed[block])))
        seen.append(block)
    if cpu:
        assert name == 'floor16' and s.c512_int8.ffn_probe is None
        s.c512_int8.ffn_probe = probe
    try:
        with s.installed(), body.installed(), use_arithmetic_backend('triton'):
            eager = body.forward_front(s.model, **entry.inputs, sigmoid=s.model.sigmoid,
                blend_scale=s.model.blend_scale, return_float32=False)
            torch.xpu.synchronize()
        reference_bytes = raw(eager)
        assert np.isfinite(eager.cpu().numpy()).all()
        entry.graph.replay()
        torch.xpu.synchronize()
        assert raw(entry.output) == reference_bytes
        if live:
            assert reference_bytes != original_output, 'Graph did not consume changed inputs'
        else:
            assert reference_bytes == original_output
        assert inputs_before == {k: None if t is None else raw(t) for k, t in entry.inputs.items()}
        if cpu:
            assert seen == list(cpu_packed)
    finally:
        if cpu:
            s.c512_int8.ffn_probe = None
        for k, t in saved.items():
            if t is not None:
                entry.inputs[k].copy_(t)
        entry.graph.replay()
        torch.xpu.synchronize()
    assert raw(entry.output) == original_output
    assert s.model._previous is previous and raw(previous) == history and s.model.next_seed == seed
    report['eager_graph_checks'].append(dict(route=name, label=label, byte_equal=True,
        live_input_changed=live, live_input_consumed=live, inputs_restored=True,
        private_history_and_seed_unchanged=True, cpu_blocks=len(seen)))


save()
try:
    torch.set_num_threads(2)
    scales = [arrays.load(row['hidden_scale']) for row in repair['calibration']['layers']]
    donor = DonorStack(EXACT)
    donor_buffers = dict(donor.model.named_buffers(remove_duplicate=False))
    assert len(donor_buffers) == 779 and donor.model._previous is None
    stacks['baseline'] = BaselineStack(EXACT, hidden_scales=scales, share_with=donor, bm=32)
    stacks['floor16'] = CandidateStack(EXACT, hidden_scales=scales,
        calibration=range_report['calibration'], share_with=donor)
    candidate = stacks['floor16']
    cpu_packed = {row['name'].removesuffix('.ffn'): reconstruct(
        dict(candidate.model.named_modules())[row['name'].removesuffix('.ffn')], row['variants']['floor16'])
        for row in range_report['calibration']}
    pointers = set()
    report['sharing'] = {}
    for name, s in stacks.items():
        own = dict(s.model.named_buffers(remove_duplicate=False))
        assert all(own[n] is t for n, t in donor_buffers.items())
        extras = {n: t for n, t in own.items() if n not in donor_buffers}
        assert len(extras) == (40 if name == 'baseline' else 232)
        addresses = {t.data_ptr() for t in extras.values()}
        assert len(addresses) == len(extras) and pointers.isdisjoint(addresses)
        pointers.update(addresses)
        report['sharing'][name] = dict(base_buffers=779, extras=len(extras),
            base_objects_shared=True, packed_constants_disjoint=True)
    report['shared_transient_pool'] = share_before_capture([s.graph for s in stacks.values()])
    report['constant_guard'] = {n: dict(before=constants(s)) for n, s in stacks.items()}
    report['reconstructed_pack_metadata'] = candidate.c512_int8.packed_metadata
    report['runtime'] = dict(torch=torch.__version__, triton=triton.__version__, device=torch.xpu.get_device_name())
    car_scaler, face_scaler = ResidualScale(256), Face480Scale()
    tables = {n: [digest(t.cpu().numpy()) for row in sc.tables.values() for t in row]
              for n, sc in [('car', car_scaler), ('face', face_scaler)]}
    for s in stacks.values():
        sources.update(s.rewrite.fork.sources)
    with torch.inference_mode():
        resident = [car_pair(i) for i in range(13)]
        report['phase'] = 'warmup'
        save()
        for name, s in stacks.items():
            for i in (0, 1):
                _, _, rgb, motion = resident[i]
                out, low, _ = run(name, car_scaler, rgb, motion, i == 0)
                assert np.isfinite(out.cpu().numpy()).all() and np.isfinite(low.cpu().numpy()).all()
                if name == 'baseline':
                    assert digest(out.cpu().numpy()) == expected[i]['full_raw_sha256']
                    assert digest(low.cpu().numpy()) == expected[i]['low_raw_sha256']
            assert s.rewrite.scopes == len(s.rewrite.builds) == 6 and len(s.graph.entries) == 2
            print(json.dumps(dict(warmup=name,completed=True)), flush=True)
        assert candidate.c512_int8.ffn_calls == 96
        frozen_resources = range_report['resources'][range_report['calibration'][0]['name']]['floor16']
        for key, value in candidate.c512_int8.ffn_resources.items():
            assert value == frozen_resources['kernels'][key], ('kernel changed', key)
            assert candidate.c512_int8.ffn_selections[key] == frozen_resources['selections'][key]
        for entry in candidate.graph.entries.values():
            assert entry.dispatch['c512_int8_ffn_segment'] == 16
        report['capture_build_counts'] = {n: [{k: v[k] for k in
            ('triton_calls', 'standalone_fp8', 'quantization_calls', 'elided_fp8')} for v in s.rewrite.builds]
            for n, s in stacks.items()}
        for row in report['capture_build_counts']['baseline']:
            assert list(row.values()) == [643, 172, 395, 223]
        report['phase'] = 'propagated_and_live_graph_checks'
        save()
        for name, s in stacks.items():
            for entry in s.graph.entries.values():
                mode = 'temporal' if entry.inputs['previous'] is not None else 'reset'
                check_body(name, entry, 'car1_' + mode, cpu=name == 'floor16')
                check_body(name, entry, 'live_' + mode, live=True)
        assert len(report['cpu_checks']) == 32
        report['phase'] = 'paired_body_timing'
        save()
        histories = {n: (s.model._previous, raw(s.model._previous), s.model.next_seed) for n, s in stacks.items()}
        body_inputs = {n: {k: None if t is None else raw(t) for k, t in s.graph.last_entry.inputs.items()}
                       for n, s in stacks.items()}
        for repeat in range(8):
            order = ROUTES if repeat % 2 == 0 else ROUTES[::-1]
            for name in order:
                entry = stacks[name].graph.last_entry
                before = raw(entry.output)
                torch.xpu.synchronize()
                start = time.perf_counter()
                for _ in range(10):
                    entry.graph.replay()
                torch.xpu.synchronize()
                report['body'][name]['samples_ms'].append((time.perf_counter() - start)*100)
                assert raw(entry.output) == before
        for name, s in stacks.items():
            report['body'][name]['median_ms'] = statistics.median(report['body'][name]['samples_ms'])
            previous, history, seed = histories[name]
            assert s.model._previous is previous and raw(previous) == history and s.model.next_seed == seed
            assert body_inputs[name] == {k: None if t is None else raw(t) for k, t in s.graph.last_entry.inputs.items()}
            s.model.reset()
        report['body_inputs_and_private_history_unchanged'] = True
        report['phase'] = 'resident_pipeline_timing'
        save()
        repeats = {}
        for repeat in range(4):
            for i, (a, m, rgb, motion) in enumerate(resident):
                order = ROUTES if (repeat + i) % 2 == 0 else ROUTES[::-1]
                reset = bool(manifest['frames'][i]['reset'])
                values = {}
                for name in order:
                    s = stacks[name]
                    out, low, ms = run(name, car_scaler, rgb, motion, reset)
                    o, b = out.cpu().numpy(), low.cpu().numpy()
                    assert np.isfinite(o).all() and np.isfinite(b).all()
                    pair = (digest(o), digest(b))
                    key = name, i
                    if repeat == 0:
                        repeats[key] = pair
                    else:
                        assert repeats[key] == pair
                    if name == 'baseline':
                        assert pair == (expected[i]['full_raw_sha256'], expected[i]['low_raw_sha256'])
                    assert raw(s.model._previous) == b.tobytes() and s.model.next_seed == expected[i]['next_seed']
                    report['paired']['runs'][name].append(dict(round=repeat, frame=i, reset=reset,
                        order=list(order), host_ms=ms, full_raw_sha256=pair[0], low_raw_sha256=pair[1],
                        next_seed=s.model.next_seed, own_repeat_equal=True, private_history_matches_low=True))
                    values[name] = o.copy()
                    out.zero_()
                    low.zero_()
                    assert raw(s.model._previous) == b.tobytes()
                if repeat == 0:
                    report['paired'].setdefault('quality', []).append(dict(frame=i,
                        full=metrics(values['floor16'], values['baseline'])))
                assert raw(rgb) == a.tobytes() and raw(motion) == m.tobytes()
            save()
            print(json.dumps(dict(paired_round=repeat,completed=True)), flush=True)
        report['paired']['summary'] = {n: {mode: dict(
            mean_host_ms=statistics.mean(v['host_ms'] for v in rows if mode == 'all' or not v['reset']),
            round_mean_host_ms=[statistics.mean(v['host_ms'] for v in rows if v['round'] == k and
                (mode == 'all' or not v['reset'])) for k in range(4)]) for mode in ('all', 'temporal')}
            for n, rows in report['paired']['runs'].items()}
        report['paired']['passed'] = True
        report['steady_scope_counts'] = {n: s.rewrite.scopes for n, s in stacks.items()}
        for s in stacks.values():
            s.model.reset()
        del resident
        report['phase'] = 'full_face_video'
        save()
        decoder = child('decode', [str(ffmpeg), '-v', 'error', '-threads', '2', '-i', str(source),
            '-vf', full['decode_filter'], '-frames:v', '243', '-fps_mode', 'passthrough',
            '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'], stdout=subprocess.PIPE)
        encoded = OUT / 'encoded-before-finalization.mp4'
        writer = child('encode', [str(encoder), '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
            '-s', '2592x1352', '-r', '24', '-i', '-', '-an',
            '-vf', 'scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p',
            '-c:v', 'libx264', '-threads', '4', '-preset', 'fast', '-crf', '12',
            '-x264-params', 'colorprim=bt709:transfer=bt709:colormatrix=bt709:fullrange=off',
            '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
            '-color_range', 'tv', '-movflags', '+faststart+write_colr', '-n', str(encoded)], stdin=subprocess.PIPE)
        font = ImageFont.truetype(str(font_path), 25)
        small = ImageFont.truetype(str(font_path), 18)
        yy, xx = np.indices((480, 864), dtype='f4')
        previous_error = None
        first = None
        for i in range(243):
            pixels = np.frombuffer(read_exact(decoder.stdout, 480*864*3), dtype='u1').reshape(480, 864, 3).copy()
            ref = repair['frames'][i]
            assert digest(pixels) == ref['input_rgb8_sha256']
            a = pixels.astype('f4') / 255
            m = arrays.load(ref['motion'])
            rgb, motion = torch.from_numpy(a).to('xpu'), torch.from_numpy(m).to('xpu')
            values, pairs = {'input': a}, {}
            row = dict(frame=i, reset=i == 0, input_rgb8_sha256=digest(pixels), motion=ref['motion'], runs={})
            order = ROUTES if i % 2 == 0 else ROUTES[::-1]
            for name in order:
                s = stacks[name]
                out, low, _ = run(name, face_scaler, rgb, motion, i == 0)
                o, b = out.cpu().numpy(), low.cpu().numpy()
                assert o.shape == (480, 864, 3) and o.dtype == np.float32 and np.isfinite(o).all()
                assert b.shape == (256, 256, 3) and b.dtype == np.float16 and np.isfinite(b).all()
                if name == 'baseline':
                    assert digest(o) == ref['full_raw_sha256'] and digest(b) == ref['low']['raw_sha256']
                assert raw(s.model._previous) == b.tobytes() and s.model.next_seed == i + 1
                values[name] = o.copy()
                pairs[name] = digest(o), digest(b)
                row['runs'][name] = dict(full_raw_sha256=digest(o), low_raw_sha256=digest(b),
                    next_seed=i+1, private_history_matches_low=True, finite=True)
                if i in (0, 120):
                    held[name, i] = out, low, o.tobytes(), b.tobytes()
                else:
                    out.zero_()
                    low.zero_()
                    assert raw(s.model._previous) == b.tobytes()
            if i == 0:
                first = pixels.copy(), pairs
            for out, low, ob, lb in held.values():
                assert raw(out) == ob and raw(low) == lb
            assert raw(rgb) == a.tobytes() and raw(motion) == m.tobytes()
            error = values['floor16'] - values['baseline']
            row['metrics'] = dict(full=metrics(values['floor16'], values['baseline']),
                roi=metrics(values['floor16'], values['baseline'], True))
            if i:
                fx, fy = xx + m[..., 0], yy + m[..., 1]
                valid = (fx >= 1) & (fx < 863) & (fy >= 1) & (fy < 479)
                warped = cv2.remap(previous_error, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                row['metrics']['temporal_error_mae_8bit'] = float(np.abs((error - warped)[valid]).mean(dtype='f8')*255)
            previous_error = error.copy()
            row.update(inputs_unchanged=True, held_outputs_unchanged=True)
            grid = Image.new('RGB', (2592, 1352), (19, 24, 32))
            draw = ImageDraw.Draw(grid)
            for j, (name, label) in enumerate(zip(('input', 'baseline', 'floor16'),
                    ('原始输入', '已审核参考：C512 FP16', '候选：C512 INT8 floor16'))):
                x = j*864
                im = Image.fromarray(np.rint(np.clip(values[name], 0, 1)*255).astype('u1'))
                draw.text((x+10, 0), label, font=font, fill='white')
                draw.text((x+10, 32), f'帧 {i:03d}/242 · {i/24:.3f}/10.125 秒 · 离线生成，原速回放', font=small, fill=(182, 194, 210))
                grid.paste(im, (x, 56))
                draw.text((x+10, 544), '同一区域放大 2 倍 · 模型处理完整帧', font=small, fill=(182, 194, 210))
                grid.paste(im.crop((256, 16, 640, 400)).resize((768, 768), Image.Resampling.NEAREST), (x+48, 584))
            writer.stdin.write(np.asarray(grid).tobytes())
            if i in (48, 96, 192):
                p = OUT / f'frame{i:03d}-full-and-face.png'
                grid.save(p)
                report['images'][str(p)] = sha(p)
            report['frames'].append(row)
            if i in (96, 192):
                check_body('floor16', candidate.graph.last_entry, f'face{i}', cpu=True)
            if i % 24 == 0 or i == 242:
                save()
                print(json.dumps(dict(face_frame=i,total=243)), flush=True)
        assert decoder.stdout.read(1) == b'' and decoder.wait(timeout=30) == 0
        writer.stdin.close()
        assert writer.wait(timeout=60) == 0
        rgb = torch.from_numpy(first[0].astype('f4')/255).to('xpu')
        motion = torch.from_numpy(arrays.load(repair['frames'][0]['motion'])).to('xpu')
        for name, s in stacks.items():
            out, low, _ = run(name, face_scaler, rgb, motion, True)
            assert (digest(out.cpu().numpy()), digest(low.cpu().numpy())) == first[1][name]
            assert s.model.next_seed == 1 and raw(s.model._previous) == raw(low)
            assert len(s.graph.entries) == 2 and s.graph.replays == 298
            assert s.rewrite.scopes == report['steady_scope_counts'][name] + (2 if name == 'floor16' else 0)
            report['constant_guard'][name]['after'] = constants(s)
            assert report['constant_guard'][name]['before'] == report['constant_guard'][name]['after']
            with s.installed():
                s.graph._validate()
            segments = torch.xpu.memory_snapshot(s.graph.capture_pool)
            for _, t in s.model.named_buffers():
                assert not any(v['address'] <= t.data_ptr() < v['address']+v['total_size'] for v in segments)
        for n, sc in [('car', car_scaler), ('face', face_scaler)]:
            assert tables[n] == [digest(t.cpu().numpy()) for row in sc.tables.values() for t in row]
        for out, low, ob, lb in held.values():
            assert raw(out) == ob and raw(low) == lb
        assert len(report['cpu_checks']) == 64
        assert candidate.c512_int8.ffn_calls == 192
        assert set(candidate.c512_int8.ffn_blocks.values()) == {12}
        report['candidate'] = candidate.metadata()
        for resources in (candidate.c512_int8.ffn_resources, candidate.c512_int8.compact_resources,
                          candidate.rewrite.resources, candidate.int8_vit.ffn_resources):
            assert resources and all(v['spills'] == 0 for v in resources.values())
        frozen_resources = range_report['resources'][range_report['calibration'][0]['name']]['floor16']
        for key, value in candidate.c512_int8.ffn_resources.items():
            assert value == frozen_resources['kernels'][key]
            assert candidate.c512_int8.ffn_selections[key] == frozen_resources['selections'][key]
        report['graphs'] = {n: s.graph.metadata() for n, s in stacks.items()}
        assert donor.graph.replays == 0 and not donor.graph.entries and donor.model._previous is None
        donor_after = dict(donor.model.named_buffers(remove_duplicate=False))
        assert donor.model.next_seed == 0 and donor_after.keys() == donor_buffers.keys()
        assert all(donor_after[k] is t for k, t in donor_buffers.items())
        report['donor_remained_fresh'] = True
    report['quality_summary'] = dict(
        mean_full_rgb_rmse_8bit=statistics.mean(v['metrics']['full']['rgb_rmse_8bit'] for v in report['frames']),
        mean_roi_rgb_rmse_8bit=statistics.mean(v['metrics']['roi']['rgb_rmse_8bit'] for v in report['frames']),
        mean_temporal_error_mae_8bit=statistics.mean(v['metrics']['temporal_error_mae_8bit'] for v in report['frames'][1:]))
    report['phase'] = 'video_finalization'
    save()
    print('Finalizing video and verifying243 decoded frames; no model rerun', flush=True)
    report['video_finalization'] = finalize(ffmpeg, encoded, OUT / 'comparison-full-and-face-24fps.mp4',
        2592, 1352, 243, '24/1', 10.125)
    report['video'] = report['video_finalization']['video']
    report.update(passed=True, phase='completed', frames_completed=243, reset_reproduces_first_frame=True,
        all_baseline_outputs_match_accepted=True, inputs_constants_history_held_outputs_checked=True,
        all_constants_and_history_outside_pool=True, no_extra_c512_debug_stores=True)
except BaseException:
    report.update(passed=False, phase='failed', error=traceback.format_exc())
    raise
finally:
    try:
        for p in processes:
            if p.poll() is None:
                p.kill()
                p.wait()
        for f in files:
            f.close()
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
print(json.dumps(dict(passed=True, frames=243, cpu_checks=64, video=report['video'],
    body=report['body'], paired=report['paired']['summary'], human_review='pending')), flush=True)
