"""Full390-frame1080 stream; one resident model, independent histories per precision.

The reference for this new stream is B580 exact arithmetic, not a new4060 capture.
All complete half RGB and shared half motion are retained losslessly on D.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import traceback
import urllib.request

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXACT = ROOT.parent / 'nr-b580'
R = EXACT / 'reference'
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
p = argparse.ArgumentParser()
p.add_argument('--mode', required=True, choices=['baseline', 'fp16_xmx', 'int8_fused_v2'])
MODE = p.parse_args().mode
OUT = DREF / f'results/long-precision-1080-{MODE}-v1'
assert not OUT.exists(), 'Completed or partial experiments are immutable'
MIN_FREE = 64 * 1024**3
assert shutil.disk_usage(DREF).free > MIN_FREE
os.environ['TRITON_CACHE_DIR'] = str(DREF / 'triton-cache-sm89-v1')
sys.path[:0] = [str(R), str(ROOT / 'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js, sha

gate = authenticate_main()
source = R / 'inputs/visual-qa-01/clip1080.mp4'
assert sha(source) == '6432062912727d1f1c46730c81e3f67794808dc850dfd4a4ccb98995b9c841df'
ffmpeg = ROOT.parent / 'xess-tools/work/d3d12-media-pipeline/deps/ffmpeg-lgpl-shared-9.0/ffmpeg-n9.0-latest-win64-lgpl-shared-9.0/bin/ffmpeg.exe'
assert sha(ffmpeg) == '012b2f638ecc1dd0c58273a0db1d7ae28f1f287d4fee718dd16bc714600fd7d4'
ffprobe = ffmpeg.with_name('ffprobe.exe')
probe = json.loads(subprocess.check_output([str(ffprobe), '-v', 'error', '-select_streams', 'v:0',
    '-show_entries', 'stream=width,height,avg_frame_rate,nb_frames,duration', '-of', 'json', str(source)], creationflags=0x08000000))
s = probe['streams'][0]
assert (s['width'], s['height'], s['avg_frame_rate'], int(s['nb_frames'])) == (1920, 1080, '60000/1001', 390)
decode_filter = 'scale=in_color_matrix=bt709:in_range=tv:flags=lanczos+accurate_rnd+full_chroma_int+full_chroma_inp,format=rgb24'
short_name = 'exact-window-1920x1080-v1' if MODE == 'baseline' else f'window-layout-full-1920x1080-{MODE}-' + ('v1' if MODE == 'fp16_xmx' else 'v2')
short_path = DREF / 'results' / short_name / 'validation.json'
assert sha(short_path) == {
    'baseline': '751ae532d00fff2c14cc3e1449047f364afd5452b778518d4479712f5a578a8b',
    'fp16_xmx': 'c982f10b46a5c29947a3ed4acb9b2d2c921d0c907e8497c1bdded00937519305',
    'int8_fused_v2': '46b54b2698e0c646ddd3a5f836d836146ba80c565f8dd28aa0b3e0b3d150f3cc',
}[MODE]
short = js(short_path)
short_audit = short_path.with_name('saved-audit-v1.json')
assert short['passed'] and js(short_audit)['passed'] and js(short_audit)['report_sha256'] == sha(short_path)
assert all(sha(Path(path)) == digest for path, digest in short['sources'].items())
short_rows = short['runs']['graph' if MODE == 'baseline' else MODE + '_layout_graph']
teacher_path = DREF / 'results/long-precision-1080-baseline-v1/validation.json'
teacher = None
if MODE != 'baseline':
    teacher = js(teacher_path)
    assert teacher['passed'] and teacher['frames_completed'] == len(teacher['frames']) == 390
    assert teacher['mode'] == 'baseline' and not teacher['native_capture_for_this_sequence']
    assert teacher['decode_filter'] == decode_filter
    assert js(Path(str(teacher_path.parent) + '.log.lease.json'))['returncode'] == 0
    assert all(sha(Path(path)) == digest for path, digest in teacher['sources'].items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']

import cv2
import numpy as np
import torch
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from fused_cached_matrices_v2 import FusedCachedMatrices
from strided_batched_v2 import StridedMatrices
from window_layout_v1 import WindowLayout
from swin_scheduling_v1 import SwinScheduling
from graph_front_v6 import GraphFront
import compressed_arrays_v1 as arrays

cv2.setNumThreads(2)
cv2.ocl.setUseOpenCL(False)
torch.set_num_threads(2)
names = ['Run-Long1080V1.cmd', 'strided_batched_v2.py', 'strided_batched_v1.py', 'window_layout_v1.py',
         'graph_front_v6.py', 'graph_front_v5.py', 'graph_front_v4.py', 'graph_front_v3.py', 'graph_front_v2.py',
         'graph_front_v1.py', 'capture_body_v1.py', 'swin_scheduling_v1.py', 'fused_cached_matrices_v2.py',
         'fused_cached_matrices_v1.py', 'fused_activation_int8_v1.py', 'static_weight_cache_v2.py',
         'static_weight_cache_v1.py', 'fast_matrices_v3.py', 'compressed_arrays_v1.py', 'immutable_artifacts_v1.py']
paths = [Path(__file__), source, ffmpeg, ffprobe, short_path, short_audit, *[HERE / n for n in names],
         *sorted((ROOT / 'backend/nr_backend').glob('*.py')), *sorted(Path(cv2.__file__).parent.glob('*.pyd'))]
if teacher is not None:
    paths += [teacher_path, Path(str(teacher_path.parent) + '.log.lease.json')]
frozen = {str(path): sha(path) for path in paths}
assert {path.name: path.read_bytes() for path in (ROOT / 'backend/nr_backend').glob('*.py')} == {
    path.name: path.read_bytes() for path in (EXACT / 'backend/nr_backend').glob('*.py')}
OUT.mkdir()
report = dict(scope=__doc__, sources=frozen, exact_gate=gate, mode=MODE, passed=False,
    source=str(source), probe=probe, dimension='1920x1080', expected_frames=390, frames_completed=0,
    fps='60000/1001', duration_seconds=390 * 1001 / 60000, decode_filter=decode_filter,
    reference='B580 exact35 arithmetic, default controls, SDR, zero depth', native_capture_for_this_sequence=False,
    history='Independent model, reset only on first frame, 389 uninterrupted temporal frames, final reset reproduction',
    flow='OpenCV DIS MEDIUM current-to-previous pixels, OpenCL off, 2 CPU threads, prequantized FP16; shared complete arrays across modes',
    frames=[], warmup=[], complete_migration=False, human_review='pending_1080_long_fast_output',
    versions=dict(torch=torch.__version__, numpy=np.__version__, opencv=cv2.__version__),
    timing_scope='Diagnostic full model plus synchronization including dynamic front/graph copies/guards/history; excludes upload, flow, validation and IO. Separate process per precision; not a paired performance benchmark.',
    minimum_free_disk_bytes=MIN_FREE, free_disk_before=shutil.disk_usage(DREF).free)
decoder = adapter = None
held = []
used = {}
experiment = FusedCachedMatrices() if MODE == 'baseline' else StridedMatrices()
scheduler = SwinScheduling(enabled=True)
layout = WindowLayout()

def rawsha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()

def save():
    temp = OUT / 'progress.tmp'
    temp.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8', newline='\n')
    temp.replace(OUT / 'validation.json')

def read_frame():
    chunks, remaining = [], 1080 * 1920 * 3
    while remaining:
        data = decoder.stdout.read(remaining)
        if not data:
            raise EOFError('Incomplete decoded frame')
        chunks.append(data)
        remaining -= len(data)
    return np.frombuffer(b''.join(chunks), dtype='u1').reshape(1080, 1920, 3).copy()

def flow_for(pixels, previous_gray):
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    flow = np.zeros((1080, 1920, 2), dtype='f4') if previous_gray is None else cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(gray, previous_gray, None)
    assert np.isfinite(flow).all() and np.abs(flow).max() <= 65504
    return gray, flow.astype('f2')

def tensors(pixels, flow):
    return torch.from_numpy(pixels.astype('f4') / 255).to('xpu'), torch.from_numpy(flow).to('xpu')

try:
    # Independent convention check before creating any sequence flow.
    rng = np.random.default_rng(90611)
    a = cv2.GaussianBlur(rng.integers(0, 256, (192, 256), dtype='u1'), (5, 5), .8)
    b = cv2.warpAffine(a, np.array([[1, 0, 5], [0, 1, -3]], dtype='f4'), (256, 192), borderMode=cv2.BORDER_REFLECT_101)
    test = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(b, a, None)
    endpoint_error = float(np.median(np.linalg.norm(test[32:-32, 32:-32] - np.array([-5, 3]), axis=2)))
    assert endpoint_error < .5
    report['flow_direction_endpoint_error'] = endpoint_error
    decoder = subprocess.Popen([str(ffmpeg), '-v', 'error', '-i', str(source), '-vf', decode_filter,
        '-frames:v', '390', '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', 'rgb24', 'pipe:1'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=0x08000000)
    warm_pixels = [read_frame(), read_frame()]
    if teacher is None:
        gray0, flow0 = flow_for(warm_pixels[0], None)
        gray1, flow1 = flow_for(warm_pixels[1], gray0)
        warm_flows = [flow0, flow1]
    else:
        warm_flows = [arrays.load(teacher['frames'][i]['motion']) for i in (0, 1)]
    warm_outputs = []
    model = MotionNR.from_assets(EXACT / 'model-assets/sf-v2/WEIGHTS_HT.bin',
        EXACT / 'model-assets/noise-sm89-v2', EXACT / 'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    experiment.select(MODE)
    adapter = GraphFront(model, arithmetic=experiment)
    with torch.inference_mode(), experiment.installed(), scheduler.installed(), layout.installed(), adapter.installed():
        if MODE == 'int8_fused_v2':
            experiment.static_cache.prepack(model)
        for i in (0, 1):
            rgb, motion = tensors(warm_pixels[i], warm_flows[i])
            started = time.perf_counter()
            with use_arithmetic_backend('triton'):
                value = model(rgb, motion, reset=i == 0)
                torch.xpu.synchronize()
            warm_outputs.append(value.cpu().numpy().copy())
            report['warmup'].append(dict(frame=i, seconds=time.perf_counter() - started))
        model.reset()
        report['released_provider_packed_bytes'] = experiment.static_cache.packed_bytes
        experiment.static_cache.entries = {}
        del value, rgb, motion
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        print('Prewarmed full390 ' + MODE, flush=True)
        save()
        previous_gray = None
        for i in range(390):
            assert shutil.disk_usage(DREF).free > MIN_FREE, 'Preserve minimum free D space'
            pixels = warm_pixels[i] if i < 2 else read_frame()
            if teacher is None:
                if i < 2:
                    previous_gray, flow = (gray0 if i == 0 else gray1), warm_flows[i]
                else:
                    previous_gray, flow = flow_for(pixels, previous_gray)
                motion_meta = arrays.save(flow)
            else:
                original = teacher['frames'][i]
                assert rawsha(pixels) == original['input_rgb8_sha256']
                motion_meta = original['motion']
                flow = arrays.load(motion_meta)
            assert flow.shape == (1080, 1920, 2) and flow.dtype == np.dtype('f2') and np.isfinite(flow).all()
            rgb, motion = tensors(pixels, flow)
            experiment.select(MODE)
            torch.xpu.synchronize()
            torch.xpu.reset_peak_memory_stats()
            started = time.perf_counter()
            with use_arithmetic_backend('triton') as dispatch:
                value = model(rgb, motion, reset=i == 0)
                torch.xpu.synchronize()
            seconds = time.perf_counter() - started
            actual, private = value.cpu().numpy(), model._previous.cpu().numpy()
            assert actual.shape == (1080, 1920, 3) and actual.dtype == np.dtype('f2')
            assert np.isfinite(actual).all() and private.tobytes() == actual.tobytes()
            assert model.next_seed == i + 1
            if i < 2:
                assert actual.tobytes() == warm_outputs[i].tobytes()
            meta = arrays.save(actual)
            effective = dict(dispatch)
            assert effective.pop('xpu_graph_replay') == 1
            for key, count in adapter.last_entry.dispatch.items():
                if key != 'backend':
                    effective[key] = effective.get(key, 0) + count
            assert effective == short_rows[0 if i == 0 else 1]['effective_dispatch']
            assert adapter.owned_packed_bytes == (147185664 if MODE == 'int8_fused_v2' else 0)
            for old_value, old_bytes in held:
                assert old_value.cpu().numpy().tobytes() == old_bytes
            if i in (0, 195):
                held.append((value, actual.tobytes()))
            else:
                value.zero_()
                assert model._previous.cpu().numpy().tobytes() == private.tobytes()
            assert rgb.cpu().numpy().tobytes() == (pixels.astype('f4') / 255).tobytes()
            assert motion.cpu().numpy().tobytes() == flow.tobytes()
            row = dict(frame=i, source_frame=i, input_rgb8_sha256=rawsha(pixels), motion=motion_meta,
                output=meta, private=meta, private_byte_equal_output=True, reset=i == 0, next_seed=model.next_seed,
                warmup_repeat_equal=True if i < 2 else None, finite=True, seconds=seconds,
                runtime_dispatch=dict(dispatch), effective_dispatch=effective, graph_replays=adapter.replays,
                peak_allocated=torch.xpu.max_memory_allocated(), peak_reserved=torch.xpu.max_memory_reserved())
            if teacher is not None:
                target = arrays.load(teacher['frames'][i]['output'])
                difference = actual.astype('f4') - target.astype('f4')
                mse = float(np.square(difference.astype('f8')).mean())
                row.update(mse=mse, psnr_db=None if mse == 0 else float(-10 * np.log10(mse)),
                    mae=float(np.abs(difference).mean()), max_abs=float(np.abs(difference).max()))
            report['frames'].append(row)
            report['frames_completed'] = i + 1
            for artifact in (meta, motion_meta):
                used[artifact['path']] = artifact['stored_bytes']
            if i % 15 == 0 or i == 389:
                report['graphs'] = adapter.metadata()
                save()
                print(json.dumps(dict(mode=MODE, frame=i, total=390, next_seed=model.next_seed, finite=True,
                    diagnostic_seconds=seconds, psnr_db=row.get('psnr_db'))), flush=True)
        assert decoder.stdout.read() == b''
        assert decoder.wait(timeout=30) == 0, decoder.stderr.read().decode(errors='replace')
        rgb, motion = tensors(warm_pixels[0], warm_flows[0])
        experiment.select(MODE)
        with use_arithmetic_backend('triton'):
            value = model(rgb, motion, reset=True)
        assert value.cpu().numpy().tobytes() == arrays.load(report['frames'][0]['output']).tobytes()
        assert model.next_seed == 1 and adapter.replays == 393 and len(adapter.entries) == 2
        assert len({str(entry.graph.pool()) for entry in adapter.entries.values()}) == 1
        for old_value, old_bytes in held:
            assert old_value.cpu().numpy().tobytes() == old_bytes
        report.update(passed=True, graphs=adapter.metadata(), held_outputs_survive_replay=True,
            reset_reproduces_first_frame=True, independent_history=True, caller_ownership_guards_passed=True,
            owned_packed_bytes=adapter.owned_packed_bytes, unique_used_artifact_bytes=sum(used.values()),
            diagnostic_mean_model_seconds=statistics.mean(row['seconds'] for row in report['frames']),
            peak_allocated_bytes=max(row['peak_allocated'] for row in report['frames']),
            peak_reserved_bytes=max(row['peak_reserved'] for row in report['frames']),
            free_disk_after=shutil.disk_usage(DREF).free)
except Exception as error:
    report.update(passed=False, error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    if adapter is not None:
        adapter.close()
    if decoder is not None and decoder.poll() is None:
        decoder.kill()
        decoder.wait()
    assert all(sha(Path(path)) == digest for path, digest in frozen.items())
    authenticate_main()
    save()
print(json.dumps({key: report[key] for key in ('passed', 'mode', 'frames_completed', 'unique_used_artifact_bytes', 'diagnostic_mean_model_seconds')}), flush=True)
