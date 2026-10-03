"""First reduced-input full-NR latency/quality probe, not a visual acceptance run.

Full1080 source frames180..191 plus reset180, repeated twice. Motion is the same
saved current-to-previous full-resolution flow, area-resized with XY scaling.
Time includes GPU color/motion resampling, full low-resolution NR with its own
history and full1080 residual composition. Excludes upload, flow estimation,
capture/JIT, validation, output saving and game presentation/resource handoff.
"""
import argparse, hashlib, json, os, statistics, sys, time, traceback, urllib.request
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXACT = ROOT.parent / 'nr-b580'
R = EXACT / 'reference'
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['fp16_xmx', 'int8_fused_v2'], required=True)
p.add_argument('--size', type=int, choices=[256, 512], required=True)
args = p.parse_args()
MODE, SIZE = args.mode, args.size
OUT = DREF / f'results/residual-scale-{MODE}-{SIZE}-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR'] = str(DREF / 'triton-cache-sm89-v1')
sys.path[:0] = [str(R), str(ROOT / 'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js, sha
gate = authenticate_main()
prior_path = DREF / f'results/graph-native-1920x1080-{MODE}-v4/validation.json'
assert sha(prior_path) == {'fp16_xmx':'52b37220026c04ba5c2ebfcad8269ee0eda2edb2c580238241149f96563e1be7', 'int8_fused_v2':'6200438be6f5e3b5ebdbdaa35fea3c5366a30ccc2b9f889a33ab54f72cfbb6fb'}[MODE]
prior = js(prior_path)
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
from PIL import Image
import torch
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from strided_batched_v2 import StridedMatrices
from graph_front_v6 import GraphFront
from swin_scheduling_v1 import SwinScheduling
from window_layout_v1 import WindowLayout
from residual_scale_v1 import ResidualScale, filter_table
import compressed_arrays_v1 as arrays
inputs = R / 'inputs/flow-full-1920x1080-v3'
manifest = js(inputs / 'manifest.json')
names = ['benchmark_residual_scale_v1.py', 'Run-ResidualScaleV1.cmd', 'residual_scale_v1.py',
         'strided_batched_v2.py', 'strided_batched_v1.py', 'window_layout_v1.py', 'swin_scheduling_v1.py',
         'capture_body_v1.py', 'fused_cached_matrices_v2.py', 'fused_cached_matrices_v1.py',
         'fused_activation_int8_v1.py', 'fast_matrices_v3.py', 'static_weight_cache_v2.py', 'static_weight_cache_v1.py',
         'compressed_arrays_v1.py', 'immutable_artifacts_v1.py', *[f'graph_front_v{i}.py' for i in range(1, 7)]]
paths = [*(HERE / name for name in names), prior_path, inputs / 'manifest.json',
         *sorted((ROOT / 'backend/nr_backend').glob('*.py'))]
frozen = {str(path): sha(path) for path in paths}
OUT.mkdir()
report = dict(scope=__doc__, sources=frozen, exact_gate=gate, mode=MODE, size=SIZE, passed=False,
              runs=[], primitive_checks={}, full_same_precision_reference=str(prior_path),
              complete_migration=False, new_quality_approved=False, timing_scope=__doc__,
              timing_boundary='Synchronized prepare, NR and composite stages; three stage boundaries included.')
adapter = None

def save():
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')

def tensors(i):
    spec = manifest['frames'][i]
    rgb = np.asarray(Image.open(inputs / spec['file']).convert('RGB'), dtype='f4') / 255
    flow = np.fromfile(inputs / spec['motion_file'], '<f4').reshape(1080, 1920, 2)
    return rgb, flow, torch.from_numpy(rgb).to('xpu'), torch.from_numpy(flow).to('xpu')

def cpu_axis(value, kind, axis, destination, half=True):
    ii, ww = filter_table(value.shape[axis], destination, kind)
    shape = list(value.shape)
    shape[axis] = destination
    out = np.zeros(shape, dtype='f4')
    weight_shape = [1, 1, 1]
    weight_shape[axis] = destination
    for tap in range(ii.shape[1]):
        out += np.take(value, ii[:, tap], axis=axis).astype('f4') * ww[:, tap].reshape(weight_shape)
    return out.astype('f2') if half else out

try:
    torch.set_num_threads(2)
    scaler = ResidualScale(SIZE)
    report['scale'] = scaler.metadata()
    pixels, flow, rgb, motion = tensors(0)
    with torch.inference_mode():
        canvas, low_motion = scaler.prepare(rgb, motion)
        expected = cpu_axis(cpu_axis(pixels, 'lanczos2', 0, scaler.active_h), 'lanczos2', 1, SIZE)
        active = canvas[scaler.top:scaler.top+scaler.active_h].cpu().numpy()
        difference = float(np.max(np.abs(active - expected.astype('f4'))))
        assert difference <= 0.0009765625
        identity = scaler.composite(rgb, canvas, canvas)
        assert identity.cpu().numpy().tobytes() == pixels.tobytes()
        constant = torch.ones_like(motion)
        constant[..., 0] *= 7.5
        constant[..., 1] *= -3.75
        _, scaled = scaler.prepare(rgb, constant)
        expected_motion = np.array([7.5 * SIZE / 1920, -3.75 * scaler.active_h / 1080], dtype='f2')
        actual_motion = scaled[scaler.top:scaler.top+scaler.active_h].cpu().numpy()
        assert np.array_equal(actual_motion, np.broadcast_to(expected_motion, actual_motion.shape))
        assert torch.count_nonzero(scaled[:scaler.top]).item() == 0 and torch.count_nonzero(scaled[scaler.top+scaler.active_h:]).item() == 0
        report['primitive_checks'] = dict(cpu_gpu_color_max_abs=difference, zero_residual_identity_bytes=True,
            constant_motion_units_scaled_exact_half=True, padded_motion_zero=True,
            first_canvas=arrays.save(canvas.cpu().numpy()), first_low_motion=arrays.save(low_motion.cpu().numpy()))
    model = MotionNR.from_assets(EXACT / 'model-assets/sf-v2/WEIGHTS_HT.bin',
        EXACT / 'model-assets/noise-sm89-v2', EXACT / 'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    provider = StridedMatrices()
    provider.select(MODE)
    adapter = GraphFront(model, arithmetic=provider)
    held = []
    with torch.inference_mode(), provider.installed(), SwinScheduling(enabled=True).installed(), WindowLayout().installed(), adapter.installed():
        if MODE.startswith('int8'):
            provider.static_cache.prepack(model)
        for i in (0, 1):
            _, _, rgb, motion = tensors(i)
            low_rgb, low_motion = scaler.prepare(rgb, motion)
            with use_arithmetic_backend('triton'):
                low_nr = model(low_rgb, low_motion, reset=i == 0)
            scaler.composite(rgb, low_rgb, low_nr)
        torch.xpu.synchronize()
        report['released_provider_packing'] = provider.static_cache.packed_bytes
        provider.static_cache.entries = {}
        torch.xpu.empty_cache()
        model.reset()
        print('Prewarmed reduced NR ' + MODE + ' ' + str(SIZE), flush=True)
        for repetition in range(2):
            for i, spec in enumerate(manifest['frames']):
                pixels, flow, rgb, motion = tensors(i)
                provider.select(MODE)
                torch.xpu.synchronize()
                start = time.perf_counter()
                low_rgb, low_motion = scaler.prepare(rgb, motion)
                torch.xpu.synchronize()
                after_prepare = time.perf_counter()
                with use_arithmetic_backend('triton') as dispatch:
                    low_nr = model(low_rgb, low_motion, reset=spec['reset'])
                    torch.xpu.synchronize()
                after_nr = time.perf_counter()
                output = scaler.composite(rgb, low_rgb, low_nr)
                torch.xpu.synchronize()
                stop = time.perf_counter()
                actual, nr = output.cpu().numpy(), low_nr.cpu().numpy()
                private = model._previous.cpu().numpy()
                assert np.isfinite(actual).all() and np.isfinite(nr).all() and private.tobytes() == nr.tobytes()
                assert model.next_seed == (1 if spec['reset'] else i + 1)
                reference = arrays.load(prior['runs']['graph'][i]['output']).astype('f4')
                error = actual - reference
                mse = float(np.mean(error.astype('f8') ** 2))
                if repetition:
                    old = report['runs'][i]
                    assert actual.tobytes() == arrays.load(old['output']).tobytes()
                    assert nr.tobytes() == arrays.load(old['low_nr']).tobytes()
                    output_meta, nr_meta = old['output'], old['low_nr']
                else:
                    output_meta, nr_meta = arrays.save(actual), arrays.save(nr)
                report['runs'].append(dict(round=repetition, frame=i, reset=spec['reset'],
                    input_rgb8_sha256=hashlib.sha256(np.rint(pixels*255).astype('u1').tobytes()).hexdigest(),
                    output=output_meta, low_nr=nr_meta, private_byte_equal_low_nr=True, next_seed=model.next_seed,
                    seconds=stop-start, prepare_seconds=after_prepare-start, nr_seconds=after_nr-after_prepare,
                    composite_seconds=stop-after_nr, runtime_dispatch=dict(dispatch), captured_dispatch=adapter.last_entry.dispatch,
                    psnr_vs_full_same_precision=-10*np.log10(mse) if mse else None, max_abs_vs_full=float(np.max(np.abs(error))),
                    repeat_bytes_equal=True if repetition else None))
                for old_value, old_bytes in held:
                    assert old_value.cpu().numpy().tobytes() == old_bytes
                if not held:
                    held.append((output, actual.tobytes()))
                else:
                    output.zero_()
                    assert model._previous.cpu().numpy().tobytes() == private.tobytes()
                assert rgb.cpu().numpy().tobytes() == pixels.tobytes() and motion.cpu().numpy().tobytes() == flow.tobytes()
                if not repetition and i in (0, 6, 11):
                    panels = [pixels, reference, actual]
                    strip = np.concatenate([np.rint(np.clip(x,0,1)*255).astype('u1') for x in panels], axis=1)
                    Image.fromarray(strip).save(OUT / f'frame{i:02d}-original-full-reduced.png')
                save()
                print(json.dumps(dict(round=repetition, frame=i, size=SIZE, mode=MODE, milliseconds=(stop-start)*1000, psnr=report['runs'][-1]['psnr_vs_full_same_precision'])), flush=True)
        # Restore the ordinary body once and compare its reset output at the same small input.
        if MODE.startswith('int8'):
            provider.static_cache.prepack(model)
        _, _, rgb, motion = tensors(0)
        low_rgb, low_motion = scaler.prepare(rgb, motion)
        before = adapter.replays
        with use_arithmetic_backend('triton'):
            eager = model(low_rgb, low_motion, reset=True, progress=lambda name: None)
        assert eager.cpu().numpy().tobytes() == arrays.load(report['runs'][0]['low_nr']).tobytes() and adapter.replays == before
        report.update(passed=True, complete_frames=26, graph_eager_low_nr_reset_bytes_equal=True,
            graphs=adapter.metadata(), held_output_preserved=True, caller_inputs_and_private_preserved=True,
            mean_seconds={key:statistics.mean(row[key] for row in report['runs']) for key in ('seconds','prepare_seconds','nr_seconds','composite_seconds')},
            round_mean_seconds=[statistics.mean(row['seconds'] for row in report['runs'] if row['round']==i) for i in range(2)])
except Exception as error:
    report.update(error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    if adapter is not None:
        adapter.close()
    assert all(sha(Path(path)) == digest for path, digest in frozen.items())
    authenticate_main()
    save()
print(json.dumps({key:report[key] for key in ('passed','mean_seconds','round_mean_seconds')}), flush=True)
