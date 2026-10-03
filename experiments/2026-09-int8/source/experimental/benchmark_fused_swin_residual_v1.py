"""Four-way MLP/attention paired reduced-NR256 plus full1080 residual composition.

Three rotated 13-frame rounds, same prior low-resolution history and images.
Timing includes GPU prepare, whole NR, composition and final completion;
excludes uploads, motion estimation, capture/JIT, comparison, IO and presentation.
Unlike residual-scale-v1 this paired benchmark has one final synchronization.
"""
import argparse, hashlib, json, os, statistics, sys, time, traceback, urllib.request
from contextlib import contextmanager, nullcontext
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXACT = ROOT.parent / 'nr-b580'
R = EXACT / 'reference'
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
MODE = 'fp16_xmx'
VARIANTS = ('previous', 'mlp', 'swin', 'both')
OUT = DREF / 'results/fused-swin-residual256-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR'] = str(DREF / 'triton-cache-sm89-v1')
sys.path[:0] = [str(R), str(ROOT / 'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js, sha
gate = authenticate_main()
primitive_paths = [DREF / f'experimental/{name}/validation.json' for name in
                   ('fused-branched-pairs-v1', 'fused-c32-mlp-v1', 'fused-swin-core-v2')]
for primitive_path in primitive_paths:
    primitive = js(primitive_path)
    assert primitive['passed']
    assert all(sha(Path(path)) == digest for path, digest in primitive['sources'].items())
    assert js(primitive_path.parent.with_suffix('.log.lease.json'))['returncode'] == 0
prior_path = DREF / 'results/residual-scale-fp16_xmx-256-v1/validation.json'
assert sha(prior_path) == '056251b4e2a839fcc44845d9646d2cd8ae1e9ed7f99fd1858dc7e3d51e3e50f9'
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
from fused_branched_pairs_v2 import FusedPairs
from fused_c32_mlp_v1 import FusedC32
from serial_graph_workspace_v1 import share_before_capture
from residual_scale_v1 import ResidualScale
from graph_front_v6 import GraphFront
from swin_scheduling_v1 import SwinScheduling
from fused_swin_scheduling_v1 import FusedSwin
from window_layout_v1 import WindowLayout
from shared_model_buffers_v1 import share_identical_buffers
import compressed_arrays_v1 as arrays
inputs = R / 'inputs/flow-full-1920x1080-v3'
manifest = js(inputs / 'manifest.json')
names = ['benchmark_fused_swin_residual_v1.py', 'Run-FusedSwinResidualV1.cmd',
         'fused_branched_pairs_v2.py', 'fused_branched_pairs_v1.py', 'fused_mlp_pair_v1.py',
         'fused_c32_mlp_v1.py', 'fused_branched_mlp_v1.py', 'serial_graph_workspace_v1.py', 'residual_scale_v1.py',
         'strided_batched_v2.py', 'strided_batched_v1.py', 'window_layout_v1.py',
         'swin_scheduling_v1.py', 'fused_swin_scheduling_v1.py', 'fused_swin_core_v2.py', 'capture_body_v1.py', 'fused_cached_matrices_v2.py', 'fused_cached_matrices_v1.py',
         'fused_activation_int8_v1.py', 'fast_matrices_v3.py', 'static_weight_cache_v2.py', 'static_weight_cache_v1.py',
         'shared_model_buffers_v1.py', 'compressed_arrays_v1.py', 'immutable_artifacts_v1.py',
         *[f'graph_front_v{i}.py' for i in range(1, 7)]]
paths = [*(HERE / name for name in names), *primitive_paths, prior_path, inputs / 'manifest.json',
         *sorted((ROOT / 'backend/nr_backend').glob('*.py'))]
frozen = {str(path): sha(path) for path in paths}
for path in (ROOT / 'backend/nr_backend').glob('*.py'):
    assert path.read_bytes() == (EXACT / 'backend/nr_backend' / path.name).read_bytes()
OUT.mkdir()
report = dict(scope=__doc__, sources=frozen, exact_gate=gate, mode=MODE, passed=False,
              runs={name: [] for name in VARIANTS}, warmup=[], complete_migration=False,
              timing_scope=__doc__, progress_fallback=[], changes_quality_from_prior_residual=False, new_quality_approved=False, dispatch_semantics='Original logical operations; fused kernels do not issue each as separate GPU work')
models, providers, adapters, schedulers, layouts, held = {}, {}, {}, {}, {}, {}
pairs, c32 = {}, {}
expected = []
for old in prior['runs'][:13]:
    expected.append((arrays.load(old['output']), old['output'], old['runtime_dispatch'],
                     arrays.load(old['low_nr']), old['low_nr'], old['captured_dispatch']))
scaler = None

def save():
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')

def tensors(i):
    spec = manifest['frames'][i]
    pixels = np.asarray(Image.open(inputs / spec['file']).convert('RGB'), dtype='f4') / 255
    flow = np.fromfile(inputs / spec['motion_file'], '<f4').reshape(1080, 1920, 2)
    return pixels, flow, torch.from_numpy(pixels).to('xpu'), torch.from_numpy(flow).to('xpu')

@contextmanager
def installed(name):
    with providers[name].installed(), schedulers[name].installed(), layouts[name].installed(), adapters[name].installed():
        with (pairs[name].installed() if name in ('mlp', 'both') else nullcontext()):
            with (c32[name].installed() if name in ('mlp', 'both') else nullcontext()):
                yield

try:
    torch.set_num_threads(2)
    scaler = ResidualScale(256)
    for name in VARIANTS:
        model = MotionNR.from_assets(EXACT / 'model-assets/sf-v2/WEIGHTS_HT.bin', EXACT / 'model-assets/noise-sm89-v2', EXACT / 'model-assets/sigmoid-sm89-v1').to('xpu').eval()
        if models:
            report['shared_constants'] = share_identical_buffers(model, models['previous'])
        models[name] = model
        providers[name] = StridedMatrices()
        providers[name].select(MODE)
        adapters[name] = GraphFront(model, arithmetic=providers[name])
        schedulers[name] = FusedSwin(providers[name]) if name in ('swin', 'both') else SwinScheduling(enabled=True)
        layouts[name] = WindowLayout()
        pairs[name] = FusedPairs(model, providers[name])
        c32[name] = FusedC32(model, providers[name], bm=32, warps=4, stages=1)
    report['shared_transient_pool'] = share_before_capture(adapters.values())
    with torch.inference_mode():
        for name, model in models.items():
            if MODE.startswith('int8'):
                providers[name].static_cache.prepack(model)
            with installed(name):
                for i in (0, 1):
                    _, _, rgb, motion = tensors(i)
                    low_rgb, low_motion = scaler.prepare(rgb, motion)
                    with use_arithmetic_backend('triton'):
                        low_nr = model(low_rgb, low_motion, reset=i == 0)
                    value = scaler.composite(rgb, low_rgb, low_nr)
                    assert value.cpu().numpy().tobytes() == expected[i][0].tobytes()
                    assert low_nr.cpu().numpy().tobytes() == expected[i][3].tobytes()
            model.reset()
            report['warmup'].append(dict(name=name, graphs=adapters[name].metadata(), matrix_calls=dict(providers[name].calls)))
            print('Prewarmed ' + name + ' ' + MODE, flush=True)
        report['released_provider_packing'] = {}
        for name, provider in providers.items():
            report['released_provider_packing'][name] = provider.static_cache.packed_bytes
            provider.static_cache.entries = {}
        torch.xpu.synchronize()
        torch.xpu.empty_cache()
        for repetition in range(3):
            for i, spec in enumerate(manifest['frames']):
                pixels, flow, rgb, motion = tensors(i)
                order = list(VARIANTS)
                offset = (i + repetition) % len(order)
                order = order[offset:] + order[:offset]
                for name in order:
                    model, provider, adapter = models[name], providers[name], adapters[name]
                    provider.select(MODE)
                    torch.xpu.synchronize()
                    with installed(name):
                        started = time.perf_counter()
                        low_rgb, low_motion = scaler.prepare(rgb, motion)
                        with use_arithmetic_backend('triton') as dispatch:
                            low_nr = model(low_rgb, low_motion, reset=spec['reset'])
                            value = scaler.composite(rgb, low_rgb, low_nr)
                            torch.xpu.synchronize()
                        seconds = time.perf_counter() - started
                    actual, private = value.cpu().numpy(), model._previous.cpu().numpy()
                    target, meta, runtime_target, nr_target, nr_meta, body_dispatch = expected[i]
                    expected_dispatch = dict(runtime_target)
                    expected_dispatch.pop('xpu_graph_replay')
                    for key, count in body_dispatch.items():
                        if key != 'backend':
                            expected_dispatch[key] = expected_dispatch.get(key, 0) + count
                    nr = low_nr.cpu().numpy()
                    assert nr.tobytes() == nr_target.tobytes()
                    effective = dict(dispatch)
                    assert effective.pop('xpu_graph_replay') == 1
                    for key, count in adapter.last_entry.dispatch.items():
                        if key != 'backend':
                            effective[key] = effective.get(key, 0) + count
                    equal = actual.tobytes() == target.tobytes()
                    row = dict(round=repetition, frame=i, reset=spec['reset'], order=order, seconds=seconds,
                               byte_equal_prior=equal, output=meta if equal else arrays.save(actual),
                               low_nr=nr_meta, private_byte_equal_low_nr=private.tobytes() == nr.tobytes(), next_seed=model.next_seed,
                               effective_dispatch=effective, runtime_dispatch=dict(dispatch), graph_replays=adapter.replays)
                    report['runs'][name].append(row)
                    save()
                    assert equal and row['private_byte_equal_low_nr'] and np.isfinite(actual).all()
                    assert effective == expected_dispatch and model.next_seed == (1 if spec['reset'] else i + 1)
                    if name not in held:
                        held[name] = value, actual.tobytes()
                    else:
                        value.zero_()
                        assert model._previous.cpu().numpy().tobytes() == private.tobytes()
                for old_value, old_bytes in held.values():
                    assert old_value.cpu().numpy().tobytes() == old_bytes
                assert rgb.cpu().numpy().tobytes() == pixels.tobytes() and motion.cpu().numpy().tobytes() == flow.tobytes()
                print(json.dumps(dict(round=repetition, frame=i, mode=MODE, seconds={name: report['runs'][name][-1]['seconds'] for name in models}, all_equal=True)), flush=True)
        for name, model in models.items():
            provider, adapter = providers[name], adapters[name]
            if MODE.startswith('int8'):
                provider.static_cache.prepack(model)
            provider.select(MODE)
            _, _, rgb, motion = tensors(1)
            before = adapter.replays
            marks = []
            pairs[name].calls = {}
            c32[name].calls = {}
            low_rgb, low_motion = scaler.prepare(rgb, motion)
            with installed(name), use_arithmetic_backend('triton'):
                low_nr = model(low_rgb, low_motion, reset=False, progress=marks.append)
                value = scaler.composite(rgb, low_rgb, low_nr)
            assert value.cpu().numpy().tobytes() == expected[1][0].tobytes() and model.next_seed == 2 and adapter.replays == before
            assert marks == ['pre', 'encoder C32', 'encoder C64', 'encoder C128', 'encoder C256', 'encoder C512', 'ViT', 'decoder C512', 'decoder C256', 'decoder C128', 'decoder C64', 'decoder C32', 'RGB']
            report['progress_fallback'].append(dict(name=name, marks=marks, byte_equal=True, matrix_calls=dict(provider.calls), fused_pairs=dict(pairs[name].calls), fused_c32=dict(c32[name].calls), swin_calls=dict(schedulers[name].counts)))
        # Diagnostic only: static body repeats exclude every dynamic front task.
        # Done after functional runs; no history/seed is advanced by direct replay.
        report['body_only_diagnostic'] = {}
        for name, adapter in adapters.items():
            entries = []
            for entry in adapter.entries.values():
                target = entry.output.cpu().numpy().tobytes()
                samples = []
                for _ in range(5):
                    torch.xpu.synchronize()
                    started = time.perf_counter()
                    for _ in range(10):
                        entry.graph.replay()
                    torch.xpu.synchronize()
                    samples.append((time.perf_counter() - started) / 10)
                    assert entry.output.cpu().numpy().tobytes() == target
                entries.append(dict(temporal=entry.inputs['previous'] is not None,
                                    samples_seconds=samples, median_seconds=statistics.median(samples),
                                    output_unchanged=True))
            report['body_only_diagnostic'][name] = entries
        means = {name: statistics.mean(row['seconds'] for row in rows) for name, rows in report['runs'].items()}
        report.update(passed=True, matching_mean_seconds=means, speedup={name: means['previous'] / means[name] for name in VARIANTS[1:]},
                      round_mean_seconds={name: [statistics.mean(row['seconds'] for row in rows if row['round'] == i) for i in range(3)] for name, rows in report['runs'].items()},
                      full_outputs_verified=156, held_outputs_survive_replay=True, caller_ownership_guards_passed=True,
                      graphs={name: adapter.metadata() for name, adapter in adapters.items()})
except Exception as error:
    report.update(error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    for adapter in adapters.values():
        adapter.close()
    assert all(sha(Path(path)) == digest for path, digest in frozen.items())
    authenticate_main()
    save()
print(json.dumps({key: report[key] for key in ('passed', 'matching_mean_seconds', 'speedup')}), flush=True)
