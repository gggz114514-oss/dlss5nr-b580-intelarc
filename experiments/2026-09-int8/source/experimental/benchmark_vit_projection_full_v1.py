"""Two-way four-part ViT projection fusion against the complete prior QKV/C512/head fused stack.

Three rotated 13-frame rounds with independent history, authenticated shared constants
and a serialized transient pool. Includes GPU model and completion; residual mode
also includes color/motion resize and full1080 composition. Excludes motion
estimation, upload, JIT, validation, IO, presentation and game contention.
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
VARIANTS = ('previous', 'projection')
OUT = DREF / 'results/vit-projection-full480-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR'] = str(DREF / 'triton-cache-sm89-v1')
sys.path[:0] = [str(R), str(ROOT / 'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js, sha
gate = authenticate_main()
primitive_paths = [DREF / f'experimental/{name}/validation.json' for name in
                   ('fused-branched-pairs-v1', 'fused-c32-mlp-v1', 'fused-swin-core-v2', 'fused-dynamic-front-v1', 'fused-split-ffwd-v1', 'fused-qkv-pack-v1', 'fused-vit-projection-v1', 'fused-vit-projection-v2')]
for primitive_path in primitive_paths:
    primitive = js(primitive_path)
    assert primitive['passed']
    assert all(sha(Path(path)) == digest for path, digest in primitive['sources'].items())
    assert js(primitive_path.parent.with_suffix('.log.lease.json'))['returncode'] == 0
prior_path = DREF / 'results/fast-precision-864x480-v1/validation.json'
assert sha(prior_path) == '7a43ff73d452cabe3e68ea1a15db1537417384a57c9a2443fcb16b095d5a5110'
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
from fused_split_ffwd_v2 import FusedSplit
from fused_head_layout_v1 import HeadLayout
from qkv_head_layout_v1 import QKVHeadLayout
from fused_vit_projection_v3 import FusedVitProjection
from fused_dynamic_front_v1 import FusedFront
from fused_c32_mlp_v1 import FusedC32
from serial_graph_workspace_v1 import share_before_capture
from graph_front_v6 import GraphFront
from swin_scheduling_v1 import SwinScheduling
from fused_swin_scheduling_v1 import FusedSwin
from window_layout_v1 import WindowLayout
from shared_model_buffers_v1 import share_identical_buffers
import compressed_arrays_v1 as arrays
inputs = R / 'inputs/flow-full-864x480-v2'
manifest = js(inputs / 'manifest.json')
names = ['fused_vit_projection_v3.py', 'fused_vit_projection_v1.py', 'fused_vit_projection_v2.py', 'fused_qkv_pack_v1.py', 'qkv_head_layout_v1.py', 'benchmark_vit_projection_full_v1.py', 'Run-VitProjectionFullV1.cmd',
         'fused_split_ffwd_v1.py', 'fused_split_ffwd_v2.py', 'fused_head_layout_v1.py', 'fused_swin_heads_v1.py', 'fused_dynamic_front_v1.py', 'fused_branched_pairs_v2.py', 'fused_branched_pairs_v1.py', 'fused_mlp_pair_v1.py',
         'fused_c32_mlp_v1.py', 'fused_branched_mlp_v1.py', 'serial_graph_workspace_v1.py',
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
              timing_scope=__doc__, progress_fallback=[], changes_quality=False, dispatch_semantics='Original logical operations; fused kernels do not issue each as separate GPU work')
models, providers, adapters, schedulers, layouts, held = {}, {}, {}, {}, {}, {}
pairs, c32, splits, fronts = {}, {}, {}, {}
vits = {}
expected = []
for old in prior['runs']['fp16_xmx' if MODE == 'fp16_xmx' else 'int8_dense']:
    meta = old['actual']
    assert sha(Path(meta['path'])) == meta['sha256']
    expected.append((np.load(meta['path'], allow_pickle=False).astype('f2'), meta, old['dispatches']))

def save():
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')

def tensors(i):
    spec = manifest['frames'][i]
    pixels = np.asarray(Image.open(inputs / spec['file']).convert('RGB'), dtype='f4') / 255
    flow = np.fromfile(inputs / spec['motion_file'], '<f4').reshape(480, 864, 2)
    return pixels, flow, torch.from_numpy(pixels).to('xpu'), torch.from_numpy(flow).to('xpu')

@contextmanager
def installed(name):
    with providers[name].installed(), schedulers[name].installed(), layouts[name].installed(), adapters[name].installed():
        with pairs[name].installed(), c32[name].installed(), fronts[name].installed():
            with splits[name].installed():
                with (vits[name].installed() if name == 'projection' else nullcontext()):
                    yield

try:
    torch.set_num_threads(2)
    for name in VARIANTS:
        model = MotionNR.from_assets(EXACT / 'model-assets/sf-v2/WEIGHTS_HT.bin', EXACT / 'model-assets/noise-sm89-v2', EXACT / 'model-assets/sigmoid-sm89-v1').to('xpu').eval()
        if models:
            report['shared_constants'] = share_identical_buffers(model, models['previous'])
        models[name] = model
        providers[name] = StridedMatrices()
        providers[name].select(MODE)
        adapters[name] = GraphFront(model, arithmetic=providers[name])
        schedulers[name] = FusedSwin(providers[name])
        layouts[name] = QKVHeadLayout(model, providers[name])
        pairs[name] = FusedPairs(model, providers[name])
        c32[name] = FusedC32(model, providers[name], bm=32, warps=4, stages=1)
        splits[name] = FusedSplit(model, providers[name])
        vits[name] = FusedVitProjection(model, providers[name])
        fronts[name] = FusedFront(model)
    report['shared_transient_pool'] = share_before_capture(adapters.values())
    with torch.inference_mode():
        for name, model in models.items():
            if MODE.startswith('int8'):
                providers[name].static_cache.prepack(model)
            with installed(name):
                for i in (0, 1):
                    _, _, rgb, motion = tensors(i)
                    with use_arithmetic_backend('triton'):
                        value = model(rgb, motion, reset=i == 0)
                    assert value.cpu().numpy().tobytes() == expected[i][0].tobytes()
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
                        with use_arithmetic_backend('triton') as dispatch:
                            value = model(rgb, motion, reset=spec['reset'])
                            torch.xpu.synchronize()
                        seconds = time.perf_counter() - started
                    actual, private = value.cpu().numpy(), model._previous.cpu().numpy()
                    target, meta, expected_dispatch = expected[i]
                    effective = dict(dispatch)
                    assert effective.pop('xpu_graph_replay') == 1
                    for key, count in adapter.last_entry.dispatch.items():
                        if key != 'backend':
                            effective[key] = effective.get(key, 0) + count
                    equal = actual.tobytes() == target.tobytes()
                    row = dict(round=repetition, frame=i, reset=spec['reset'], order=order, seconds=seconds,
                               byte_equal_prior=equal, output=meta if equal else arrays.save(actual),
                               private_byte_equal_output=private.tobytes() == actual.tobytes(), next_seed=model.next_seed,
                               effective_dispatch=effective, runtime_dispatch=dict(dispatch), graph_replays=adapter.replays)
                    report['runs'][name].append(row)
                    save()
                    assert equal and row['private_byte_equal_output'] and np.isfinite(actual).all()
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
            vits[name].calls = 0
            c32[name].calls = {}
            with installed(name), use_arithmetic_backend('triton'):
                value = model(rgb, motion, reset=False, progress=marks.append)
            assert value.cpu().numpy().tobytes() == expected[1][0].tobytes() and model.next_seed == 2 and adapter.replays == before
            assert marks == ['pre', 'encoder C32', 'encoder C64', 'encoder C128', 'encoder C256', 'encoder C512', 'ViT', 'decoder C512', 'decoder C256', 'decoder C128', 'decoder C64', 'decoder C32', 'RGB']
            report['progress_fallback'].append(dict(name=name, marks=marks, byte_equal=True, matrix_calls=dict(provider.calls), fused_pairs=dict(pairs[name].calls), fused_c32=dict(c32[name].calls), swin_calls=dict(schedulers[name].counts)))
        means = {name: statistics.mean(row['seconds'] for row in rows) for name, rows in report['runs'].items()}
        assert vits['projection'].calls == 16 and vits['previous'].calls == 0
        report.update(passed=True, vit_projection_calls={name: v.calls for name, v in vits.items()}, matching_mean_seconds=means, speedup={name: means['previous'] / means[name] for name in VARIANTS[1:]},
                      round_mean_seconds={name: [statistics.mean(row['seconds'] for row in rows if row['round'] == i) for i in range(3)] for name, rows in report['runs'].items()},
                      full_outputs_verified=78, held_outputs_survive_replay=True, caller_ownership_guards_passed=True,
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
