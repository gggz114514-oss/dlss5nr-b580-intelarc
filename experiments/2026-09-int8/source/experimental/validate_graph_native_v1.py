"""Paired eager/graph streams against authenticated complete native RGB fixtures."""
import argparse, hashlib, json, math, os, statistics, sys, time, traceback, urllib.request
from contextlib import nullcontext
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
EXACT = ROOT.parent / 'nr-b580'
R = EXACT / 'reference'
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
p = argparse.ArgumentParser()
p.add_argument('--dimension', choices=['864x480', '1920x1080'], required=True)
p.add_argument('--mode', choices=['baseline', 'fp16_xmx', 'int8_fused_v2'], required=True)
a = p.parse_args()
w, h = map(int, a.dimension.split('x'))
OUT = DREF / f'results/graph-native-{a.dimension}-{a.mode}-v1'
assert not OUT.exists(), 'An existing experiment is immutable'
os.environ['TRITON_CACHE_DIR'] = str(DREF / 'triton-cache-sm89-v1')
sys.path[:0] = [str(R), str(ROOT / 'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import authenticated, js, sha
gate = authenticate_main()
assert {x.name: sha(x) for x in (ROOT / 'backend/nr_backend').glob('*.py')} == {
    x.name: sha(x) for x in (EXACT / 'backend/nr_backend').glob('*.py')}
version = 2 if a.dimension == '864x480' else 3
inputs = R / f'inputs/flow-full-{a.dimension}-v{version}'
manifest_path = inputs / 'manifest.json'
manifest = js(manifest_path)
audit_path = DREF / f'attention-weights-main-{a.dimension}-native-audit-v1.json'
audit = js(audit_path)
assert audit['all_native_hashes_verified'] and audit['independent_native_repeats_byte_equal']
assert audit['validation_script_sha256'] == sha(R / 'validate_attention_weights_main_motion_v1.py')
assert audit['input_manifest_sha256'] == sha(manifest_path)
for name, digest in audit['native_metadata'].items():
    assert sha(EXACT / name) == digest
metadata_path = R / f'4060-real-flow-{a.dimension}-sequence-v{version}.json'
native = authenticated(js(metadata_path))
assert len(manifest['frames']) == len(audit['frames']) == 13
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']

import numpy as np
from PIL import Image
import torch
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from fused_cached_matrices_v2 import FusedCachedMatrices
from swin_scheduling_v1 import SwinScheduling
from graph_front_v3 import GraphFront
import compressed_arrays_v1 as compressed

names = ['graph_front_v3.py', 'graph_front_v2.py', 'graph_front_v1.py', 'capture_body_v1.py',
         'fused_cached_matrices_v2.py', 'fused_cached_matrices_v1.py', 'fused_activation_int8_v1.py',
         'static_weight_cache_v2.py', 'static_weight_cache_v1.py', 'fast_matrices_v3.py',
         'swin_scheduling_v1.py', 'compressed_arrays_v1.py', 'immutable_artifacts_v1.py',
         'Run-GraphNativeV1.cmd']
paths = [Path(__file__), manifest_path, audit_path, metadata_path,
         *[HERE / n for n in names], *sorted((ROOT / 'backend/nr_backend').glob('*.py'))]
sources = {str(path): sha(path) for path in paths}
OUT.mkdir()
report = dict(scope=__doc__, dimension=a.dimension, mode=a.mode, sources=sources,
              exact_gate=gate, passed=False, complete_migration=False,
              fixture_scope=manifest['scope'], runs={'eager': [], 'graph': []}, warmup=[],
              history='Each model keeps its own history. Native output is comparison only.',
              timing_scope='Two resident independent models, two complete 13-frame rounds with alternating/reversed order after reset+temporal warmup/capture. Full model and synchronization, including dynamic front, copies, graph guards and private history. Excludes upload, JIT/capture, validation and IO.',
              dispatch_scope='Graph effective dispatch combines runtime counters with the captured body template; this is not newly observed kernel counts on replay.',
              visual_review='waived_if_native_bytes_equal' if a.mode == 'baseline' else 'pending_for_new_1080p_output')
models = {}
experiments = {}
schedulers = {}
adapter = None
held = {}
targets = {}

def rawsha(value):
    return hashlib.sha256(value.tobytes()).hexdigest()

def save():
    (OUT / 'validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8', newline='\n')

def tensors(i):
    spec = manifest['frames'][i]
    assert sha(inputs / spec['file']) == spec['sha256']
    assert sha(inputs / spec['motion_file']) == spec['motion_sha256']
    pixels = np.asarray(Image.open(inputs / spec['file']).convert('RGB'), dtype='f4') / 255
    flow = np.fromfile(inputs / spec['motion_file'], '<f4').reshape(h, w, 2)
    assert rawsha(pixels) == audit['frames'][i]['input_rgb_sha256']
    assert np.isfinite(flow).all() and float(np.abs(flow).max()) <= 65504
    assert rawsha(flow.astype('<f2').astype('<f4')) == spec['effective_half_motion_sha256']
    return pixels, flow, torch.from_numpy(pixels).to('xpu'), torch.from_numpy(flow).to('xpu')

def native_target(i):
    path = native / f"{manifest['frames'][i]['file']}_output.rgba32f.bin"
    target = np.fromfile(path, '<f4').reshape(h, w, 4)[..., :3].copy()
    assert rawsha(target) == audit['frames'][i]['native_rgb_sha256']
    assert np.isfinite(target).all() and target.astype('<f2').astype('<f4').tobytes() == target.tobytes()
    return target, dict(format='native-rgba32f-select-rgb-to-f16', path=str(path), sha256=sha(path),
                        shape=[h, w, 3], dtype='<f2', raw_sha256=rawsha(target.astype('<f2')))

def installed(mode):
    return adapter.installed() if mode == 'graph' else nullcontext()

try:
    torch.set_num_threads(2)
    for mode in ('eager', 'graph'):
        model = MotionNR.from_assets(EXACT / 'model-assets/sf-v2/WEIGHTS_HT.bin',
                                    EXACT / 'model-assets/noise-sm89-v2',
                                    EXACT / 'model-assets/sigmoid-sm89-v1').to('xpu').eval()
        models[mode] = model
        experiment = FusedCachedMatrices()
        experiment.select(a.mode)
        experiments[mode] = experiment
        schedulers[mode] = SwinScheduling(enabled=True)
        if mode == 'graph':
            adapter = GraphFront(model, arithmetic=experiment)
    with torch.inference_mode():
        for mode, model in models.items():
            experiment = experiments[mode]
            if a.mode.startswith('int8'):
                experiment.static_cache.prepack(model)
            with experiment.installed(), schedulers[mode].installed(), installed(mode):
                for i in (0, 1):
                    _, _, rgb, motion = tensors(i)
                    started = time.perf_counter()
                    with use_arithmetic_backend('triton'):
                        value = model(rgb, motion, reset=i == 0)
                    torch.xpu.synchronize()
                    report['warmup'].append(dict(mode=mode, frame=i, seconds=time.perf_counter() - started))
                    print(json.dumps(dict(warmup=report['warmup'][-1])), flush=True)
            model.reset()
        for repetition in range(2):
            for i, spec in enumerate(manifest['frames']):
                pixels, flow, rgb, motion = tensors(i)
                target, native_meta = native_target(i)
                order = ['eager', 'graph'] if (i + repetition) % 2 == 0 else ['graph', 'eager']
                rows = {}
                for mode in order:
                    model = models[mode]
                    experiment = experiments[mode]
                    experiment.select(a.mode)
                    torch.xpu.synchronize()
                    torch.xpu.reset_peak_memory_stats()
                    with experiment.installed(), schedulers[mode].installed(), installed(mode):
                        started = time.perf_counter()
                        with use_arithmetic_backend('triton') as dispatch:
                            value = model(rgb, motion, reset=spec['reset'])
                            torch.xpu.synchronize()
                        seconds = time.perf_counter() - started
                    actual = value.cpu().numpy()
                    private = model._previous.cpu().numpy()
                    actual32 = actual.astype('<f4')
                    native_equal = actual32.tobytes() == target.tobytes()
                    effective = dict(dispatch)
                    if mode == 'graph':
                        assert effective.pop('xpu_graph_replay') == 1
                        for k, v in adapter.last_entry.dispatch.items():
                            if k != 'backend':
                                effective[k] = effective.get(k, 0) + v
                    delta = actual32.astype('f8') - target.astype('f8')
                    mse = float(np.square(delta).mean())
                    if i not in targets:
                        meta = native_meta if native_equal else compressed.save(actual)
                        targets[i] = actual.copy(), meta
                    expected, meta = targets[i]
                    equal = actual.tobytes() == expected.tobytes()
                    row = dict(round=repetition, frame=i, reset=spec['reset'], execution_order=order,
                               seconds=seconds, next_seed=model.next_seed, native_byte_equal=native_equal,
                               pair_and_repeat_byte_equal=equal, private_byte_equal=private.tobytes() == actual.tobytes(),
                               output=meta if equal else compressed.save(actual), native=native_meta,
                               rgb32_sha256=rawsha(actual32), mse=mse,
                               psnr_db=None if mse == 0 else -10 * math.log10(mse),
                               max_abs=float(np.abs(delta).max()), finite=bool(np.isfinite(actual).all()),
                               runtime_dispatch=dict(dispatch), effective_dispatch=effective,
                               peak_allocated_bytes=torch.xpu.max_memory_allocated())
                    report['runs'][mode].append(row)
                    rows[mode] = row
                    save()
                    assert row['finite'] and equal and row['private_byte_equal'], (mode, i, row)
                    assert model.next_seed == (1 if spec['reset'] else i + 1)
                    if a.mode == 'baseline':
                        assert native_equal, ('Exact graph/eager differs from full native RGB', mode, i)
                    if mode not in held:
                        held[mode] = value, actual.tobytes()
                    else:
                        value.zero_()
                        assert model._previous.cpu().numpy().tobytes() == private.tobytes()
                assert rows['eager']['effective_dispatch'] == rows['graph']['effective_dispatch']
                for old_value, old_bytes in held.values():
                    assert old_value.cpu().numpy().tobytes() == old_bytes
                assert rgb.cpu().numpy().tobytes() == pixels.tobytes()
                assert motion.cpu().numpy().tobytes() == flow.tobytes()
                print(json.dumps(dict(round=repetition, frame=i, mode=a.mode, dimension=a.dimension,
                                      seconds={m: r['seconds'] for m, r in rows.items()},
                                      pair_equal=True, native_equal=rows['graph']['native_byte_equal'])), flush=True)
        assert targets[0][0].tobytes() == targets[12][0].tobytes()
        # Rejected inputs must leave the graph session's private state untouched.
        model = models['graph']
        seed, previous = model.next_seed, model._previous
        history = previous.cpu().numpy().tobytes()
        for bad_value in (float('nan'), float('inf'), float('-inf'), 65505., -65505.):
            bad = motion.clone()
            bad[0, 0, 0] = bad_value
            with experiments['graph'].installed(), schedulers['graph'].installed(), adapter.installed(), use_arithmetic_backend('triton'):
                try:
                    model(rgb, bad, reset=True)
                except ValueError as error:
                    assert 'FP16 texture range' in str(error)
                else:
                    raise AssertionError('Invalid motion accepted')
            assert model.next_seed == seed and model._previous is previous
            assert previous.cpu().numpy().tobytes() == history
        report.update(graphs=adapter.metadata(), owned_packed_bytes=adapter.owned_packed_bytes,
                      held_output_survives=True, caller_ownership_passed=True,
                      invalid_motion_preserves_state=True, reset_reproduces_frame0=True)
    adapter.close()
    assert all(sha(Path(path)) == digest for path, digest in sources.items())
    authenticate_main()
    means = {mode: statistics.mean(r['seconds'] for r in rows) for mode, rows in report['runs'].items()}
    report.update(passed=True, matching_mean_seconds=means, graph_speedup=means['eager'] / means['graph'],
                  round_mean_seconds={m: [statistics.mean(r['seconds'] for r in rows if r['round'] == j)
                                         for j in range(2)] for m, rows in report['runs'].items()},
                  all_pairs_and_repeats_byte_equal=True,
                  all_native_byte_equal=all(r['native_byte_equal'] for rows in report['runs'].values() for r in rows))
except Exception as error:
    report.update(passed=False, error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    if adapter is not None:
        adapter.close()
    save()
print(json.dumps({k: report[k] for k in ('passed', 'matching_mean_seconds', 'graph_speedup', 'all_native_byte_equal')}), flush=True)
