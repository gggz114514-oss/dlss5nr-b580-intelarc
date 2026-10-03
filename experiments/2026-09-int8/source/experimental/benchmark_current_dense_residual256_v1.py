"""Pair the complete native-half/K8 NR256 residual path with measured dense tiles; both use fused ViT QKV and history sampling.

Both variants use the same isolated Triton3.8 toolchain. Validate every full1080
output and low NR output against the frozen approved Triton3.7.2 sequence.

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
VARIANTS = ('previous', 'dense_tiles')
OUT = DREF / 'results/current-dense-residual256-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR'] = str(DREF / 'triton-cache-c32-triton38-v1')
TOOLCHAIN = DREF / 'toolchains/triton-xpu-3.8.0-git1e2d42a0'
sys.path[:0] = [str(TOOLCHAIN / 'site'), str(R), str(ROOT / 'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js, sha
gate = authenticate_main()
provision_path = TOOLCHAIN / 'provision-v1.json'
assert sha(provision_path) == 'e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN / 'site' / p) == h for p,h in js(provision_path)['files'].items())
body_path = DREF / 'experimental/current-dense-tiles-v1/validation.json'
assert sha(body_path) == '8f1eff3e0e058748965210a7eb69019208e445d2395ed86db92406697c394063'
body_record = js(body_path)
assert body_record['passed'] and js(body_path.parent.with_suffix('.log.lease.json'))['returncode'] == 0
assert all(sha(Path(p)) == h for p,h in body_record['sources'].items())
primitive_paths = [DREF / f'experimental/{name}/validation.json' for name in
                   ('fused-branched-pairs-v1', 'fused-c32-mlp-v1', 'fused-swin-core-v2', 'fused-dynamic-front-v1', 'fused-split-ffwd-v1', 'fused-qkv-pack-v1', 'fused-vit-projection-v1', 'fused-vit-projection-v2', 'c32-chunk-pack-v3', 'c32-mlp-lut-v1', 'wide-mlp-lut-v1', 'batched-branches-v1', 'c32-projection-pack-v1', 'native-half-attention-v1')]
for primitive_path in primitive_paths:
    primitive = js(primitive_path)
    assert primitive['passed']
    assert all(sha(Path(path)) == digest for path, digest in primitive['sources'].items())
    assert js(primitive_path.parent.with_suffix('.log.lease.json'))['returncode'] == 0
assert sha(DREF / 'experimental/c32-mlp-lut-v1/validation.json') == 'cff24169c9c037ff243b33205a2f6ec2bc5c3c041e069af33c77c1d582400ea6'
assert sha(DREF / 'experimental/wide-mlp-lut-v1/validation.json') == '3e71f13778ea5e9348c53c950b689e2cc08c5398e0ad4c56414682f7ab0d2679'
assert sha(DREF / 'experimental/batched-branches-v1/validation.json') == '62c7e54e919302d398ff6cd37255e68ac66a6c4a91a4c7c86f4106b810d938ab'
assert sha(DREF / 'experimental/c32-projection-pack-v1/validation.json') == '10e4cbed3bb4d503e99f304c19be4ef663d1ed9d328584452943407b646f7f89'
assert sha(DREF / 'experimental/native-half-attention-v1/validation.json') == '4ee4b5efa75fb8a78310e2824b1631980568c7dd7f128ab8fc8055d635507a1b'
prior_path = DREF / 'results/residual-scale-fp16_xmx-256-v1/validation.json'
assert sha(prior_path) == '056251b4e2a839fcc44845d9646d2cd8ae1e9ed7f99fd1858dc7e3d51e3e50f9'
prior = js(prior_path)
with urllib.request.urlopen('http://127.0.0.1:8188/queue', timeout=5) as response:
    queue = json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
from PIL import Image
import torch
import triton
assert Path(triton.__file__).is_relative_to(TOOLCHAIN / 'site') and triton.__version__.startswith('3.8.0')
from k8_tiled_provider_v1 import K8TiledMatrices
from current_dense_tiled_provider_v1 import CurrentDenseTiledMatrices
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from strided_batched_v2 import StridedMatrices
from batched_branched_mlp_v2 import FusedBatched
from fused_branched_pairs_v2 import FusedPairs as DirectPairs
from fused_branched_pairs_lut_v2 import FusedPairs as LookupPairs
from fused_split_ffwd_v2 import FusedSplit as DirectSplit
from fused_split_ffwd_lut_v2 import FusedSplit as LookupSplit
from fused_head_layout_v1 import HeadLayout
from qkv_head_layout_v1 import QKVHeadLayout
from c32_chunk_layout_v2 import ChunkedHeadLayout
from projected_head_layout_v1 import ProjectedHeadLayout
from native_half_head_layout_v1 import HeadLayout as NativeHalfLayout
from fused_swin_native_half_v1 import FusedSwin as NativeHalfSwin
from fused_vit_projection_v3 import FusedVitProjection
from fused_c32_mlp_v1 import FusedC32 as DirectC32
from fused_c32_mlp_lut_v1 import FusedC32 as LookupC32
from cubic_lut_constant_v1 import register, Constant, TABLE
from cubic_lut_graph_guards_v2 import verify as verify_lut_guards
from serial_graph_workspace_v1 import share_before_capture
from residual_scale_v1 import ResidualScale
from graph_front_v6 import GraphFront
from swin_scheduling_v1 import SwinScheduling
from fused_swin_scheduling_v1 import FusedSwin
from fused_dynamic_front_v1 import FusedFront
from graph_history_warp_v2 import GraphHistoryWarp
from fused_graph_history_warp_v2 import FusedHistoryWarp
from fused_vit_qkv_adapter_v1 import FusedVitQKV
from window_layout_v1 import WindowLayout
from shared_model_buffers_v1 import share_identical_buffers
import compressed_arrays_v1 as arrays
inputs = R / 'inputs/flow-full-1920x1080-v3'
manifest = js(inputs / 'manifest.json')
names = ['fused_c32_projection_pack_v1.py', 'projected_head_layout_v1.py', 'batched_branched_mlp_v1.py', 'batched_branched_mlp_v2.py', 'fused_mlp_pair_lut_v1.py', 'fused_branched_pairs_lut_v1.py', 'fused_branched_pairs_lut_v2.py', 'fused_split_ffwd_lut_v1.py', 'fused_split_ffwd_lut_v2.py', 'cubic_lut_constant_v1.py', 'fused_c32_mlp_lut_v1.py', 'cubic_lut_graph_guards_v2.py', 'fused_c32_qkv_scatter_v2.py', 'c32_chunk_layout_v2.py', 'fused_vit_projection_v3.py', 'fused_vit_projection_v1.py', 'fused_vit_projection_v2.py', 'fused_qkv_pack_v1.py', 'qkv_head_layout_v1.py', 'benchmark_current_dense_residual256_v1.py', 'Run-CurrentDenseResidual256V1.cmd', 'current_dense_tiled_provider_v1.py', 'dense_tiles_v1.py', 'fused_vit_qkv_v1.py', 'fused_vit_qkv_adapter_v1.py', 'fused_square_history_v2.py', 'fused_graph_history_warp_v2.py', 'k8_tiled_provider_v1.py',
         'fused_head_layout_v1.py', 'fused_swin_heads_v1.py', 'fused_split_ffwd_v1.py', 'fused_split_ffwd_v2.py', 'fused_branched_pairs_v2.py', 'fused_branched_pairs_v1.py', 'fused_mlp_pair_v1.py',
         'fused_c32_mlp_v1.py', 'fused_branched_mlp_v1.py', 'serial_graph_workspace_v1.py', 'residual_scale_v1.py',
         'strided_batched_v2.py', 'strided_batched_v1.py', 'window_layout_v1.py',
         'swin_scheduling_v1.py', 'fused_swin_scheduling_v1.py', 'fused_dynamic_front_v1.py', 'graph_history_warp_v1.py', 'graph_history_warp_v2.py', 'fused_swin_core_v2.py', 'capture_body_v1.py', 'fused_cached_matrices_v2.py', 'fused_cached_matrices_v1.py',
         'fused_activation_int8_v1.py', 'fast_matrices_v3.py', 'static_weight_cache_v2.py', 'static_weight_cache_v1.py',
         'shared_model_buffers_v1.py', 'compressed_arrays_v1.py', 'immutable_artifacts_v1.py',
         *[f'graph_front_v{i}.py' for i in range(1, 7)]]
paths = [*[Path(p) for p in js(DREF / 'experimental/native-half-attention-v1/validation.json')['sources']],*[Path(p) for p in js(DREF / 'experimental/c32-projection-pack-v1/validation.json')['sources']], *[Path(p) for p in js(DREF / 'experimental/batched-branches-v1/validation.json')['sources']], *[Path(p) for p in js(DREF / 'experimental/wide-mlp-lut-v1/validation.json')['sources']], TABLE, DREF / 'cubic-fp8-fused-cpu-v3.json', DREF / 'cubic-fp8-fused-xpu-v3.json',
         *[Path(p) for p in js(DREF / 'experimental/c32-mlp-lut-v1/validation.json')['sources']], *(HERE / name for name in names), *primitive_paths, prior_path, inputs / 'manifest.json',
         *sorted((ROOT / 'backend/nr_backend').glob('*.py'))]
paths.extend([provision_path, body_path, *[Path(p) for p in body_record['sources']]])
frozen = {str(path): sha(path) for path in paths}
for path in (ROOT / 'backend/nr_backend').glob('*.py'):
    assert path.read_bytes() == (EXACT / 'backend/nr_backend' / path.name).read_bytes()
OUT.mkdir()
report = dict(scope=__doc__, sources=frozen, exact_gate=gate, mode=MODE, passed=False,
              runs={name: [] for name in VARIANTS}, warmup=[], complete_migration=False,
              timing_scope=__doc__, progress_fallback=[], changes_quality_from_prior_residual=False, new_quality_approved=False, dispatch_semantics='Original logical operations; fused kernels do not issue each as separate GPU work')
models, providers, adapters, schedulers, layouts, held = {}, {}, {}, {}, {}, {}
pairs, c32, fronts, warps, splits = {}, {}, {}, {}, {}
vits = {}
qkvs = {}
report['runtime'] = dict(triton_version=triton.__version__, triton_file=triton.__file__, torch_version=torch.__version__, isolated=True)
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
        with (pairs[name].installed() if True else nullcontext()):
            with (c32[name].installed() if True else nullcontext()):
                with (fronts[name].installed() if True else nullcontext()):
                    with warps[name].installed():
                        with splits[name].installed():
                            with vits[name].installed():
                                with qkvs[name].installed():
                                    yield

try:
    torch.set_num_threads(2)
    scaler = ResidualScale(256)
    for name in VARIANTS:
        model = MotionNR.from_assets(EXACT / 'model-assets/sf-v2/WEIGHTS_HT.bin', EXACT / 'model-assets/noise-sm89-v2', EXACT / 'model-assets/sigmoid-sm89-v1').to('xpu').eval()
        register(model)
        if models:
            report['shared_constants'] = share_identical_buffers(model, models['previous'])
        models[name] = model
        providers[name] = CurrentDenseTiledMatrices(model) if name=='dense_tiles' else K8TiledMatrices(model)
        providers[name].select(MODE)
        adapters[name] = GraphFront(model, arithmetic=providers[name])
        schedulers[name] = NativeHalfSwin(providers[name])
        layouts[name] = NativeHalfLayout(model, providers[name])
        pairs[name] = FusedBatched(model, providers[name], workload='small')
        c32[name] = LookupC32(model, providers[name], bm=32, warps=4, stages=1)
        fronts[name] = FusedFront(model)
        warps[name] = FusedHistoryWarp(model)
        qkvs[name] = FusedVitQKV(model,providers[name])
        splits[name] = DirectSplit(model, providers[name])
        vits[name] = FusedVitProjection(model, providers[name])
    report['shared_transient_pool'] = share_before_capture(adapters.values())
    with torch.inference_mode():
        def check_provider_calls():
            source_path=DREF/'experimental/current-dense-operands-v2/validation.json'
            assert sha(source_path)=='46a16ba64e9f84425928d0a0fd93862e99574910b031ba098c789f0ca69e7290'
            captured=js(source_path)['cases'];selected=[r for r in body_record['cases'] if r['selected']!=r['baseline']]
            buffers=dict(models['dense_tiles'].named_buffers())
            candidate=providers['dense_tiles'];old=providers['previous']
            checks=[]
            def raw(t):return t.cpu().numpy().tobytes()
            for spec in selected:
                src=captured[spec['key']];owner=src['weight_owner_names'][0];w=buffers[owner]
                assert raw(w)==arrays.load(src['operands']['w']).tobytes()
                operands={}
                for name in ('a','initial'):
                    meta=src['operands'][name]
                    if meta is None:operands[name]=None;continue
                    value=arrays.load(meta);desc=src['descriptors'][name]
                    t=torch.empty_strided(tuple(desc['shape']),tuple(desc['stride']),device='xpu',dtype=torch.float16)
                    t.copy_(torch.from_numpy(value.copy()).to('xpu'));operands[name]=t
                before=sum(candidate.tiled_calls.values())
                output=candidate.dense(operands['a'],w,initial=operands['initial'],chunk_k=16)
                assert raw(output)==arrays.load(src['output']).tobytes() and sum(candidate.tiled_calls.values())==before+1
                checks.append(dict(key=spec['key'],owner=owner,full_output_byte_equal=True,tiled=True))
            src=captured[selected[0]['key']];w=buffers[src['weight_owner_names'][0]]
            a=torch.from_numpy(arrays.load(src['operands']['a']).copy()).to('xpu').reshape(-1,w.shape[0])
            backing=torch.empty((a.shape[0],a.shape[1]*2),device='xpu',dtype=torch.float16);backing[:,::2]=a
            fallbacks=[('unowned_weight',a,w.clone(),None),('unknown_rows',a[:-1],w,None),
                       ('strided_input',backing[:,::2],w,None),('float_input',a.float(),w,None),
                       ('new_initial',a,w,torch.ones((a.shape[0],w.shape[1]),device='xpu',dtype=torch.float16))]
            for label,x,weight,initial in fallbacks:
                before=sum(candidate.tiled_calls.values())
                ref=old.dense(x,weight,initial=initial,chunk_k=16)
                output=candidate.dense(x,weight,initial=initial,chunk_k=16)
                assert raw(output)==raw(ref) and output.shape==ref.shape and sum(candidate.tiled_calls.values())==before
                checks.append(dict(fallback=label,full_output_byte_equal=True,tiled=False))
            before=sum(candidate.tiled_calls.values())
            try:
                old.select('int8_dense');candidate.select('int8_dense')
                ref=old.dense(a,w,chunk_k=16);output=candidate.dense(a,w,chunk_k=16)
                assert raw(output)==raw(ref) and sum(candidate.tiled_calls.values())==before
                checks.append(dict(fallback='int8_dense',full_output_byte_equal=True,tiled=False))
            finally:old.select(MODE);candidate.select(MODE)
            candidate.tiled_calls={}
            return checks
        report['dense_provider_checks']=check_provider_calls()
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
            vits[name].calls = 0
            qkvs[name].calls = 0
            if name=='dense_tiles':providers[name].tiled_calls={}
            providers[name].k8_calls = {}
            c32[name].calls = {}
            low_rgb, low_motion = scaler.prepare(rgb, motion)
            with installed(name), use_arithmetic_backend('triton'):
                low_nr = model(low_rgb, low_motion, reset=False, progress=marks.append)
                value = scaler.composite(rgb, low_rgb, low_nr)
            assert value.cpu().numpy().tobytes() == expected[1][0].tobytes() and model.next_seed == 2 and adapter.replays == before
            assert marks == ['pre', 'encoder C32', 'encoder C64', 'encoder C128', 'encoder C256', 'encoder C512', 'ViT', 'decoder C512', 'decoder C256', 'decoder C128', 'decoder C64', 'decoder C32', 'RGB']
            report['progress_fallback'].append(dict(name=name, marks=marks, byte_equal=True, matrix_calls=dict(provider.calls), fused_pairs=dict(pairs[name].calls), fused_c32=dict(c32[name].calls), swin_calls=dict(schedulers[name].counts), front_calls=fronts[name].calls, warp_replays=warps[name].replays))
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
        assert all(v.calls == 16 for v in vits.values())
        assert all(l.calls['c32_chunk_pack'] >= 10 and l.calls['c32']==10 for l in layouts.values())
        assert all(p.k8_calls==dict(pre=1,post=1) for p in providers.values())
        report['k8_calls_after_progress']={n:dict(p.k8_calls) for n,p in providers.items()}
        assert all(w.fused_builds==1 for w in warps.values())
        assert all(q.calls==8 for q in qkvs.values())
        expected_tiles={row['key']:row['count'] for row in body_record['cases'] if row['selected']!=row['baseline']}
        assert providers['dense_tiles'].tiled_calls==expected_tiles and sum(expected_tiles.values())==83
        report['dense_tiles_after_progress']=dict(providers['dense_tiles'].tiled_calls)
        report['vit_qkv_calls_after_progress']={name:q.calls for name,q in qkvs.items()}
        assert warps['dense_tiles'].fused_builds==1
        report['fused_history_builds']=warps['dense_tiles'].fused_builds
        report['model_history_error_guards']=[]
        for name,model in models.items():
            private=model._previous;before_bytes=private.cpu().numpy().tobytes()
            before=(model.next_seed,adapters[name].replays,warps[name].replays)
            with installed(name),use_arithmetic_backend('triton'):
                bad=low_motion.clone();bad[0,0,0]=float('nan')
                try:model(low_rgb,bad,reset=False)
                except ValueError as error:assert 'Motion must be finite' in str(error)
                else:raise AssertionError('Nonfinite motion accepted')
                original_table=model.reciprocal.values
                with torch.inference_mode(False):replacement=original_table.clone()
                model.reciprocal.values=replacement
                try:
                    try:model(low_rgb,low_motion,reset=False)
                    except RuntimeError as error:assert 'reciprocal table changed' in str(error)
                    else:raise AssertionError('Changed reciprocal table accepted')
                finally:model.reciprocal.values=original_table
            assert model._previous is private and private.cpu().numpy().tobytes()==before_bytes
            assert (model.next_seed,adapters[name].replays,warps[name].replays)==before
            report['model_history_error_guards'].append(dict(name=name,invalid_motion_and_table_rejected=True,history_seed_and_replays_unchanged=True))
        # Both variants registered and shared the same extra constant before capture.
        assert getattr(models['previous'], '_cubic_fp8_lut_bits') is getattr(models['dense_tiles'], '_cubic_fp8_lut_bits')
        assert Constant(models['dense_tiles']).require().cpu().numpy().tobytes() == np.load(TABLE, allow_pickle=False).view('i2').tobytes()
        report['lut_bytes_unchanged'] = True
        report['lut_shared_before_capture'] = True
        with installed('dense_tiles'), use_arithmetic_backend('triton'):
            report['lut_graph_guards'] = verify_lut_guards(models['dense_tiles'], adapters['dense_tiles'],
                lambda: models['dense_tiles'](low_rgb, low_motion, reset=False))
        report.update(passed=True, layout_calls_after_progress={name: dict(l.calls) for name, l in layouts.items()}, vit_projection_calls={name: v.calls for name, v in vits.items()}, matching_mean_seconds=means, speedup={name: means['previous'] / means[name] for name in VARIANTS[1:]},
                      round_mean_seconds={name: [statistics.mean(row['seconds'] for row in rows if row['round'] == i) for i in range(3)] for name, rows in report['runs'].items()},
                      full_outputs_verified=78, held_outputs_survive_replay=True, caller_ownership_guards_passed=True,
                      graphs={name: adapter.metadata() for name, adapter in adapters.items()})
except Exception as error:
    report.update(error=repr(error), traceback=traceback.format_exc())
    raise
finally:
    for warp in warps.values():
        warp.close()
    for adapter in adapters.values():
        adapter.close()
    assert all(sha(Path(path)) == digest for path, digest in frozen.items())
    authenticate_main()
    save()
print(json.dumps({key: report[key] for key in ('passed', 'matching_mean_seconds', 'speedup')}), flush=True)
