"""Two prewarmed rounds of four independent 480p streams; cache-only byte check."""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
EXACT=ROOT.parent/'nr-b580'
R=EXACT/'reference'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=DREF/'results/static-cache-full-480-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1')
sys.path.insert(0,str(R))
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js,sha
gate=authenticate_main()
prior_path=DREF/'results/fast-precision-864x480-v1/validation.json'
prior=js(prior_path)
assert prior['passed'] and sha(prior_path)=='7a43ff73d452cabe3e68ea1a15db1537417384a57c9a2443fcb16b095d5a5110'
primitive_path=DREF/'experimental/static-weight-cache-v1/primitive.json'
primitive=js(primitive_path)
assert primitive['passed']
for evidence in (prior,primitive):
    assert all(sha(Path(name))==digest for name,digest in evidence['sources'].items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:queue=json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
from PIL import Image
import torch
sys.path.insert(0,str(ROOT/'backend'))
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from static_weight_cache_v1 import CachedFastMatrices
import immutable_artifacts_v1 as artifacts
inputs=R/'inputs/flow-full-864x480-v2'
manifest=js(inputs/'manifest.json')
frozen={str(p):sha(p) for p in [Path(__file__),HERE/'Run-StaticCacheFullV1.cmd',HERE/'static_weight_cache_v1.py',HERE/'fast_matrices_v3.py',HERE/'immutable_artifacts_v1.py',prior_path,primitive_path,inputs/'manifest.json',*sorted((ROOT/'backend/nr_backend').glob('*.py'))]}
for p in (ROOT/'backend/nr_backend').glob('*.py'):assert p.read_bytes()==(EXACT/'backend/nr_backend'/p.name).read_bytes()
OUT.mkdir()
modes=('baseline','fp16_xmx','int8_dense','int8_cached')
report=dict(scope=__doc__,sources=frozen,exact_gate=gate,prior_report=str(prior_path),warmup=[],runs={mode:[] for mode in modes},passed=False,complete_migration=False,timing_scope='4 independent resident models, reset+nonreset prewarm before 2 full13-frame rounds. Rotate/reverse mode order. Timing covers model plus sync; no upload, IO or validation. Each round starts by resetting its own history. No resolution reduction.')
experiment=CachedFastMatrices()
models={}
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
def tensors(i):
    spec=manifest['frames'][i]
    rgb=np.asarray(Image.open(inputs/spec['file']).convert('RGB'),dtype='f4')/255
    motion=np.fromfile(inputs/spec['motion_file'],'<f4').reshape(480,864,2)
    return rgb,motion,torch.from_numpy(rgb).to('xpu'),torch.from_numpy(motion).to('xpu')
try:
    torch.set_num_threads(2)
    # Normal buffers provide working mutation counters; no immutability override.
    for mode in modes:
        models[mode]=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    with torch.inference_mode(),experiment.installed():
        started=time.perf_counter();experiment.prepack(models['int8_dense'])
        report['old_prepack_seconds']=time.perf_counter()-started
        started=time.perf_counter();experiment.static_cache.prepack(models['int8_cached'])
        report['view_prepack_seconds']=time.perf_counter()-started
        report['view_cache_entries']=len(experiment.static_cache.entries)
        report['view_cache_bytes']=experiment.static_cache.packed_bytes
        for mode in modes:
            experiment.select(mode)
            for i in (0,1):
                _,_,rgb,motion=tensors(i)
                with use_arithmetic_backend('triton'):value=models[mode](rgb,motion,reset=i==0)
                torch.xpu.synchronize()
                report['warmup'].append(dict(mode=mode,frame=i))
            models[mode].reset()
            print('Prewarmed reset+nonreset '+mode,flush=True)
        for repetition in range(2):
            for i,spec in enumerate(manifest['frames']):
                pixels,flow,rgb,motion=tensors(i)
                offset=(i+repetition)%4
                order=list(modes[offset:]+modes[:offset])
                if repetition:order.reverse()
                for mode in order:
                    experiment.select(mode)
                    model=models[mode]
                    torch.xpu.synchronize();started=time.perf_counter()
                    with use_arithmetic_backend('triton') as dispatch:
                        value=model(rgb,motion,reset=spec['reset'])
                        torch.xpu.synchronize()
                    seconds=time.perf_counter()-started
                    actual=value.cpu().float().numpy()
                    private=model._previous.cpu().numpy()
                    old_mode='int8_dense' if mode=='int8_cached' else mode
                    expected_row=prior['runs'][old_mode][i]
                    meta=expected_row['actual']
                    assert sha(Path(meta['path']))==meta['sha256']
                    expected=np.load(meta['path'],allow_pickle=False).astype('f4')
                    equal=actual.tobytes()==expected.tobytes()
                    row=dict(round=repetition,frame=i,reset=spec['reset'],mode=mode,execution_order=order,seconds=seconds,byte_equal_prior=equal,rgb_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),actual=artifacts.array(actual.astype('f2')),private=artifacts.array(private),matrix_calls=experiment.calls.copy(),cache_counts=experiment.static_cache.counts.copy(),next_seed=model.next_seed,dispatches=dict(dispatch))
                    report['runs'][mode].append(row)
                    assert equal and private.tobytes()==expected.astype('f2').tobytes()
                    assert dict(dispatch)==expected_row['dispatches']
                    if mode=='int8_cached':assert experiment.calls.get('packed_weight_hit')==experiment.calls['int8_dense'] and not experiment.calls.get('packed_weight_miss')
                    assert model.next_seed==(1 if spec['reset'] else i+1)
                    value.zero_();assert model._previous.cpu().numpy().tobytes()==private.tobytes()
                assert rgb.cpu().numpy().tobytes()==pixels.tobytes() and motion.cpu().numpy().tobytes()==flow.tobytes()
                save()
                print(json.dumps(dict(round=repetition,frame=i,seconds={mode:report['runs'][mode][-1]['seconds'] for mode in modes},all_equal=True,cache=report['runs']['int8_cached'][-1]['cache_counts'])),flush=True)
        means={mode:statistics.mean(row['seconds'] for row in report['runs'][mode]) for mode in modes}
        report.update(passed=True,all_byte_equal_prior=True,caller_ownership_guards_passed=True,matching_mean_seconds=means,cache_speedup=means['int8_dense']/means['int8_cached'],speedup_vs_exact={mode:means['baseline']/means[mode] for mode in modes},round_mean_seconds={mode:[statistics.mean(row['seconds'] for row in report['runs'][mode] if row['round']==r) for r in range(2)] for mode in modes})
except Exception as error:
    report['error']=repr(error);report['traceback']=traceback.format_exc();raise
finally:
    assert all(sha(Path(p))==digest for p,digest in frozen.items())
    authenticate_main();save()
print(json.dumps({key:report[key] for key in ('passed','matching_mean_seconds','cache_speedup','speedup_vs_exact')}),flush=True)
