"""Paired prewarmed full480 streams: existing cached INT8, fused INT8 and FP16."""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=DREF/'results/fused-full-480-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1')
sys.path[:0]=[str(R),str(ROOT/'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js,sha
gate=authenticate_main()
primitive_path=DREF/'experimental/fused-activation-int8-v1/validation.json'
primitive=js(primitive_path);assert primitive['passed']
assert all(sha(Path(p))==h for p,h in primitive['sources'].items())
prior_path=DREF/'results/fast-precision-864x480-v1/validation.json'
assert sha(prior_path)=='7a43ff73d452cabe3e68ea1a15db1537417384a57c9a2443fcb16b095d5a5110'
prior=js(prior_path);assert prior['passed']
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:queue=json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
from PIL import Image
import torch
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from fused_cached_matrices_v1 import FusedCachedMatrices
import immutable_artifacts_v1 as artifacts
inputs=R/'inputs/flow-full-864x480-v2';manifest=js(inputs/'manifest.json')
paths=[Path(__file__),HERE/'Run-FusedFullV1.cmd',HERE/'fused_cached_matrices_v1.py',HERE/'fused_activation_int8_v1.py',HERE/'static_weight_cache_v2.py',HERE/'static_weight_cache_v1.py',HERE/'fast_matrices_v3.py',HERE/'immutable_artifacts_v1.py',primitive_path,prior_path,inputs/'manifest.json',*sorted((ROOT/'backend/nr_backend').glob('*.py'))]
frozen={str(p):sha(p) for p in paths}
for p in (ROOT/'backend/nr_backend').glob('*.py'):assert p.read_bytes()==(EXACT/'backend/nr_backend'/p.name).read_bytes()
OUT.mkdir()
modes=('int8_cached','int8_fused','fp16_xmx')
report=dict(scope=__doc__,sources=frozen,passed=False,exact_gate=gate,runs={mode:[] for mode in modes},complete_migration=False,changes_quality=False,timing_scope='Three independent resident models, each reset+nonreset prewarmed; two complete13-frame rounds, rotated/reversed order. Model plus synchronization only; exclude uploads, IO, checking and JIT. No concurrent video encoder. Compare modes from this run only.')
models={};experiments={mode:FusedCachedMatrices() for mode in modes}
expected={}
for mode in modes:
    old_mode='int8_dense' if mode.startswith('int8') else mode
    expected[mode]=[]
    for old in prior['runs'][old_mode]:
        meta=old['actual'];assert sha(Path(meta['path']))==meta['sha256']
        expected[mode].append((np.load(meta['path'],allow_pickle=False).astype('f2'),meta,old))
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
def tensors(i):
    spec=manifest['frames'][i]
    rgb=np.asarray(Image.open(inputs/spec['file']).convert('RGB'),dtype='f4')/255
    motion=np.fromfile(inputs/spec['motion_file'],'<f4').reshape(480,864,2)
    return rgb,motion,torch.from_numpy(rgb).to('xpu'),torch.from_numpy(motion).to('xpu')
try:
    torch.set_num_threads(2)
    for mode in modes:
        models[mode]=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    with torch.inference_mode():
        for mode in modes:
            model=models[mode];experiment=experiments[mode]
            if mode.startswith('int8'):experiment.static_cache.prepack(model)
            experiment.select(mode)
            with experiment.installed():
                for i in (0,1):
                    _,_,rgb,motion=tensors(i)
                    with use_arithmetic_backend('triton'):value=model(rgb,motion,reset=i==0)
                    torch.xpu.synchronize()
            model.reset();print('Prewarmed '+mode,flush=True)
        for repetition in range(2):
            for i,spec in enumerate(manifest['frames']):
                pixels,flow,rgb,motion=tensors(i)
                offset=(i+repetition)%3;order=list(modes[offset:]+modes[:offset])
                if repetition:order.reverse()
                for mode in order:
                    model=models[mode];experiment=experiments[mode];experiment.select(mode)
                    with experiment.installed():
                        torch.xpu.synchronize();started=time.perf_counter()
                        with use_arithmetic_backend('triton') as dispatch:
                            value=model(rgb,motion,reset=spec['reset']);torch.xpu.synchronize()
                        seconds=time.perf_counter()-started
                    actual=value.cpu().numpy();private=model._previous.cpu().numpy()
                    target,meta,old=expected[mode][i];equal=actual.tobytes()==target.tobytes()
                    row=dict(round=repetition,frame=i,mode=mode,reset=spec['reset'],seconds=seconds,execution_order=order,byte_equal_prior=equal,output=meta if equal else artifacts.array(actual),private_byte_equal_output=private.tobytes()==actual.tobytes(),next_seed=model.next_seed,dispatches=dict(dispatch),matrix_calls=experiment.calls.copy(),matrix_shapes=experiment.shapes.copy(),cache_counts=experiment.static_cache.counts.copy())
                    report['runs'][mode].append(row)
                    assert actual.dtype==np.dtype('f2') and equal and row['private_byte_equal_output'],(mode,i)
                    assert dict(dispatch)==old['dispatches'] and model.next_seed==(1 if spec['reset'] else i+1)
                    if mode.startswith('int8'):assert experiment.calls.get('packed_weight_hit')==experiment.calls['int8_dense'] and not experiment.calls.get('packed_weight_miss')
                    if mode=='int8_fused':assert experiment.calls.get('int8_fused',0)>0
                    value.zero_();assert model._previous.cpu().numpy().tobytes()==private.tobytes()
                assert rgb.cpu().numpy().tobytes()==pixels.tobytes() and motion.cpu().numpy().tobytes()==flow.tobytes()
                save();print(json.dumps(dict(round=repetition,frame=i,seconds={mode:report['runs'][mode][-1]['seconds'] for mode in modes},fused_calls=experiments['int8_fused'].calls['int8_fused'],all_equal=True)),flush=True)
        means={mode:statistics.mean(row['seconds'] for row in rows) for mode,rows in report['runs'].items()}
        report.update(passed=True,all_outputs_byte_equal_prior=True,caller_ownership_guards_passed=True,matching_mean_seconds=means,fusion_speedup=means['int8_cached']/means['int8_fused'],fp16_relative_fused=means['int8_fused']/means['fp16_xmx'],round_mean_seconds={mode:[statistics.mean(row['seconds'] for row in report['runs'][mode] if row['round']==r) for r in range(2)] for mode in modes})
except Exception as error:
    report['error']=repr(error);report['traceback']=traceback.format_exc();raise
finally:
    assert all(sha(Path(p))==h for p,h in frozen.items());authenticate_main();save()
print(json.dumps({k:report[k] for k in ('passed','matching_mean_seconds','fusion_speedup')}),flush=True)
