"""Paired full480 ablation: expanded matrix fusion and window-group synchronization."""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=DREF/'results/fused-full-480-v2'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1')
sys.path[:0]=[str(R),str(ROOT/'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js,sha
gate=authenticate_main()
primitive_path=DREF/'experimental/fused-shapes-v2/validation.json'
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
from fused_cached_matrices_v2 import FusedCachedMatrices
from swin_scheduling_v1 import SwinScheduling
import immutable_artifacts_v1 as artifacts
inputs=R/'inputs/flow-full-864x480-v2';manifest=js(inputs/'manifest.json')
paths=[Path(__file__),HERE/'Run-FusedFullV2.cmd',HERE/'swin_scheduling_v1.py',HERE/'fused_cached_matrices_v2.py',HERE/'fused_cached_matrices_v1.py',HERE/'fused_activation_int8_v1.py',HERE/'static_weight_cache_v2.py',HERE/'static_weight_cache_v1.py',HERE/'fast_matrices_v3.py',HERE/'immutable_artifacts_v1.py',primitive_path,prior_path,inputs/'manifest.json',*sorted((ROOT/'backend/nr_backend').glob('*.py'))]
frozen={str(p):sha(p) for p in paths}
for p in (ROOT/'backend/nr_backend').glob('*.py'):assert p.read_bytes()==(EXACT/'backend/nr_backend'/p.name).read_bytes()
OUT.mkdir()
modes=('int8_fused','int8_fused_v2','int8_fused_v2_relaxed','fp16_xmx','fp16_xmx_relaxed')
report=dict(scope=__doc__,sources=frozen,passed=False,exact_gate=gate,runs={mode:[] for mode in modes},complete_migration=False,changes_quality=False,timing_scope='Five independent resident models, each reset+nonreset prewarmed; two complete13-frame rounds, rotated/reversed order. Model plus synchronization only; exclude uploads, IO, checking and JIT. No concurrent video encoder. Compare modes from this run only.')
models={};experiments={mode:FusedCachedMatrices() for mode in modes}
schedulers={mode:SwinScheduling(enabled=mode.endswith('_relaxed')) for mode in modes}
expected={}
for mode in modes:
    old_mode='int8_dense' if mode.startswith('int8') else 'fp16_xmx'
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
            experiment.select(mode.removesuffix('_relaxed'))
            with experiment.installed(),schedulers[mode].installed():
                for i in (0,1):
                    _,_,rgb,motion=tensors(i)
                    with use_arithmetic_backend('triton'):value=model(rgb,motion,reset=i==0)
                    torch.xpu.synchronize()
            model.reset();print('Prewarmed '+mode,flush=True)
        for repetition in range(2):
            for i,spec in enumerate(manifest['frames']):
                pixels,flow,rgb,motion=tensors(i)
                offset=(i+repetition)%len(modes);order=list(modes[offset:]+modes[:offset])
                if repetition:order.reverse()
                for mode in order:
                    model=models[mode];experiment=experiments[mode];experiment.select(mode.removesuffix('_relaxed'))
                    torch.xpu.synchronize();torch.xpu.reset_peak_memory_stats()
                    memory_before=torch.xpu.memory_allocated()
                    with experiment.installed(),schedulers[mode].installed():
                        torch.xpu.synchronize();started=time.perf_counter()
                        with use_arithmetic_backend('triton') as dispatch:
                            value=model(rgb,motion,reset=spec['reset']);torch.xpu.synchronize()
                        seconds=time.perf_counter()-started
                    actual=value.cpu().numpy();private=model._previous.cpu().numpy()
                    target,meta,old=expected[mode][i];equal=actual.tobytes()==target.tobytes()
                    row=dict(round=repetition,frame=i,mode=mode,reset=spec['reset'],seconds=seconds,execution_order=order,byte_equal_prior=equal,output=meta if equal else artifacts.array(actual),private_byte_equal_output=private.tobytes()==actual.tobytes(),next_seed=model.next_seed,dispatches=dict(dispatch),matrix_calls=experiment.calls.copy(),matrix_shapes=experiment.shapes.copy(),cache_counts=experiment.static_cache.counts.copy(),scheduling=schedulers[mode].counts.copy(),allocated_before=memory_before,peak_allocated=torch.xpu.max_memory_allocated(),reserved_after=torch.xpu.memory_reserved())
                    report['runs'][mode].append(row)
                    assert actual.dtype==np.dtype('f2') and equal and row['private_byte_equal_output'],(mode,i)
                    assert dict(dispatch)==old['dispatches'] and model.next_seed==(1 if spec['reset'] else i+1)
                    if mode.startswith('int8'):assert experiment.calls.get('packed_weight_hit')==experiment.calls['int8_dense'] and not experiment.calls.get('packed_weight_miss')
                    if mode.startswith('int8'):assert experiment.calls.get('int8_fused',0)>0
                    if mode.endswith('_relaxed'):assert schedulers[mode].counts.get('groups')==479 and not schedulers[mode].counts.get('group_fences')
                    value.zero_();assert model._previous.cpu().numpy().tobytes()==private.tobytes()
                assert rgb.cpu().numpy().tobytes()==pixels.tobytes() and motion.cpu().numpy().tobytes()==flow.tobytes()
                save();print(json.dumps(dict(round=repetition,frame=i,seconds={mode:report['runs'][mode][-1]['seconds'] for mode in modes},fused_calls={mode:experiments[mode].calls.get('int8_fused') for mode in modes if mode.startswith('int8')},all_equal=True)),flush=True)
        means={mode:statistics.mean(row['seconds'] for row in rows) for mode,rows in report['runs'].items()}
        report.update(passed=True,all_outputs_byte_equal_prior=True,caller_ownership_guards_passed=True,matching_mean_seconds=means,expanded_fusion_speedup=means['int8_fused']/means['int8_fused_v2'],scheduling_speedup={'int8':means['int8_fused_v2']/means['int8_fused_v2_relaxed'],'fp16':means['fp16_xmx']/means['fp16_xmx_relaxed']},peak_allocated_bytes={mode:max(row['peak_allocated'] for row in report['runs'][mode]) for mode in modes},round_mean_seconds={mode:[statistics.mean(row['seconds'] for row in report['runs'][mode] if row['round']==r) for r in range(2)] for mode in modes})
except Exception as error:
    report['error']=repr(error);report['traceback']=traceback.format_exc();raise
finally:
    assert all(sha(Path(p))==h for p,h in frozen.items());authenticate_main();save()
print(json.dumps({k:report[k] for k in ('passed','matching_mean_seconds','expanded_fusion_speedup','scheduling_speedup','peak_allocated_bytes')}),flush=True)
