"""Paired complete480: scheduled baseline, eager queued body, and graphed body."""
from contextlib import nullcontext
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=DREF/'results/graph-full-480-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1');sys.path[:0]=[str(R),str(ROOT/'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js,sha
gate=authenticate_main()
smoke_path=DREF/'results/graph-body-480-v1/validation.json';smoke=js(smoke_path);assert smoke['passed']
assert all(sha(Path(p))==h for p,h in smoke['sources'].items())
prior_path=DREF/'results/fast-precision-864x480-v1/validation.json'
assert sha(prior_path)=='7a43ff73d452cabe3e68ea1a15db1537417384a57c9a2443fcb16b095d5a5110'
prior=js(prior_path)
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:queue=json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
from PIL import Image
import torch
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from fused_cached_matrices_v2 import FusedCachedMatrices
from swin_scheduling_v1 import SwinScheduling
from graph_front_v1 import GraphFront
from queued_front_v1 import QueuedFront
import immutable_artifacts_v1 as artifacts
inputs=R/'inputs/flow-full-864x480-v2';manifest=js(inputs/'manifest.json')
paths=[Path(__file__),HERE/'Run-GraphFullV1.cmd',HERE/'queued_front_v1.py',HERE/'graph_front_v1.py',HERE/'capture_body_v1.py',HERE/'swin_scheduling_v1.py',HERE/'fused_cached_matrices_v2.py',HERE/'fused_cached_matrices_v1.py',HERE/'fused_activation_int8_v1.py',HERE/'static_weight_cache_v2.py',HERE/'static_weight_cache_v1.py',HERE/'fast_matrices_v3.py',HERE/'immutable_artifacts_v1.py',smoke_path,prior_path,inputs/'manifest.json',*sorted((ROOT/'backend/nr_backend').glob('*.py'))]
sources={str(p):sha(p) for p in paths};OUT.mkdir()
modes=tuple(p+s for p in ('fp16_xmx','int8_fused_v2') for s in ('','_queued','_graph'))
precision=lambda mode:'fp16_xmx' if mode.startswith('fp16') else 'int8_fused_v2'
report=dict(scope=__doc__,sources=sources,passed=False,runs={mode:[] for mode in modes},exact_gate=gate,warmup=[],complete_migration=False,timing_scope='Six independent resident models; reset+nonreset prewarm/capture, then two complete13-frame rounds in rotated/reversed order. Full model and sync, INCLUDING dynamic noise/seed/warp/front, graph input copies, immutable-model checks, result clone and private history. Excludes upload, JIT/capture, validation and IO. Queued eager ablation separates fence removal from graph replay.',dispatch_scope='Graph effective dispatch is runtime Python dispatch plus its captured body template, NOT a count newly observed by Python on replay.')
models={};experiments={};schedulers={};adapters={};held={}
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
def tensors(i):
    spec=manifest['frames'][i];rgb=np.asarray(Image.open(inputs/spec['file']).convert('RGB'),dtype='f4')/255
    motion=np.fromfile(inputs/spec['motion_file'],'<f4').reshape(480,864,2)
    return rgb,motion,torch.from_numpy(rgb).to('xpu'),torch.from_numpy(motion).to('xpu')
def installed(mode):return adapters[mode].installed() if mode in adapters else nullcontext()
expected={}
for mode in modes:
    rows=prior['runs']['fp16_xmx' if mode.startswith('fp16') else 'int8_dense']
    expected[mode]=[]
    for row in rows:
        meta=row['actual'];assert sha(Path(meta['path']))==meta['sha256']
        expected[mode].append((np.load(meta['path'],allow_pickle=False).astype('f2'),meta,row))
try:
    torch.set_num_threads(2)
    for mode in modes:
        model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
        models[mode]=model;experiment=FusedCachedMatrices();experiment.select(precision(mode));experiments[mode]=experiment
        schedulers[mode]=SwinScheduling(enabled=True)
        if mode.endswith('_graph'):adapters[mode]=GraphFront(model,arithmetic=experiment)
        elif mode.endswith('_queued'):adapters[mode]=QueuedFront(model)
    with torch.inference_mode():
        for mode in modes:
            model=models[mode];experiment=experiments[mode]
            if mode.startswith('int8'):experiment.static_cache.prepack(model)
            with experiment.installed(),schedulers[mode].installed(),installed(mode):
                for i in (0,1):
                    _,_,rgb,motion=tensors(i)
                    with use_arithmetic_backend('triton'):value=model(rgb,motion,reset=i==0)
                    torch.xpu.synchronize()
            model.reset();report['warmup'].append(dict(mode=mode,graphs=adapters[mode].metadata() if mode.endswith('_graph') else None))
            print('Prewarmed '+mode,flush=True)
        for repetition in range(2):
            for i,spec in enumerate(manifest['frames']):
                pixels,flow,rgb,motion=tensors(i);offset=(i+repetition)%len(modes)
                order=list(modes[offset:]+modes[:offset])
                if repetition:order.reverse()
                for mode in order:
                    model=models[mode];experiment=experiments[mode];experiment.select(precision(mode))
                    torch.xpu.synchronize();torch.xpu.reset_peak_memory_stats()
                    with experiment.installed(),schedulers[mode].installed(),installed(mode):
                        started=time.perf_counter()
                        with use_arithmetic_backend('triton') as dispatch:
                            value=model(rgb,motion,reset=spec['reset']);torch.xpu.synchronize()
                        seconds=time.perf_counter()-started
                    actual=value.cpu().numpy();private=model._previous.cpu().numpy();target,meta,old=expected[mode][i]
                    equal=actual.tobytes()==target.tobytes();effective=dict(dispatch)
                    if mode.endswith('_graph'):
                        assert effective.pop('xpu_graph_replay')==1
                        for k,v in adapters[mode].last_entry.dispatch.items():
                            if k!='backend':effective[k]=effective.get(k,0)+v
                    row=dict(round=repetition,frame=i,mode=mode,reset=spec['reset'],execution_order=order,seconds=seconds,byte_equal_prior=equal,output=meta if equal else artifacts.array(actual),private_byte_equal_output=private.tobytes()==actual.tobytes(),next_seed=model.next_seed,runtime_dispatch=dict(dispatch),effective_dispatch=effective,peak_allocated=torch.xpu.max_memory_allocated(),graph_replays=adapters[mode].replays if mode.endswith('_graph') else None)
                    report['runs'][mode].append(row);assert equal and row['private_byte_equal_output'],(mode,i)
                    assert effective==old['dispatches'] and model.next_seed==(1 if spec['reset'] else i+1)
                    if mode not in held:held[mode]=(value,actual.tobytes())
                    else:value.zero_();assert model._previous.cpu().numpy().tobytes()==private.tobytes()
                for old_value,old_bytes in held.values():assert old_value.cpu().numpy().tobytes()==old_bytes
                assert rgb.cpu().numpy().tobytes()==pixels.tobytes() and motion.cpu().numpy().tobytes()==flow.tobytes()
                save();print(json.dumps(dict(round=repetition,frame=i,seconds={mode:report['runs'][mode][-1]['seconds'] for mode in modes},all_equal=True)),flush=True)
        # A progress callback uses the original eager body and keeps its stage list.
        report['progress_fallback']=[]
        for mode in (m for m in modes if m.endswith('_graph')):
            model=models[mode];experiment=experiments[mode];before=adapters[mode].replays
            _,_,rgb,motion=tensors(1);marks=[]
            with experiment.installed(),schedulers[mode].installed(),installed(mode),use_arithmetic_backend('triton'):
                value=model(rgb,motion,reset=False,progress=marks.append)
            assert value.cpu().numpy().tobytes()==expected[mode][1][0].tobytes() and model.next_seed==2 and adapters[mode].replays==before
            assert marks==['pre','encoder C32','encoder C64','encoder C128','encoder C256','encoder C512','ViT','decoder C512','decoder C256','decoder C128','decoder C64','decoder C32','RGB']
            report['progress_fallback'].append(dict(mode=mode,marks=marks,byte_equal=True))
        means={mode:statistics.mean(r['seconds'] for r in rows) for mode,rows in report['runs'].items()}
        report.update(passed=True,all_outputs_byte_equal_prior=True,held_outputs_survive_replay=True,caller_ownership_guards_passed=True,matching_mean_seconds=means,graph_speedup_vs_scheduled={p:means[p]/means[p+'_graph'] for p in ('fp16_xmx','int8_fused_v2')},graph_speedup_vs_queued={p:means[p+'_queued']/means[p+'_graph'] for p in ('fp16_xmx','int8_fused_v2')},round_mean_seconds={mode:[statistics.mean(r['seconds'] for r in rows if r['round']==repetition) for repetition in range(2)] for mode,rows in report['runs'].items()},graphs={mode:a.metadata() for mode,a in adapters.items() if mode.endswith('_graph')})
except Exception as error:
    report['error']=repr(error);report['traceback']=traceback.format_exc();raise
finally:
    for mode,adapter in adapters.items():
        if mode.endswith('_graph'):adapter.close()
    assert all(sha(Path(p))==h for p,h in sources.items());authenticate_main();save()
print(json.dumps({k:report[k] for k in ('passed','matching_mean_seconds','graph_speedup_vs_scheduled','graph_speedup_vs_queued')}),flush=True)
