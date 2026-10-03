"""Diagnostic CPU/XPU trace of current fast modes; not a throughput benchmark."""
import hashlib,json,os,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=DREF/'profiles/scheduled-fast-480-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1')
sys.path[:0]=[str(R),str(ROOT/'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import sha,js
gate=authenticate_main()
full_path=DREF/'results/fused-full-480-v2/validation.json';full=js(full_path);assert full['passed']
assert all(sha(Path(p))==h for p,h in full['sources'].items())
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
import immutable_artifacts_v1 as artifacts
inputs=R/'inputs/flow-full-864x480-v2';manifest=js(inputs/'manifest.json')
paths=[Path(__file__),HERE/'Run-ProfileScheduledFastV1.cmd',HERE/'swin_scheduling_v1.py',HERE/'fused_cached_matrices_v2.py',HERE/'fused_cached_matrices_v1.py',HERE/'fused_activation_int8_v1.py',HERE/'static_weight_cache_v2.py',HERE/'static_weight_cache_v1.py',HERE/'fast_matrices_v3.py',HERE/'immutable_artifacts_v1.py',full_path,prior_path,inputs/'manifest.json',*sorted((ROOT/'backend/nr_backend').glob('*.py'))]
sources={str(p):sha(p) for p in paths};OUT.mkdir(parents=True)
report=dict(scope=__doc__,sources=sources,passed=False,modes={},exact_gate=gate,timing_scope='Profiler instrumentation changes timing. Use paired fused-full-480-v2 for speed claims. Device interval union must be computed before discussing busy fraction; do not sum overlapping events.',activities=[str(a) for a in torch.profiler.supported_activities()])
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
def tensors(i):
    spec=manifest['frames'][i];pixels=np.asarray(Image.open(inputs/spec['file']).convert('RGB'),dtype='f4')/255
    motion=np.fromfile(inputs/spec['motion_file'],'<f4').reshape(480,864,2)
    return torch.from_numpy(pixels).to('xpu'),torch.from_numpy(motion).to('xpu')
try:
    torch.set_num_threads(2)
    assert torch.profiler.ProfilerActivity.XPU in torch.profiler.supported_activities(),'XPU profiler unavailable'
    model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    experiment=FusedCachedMatrices();scheduler=SwinScheduling(enabled=True)
    with torch.inference_mode(),experiment.installed(),scheduler.installed():
        experiment.static_cache.prepack(model)
        for mode in ('fp16_xmx','int8_fused_v2'):
            experiment.select(mode)
            for i in (0,1):
                rgb,motion=tensors(i)
                with use_arithmetic_backend('triton'):model(rgb,motion,reset=i==0)
                torch.xpu.synchronize()
            rgb0,motion0=tensors(0)
            with use_arithmetic_backend('triton'):model(rgb0,motion0,reset=True)
            rgb,motion=tensors(1);torch.xpu.synchronize()
            experiment.select(mode);scheduler.counts={}
            print('Profiling '+mode,flush=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.XPU],record_shapes=False,profile_memory=False,with_stack=False) as prof:
                with torch.profiler.record_function('NR_scheduled_'+mode):
                    with use_arithmetic_backend('triton') as dispatch:
                        value=model(rgb,motion,reset=False);torch.xpu.synchronize()
            actual=value.cpu().numpy();old_mode='int8_dense' if mode.startswith('int8') else mode
            meta=prior['runs'][old_mode][1]['actual'];assert sha(Path(meta['path']))==meta['sha256']
            expected=np.load(meta['path'],allow_pickle=False).astype('f2');equal=actual.tobytes()==expected.tobytes()
            trace_path=OUT/(mode+'-trace.json');prof.export_chrome_trace(str(trace_path))
            stats=[]
            for event in prof.key_averages():
                stats.append(dict(name=event.key,count=event.count,self_cpu_us=event.self_cpu_time_total,cpu_us=event.cpu_time_total,self_device_us=event.self_device_time_total,device_us=event.device_time_total))
            row=dict(byte_equal_prior=equal,output=meta if equal else artifacts.array(actual),next_seed=model.next_seed,dispatches=dict(dispatch),matrix_calls=experiment.calls.copy(),scheduling=scheduler.counts.copy(),trace=dict(path=str(trace_path),sha256=sha(trace_path),bytes=trace_path.stat().st_size),events=stats)
            report['modes'][mode]=row;save()
            assert equal and model.next_seed==2 and model._previous.cpu().numpy().tobytes()==actual.tobytes()
            print(json.dumps(dict(mode=mode,trace_bytes=row['trace']['bytes'],top_self_cpu=sorted(stats,key=lambda x:-x['self_cpu_us'])[:8],top_self_device=sorted(stats,key=lambda x:-x['self_device_us'])[:8])),flush=True)
    report['passed']=True
except Exception as error:
    report['error']=repr(error);report['traceback']=traceback.format_exc();raise
finally:
    assert all(sha(Path(p))==h for p,h in sources.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'])),flush=True)
