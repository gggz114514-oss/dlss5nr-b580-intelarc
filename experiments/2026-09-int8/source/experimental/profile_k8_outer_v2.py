"""Attribute the isolated3.8 native-half/K8 pipeline without the unstable XPU profiler.

Scoped wrappers time the unchanged warmed pipeline; no extra per-stage waits or
device events are inserted. Inclusive host intervals can contain GPU completion
waits and overlapping submitted GPU work. They are not pure CPU/GPU kernel times.
Unwrapped and wrapped13-frame rounds retain the same complete output/history.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from contextlib import contextmanager,ExitStack
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/k8-outer-profile-v2'
assert not OUT.exists();os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
TOOLCHAIN=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
sys.path[:0]=[str(TOOLCHAIN/'site'),str(EXACT/'reference'),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main()
provision_path=TOOLCHAIN/'provision-v1.json'
assert sha(provision_path)=='e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN/'site'/p)==h for p,h in js(provision_path)['files'].items())
p=D/'results/k8-tiled-residual256-v1/validation.json'
assert sha(p)=='fabe587c41400f51e03dae6dfd5e84fb95c293b76356e9f585370d3d515d3772'
prior=js(p);assert prior['passed'] and js(p.parent.with_suffix('.log.lease.json'))['returncode']==0
sources=dict(prior['sources']);sources[str(p)]=sha(p)
for p in (Path(__file__),HERE/'Run-K8OuterProfileV2.cmd',provision_path):sources[str(p)]=sha(p)
assert all(sha(p)==h for p,h in sources.items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
import torch
import triton
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site') and triton.__version__.startswith('3.8.0')
from PIL import Image
import nr_backend.temporal as temporal
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from k8_tiled_provider_v1 import K8TiledMatrices
from batched_branched_mlp_v2 import FusedBatched
from fused_split_ffwd_v2 import FusedSplit
from fused_c32_mlp_lut_v1 import FusedC32
from cubic_lut_constant_v1 import register,Constant,TABLE
from fused_vit_projection_v3 import FusedVitProjection
from fused_swin_native_half_v1 import FusedSwin
from native_half_head_layout_v1 import HeadLayout
from fused_dynamic_front_v1 import FusedFront
from graph_front_v6 import GraphFront
from graph_history_warp_v2 import GraphHistoryWarp
from residual_scale_v1 import ResidualScale
import compressed_arrays_v1 as arrays
inputs=EXACT/'reference/inputs/flow-full-1920x1080-v3';manifest=js(inputs/'manifest.json')
old=js(D/'results/residual-scale-fp16_xmx-256-v1/validation.json')['runs'][:13]
expected=[(arrays.load(r['output']),arrays.load(r['low_nr'])) for r in old]
OUT.mkdir();report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,
    runs=[],warp_cases=[],warp_capture_raw_bytes=0,complete_migration=False,new_frame_output_array_bytes=0)
adapter=warp=None
stats=None
report['runtime']=dict(triton_version=triton.__version__,triton_file=triton.__file__,torch_version=torch.__version__,isolated=True)

def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

@contextmanager
def timed_patch(obj,name,label):
    had=name in obj.__dict__;own=obj.__dict__.get(name);original=getattr(obj,name)
    def replacement(*args,**kwargs):
        started=time.perf_counter()
        try:return original(*args,**kwargs)
        finally:
            if stats is not None:stats[label]=stats.get(label,0.)+time.perf_counter()-started
    setattr(obj,name,replacement)
    try:yield
    finally:
        assert getattr(obj,name) is replacement
        if had:setattr(obj,name,own)
        else:delattr(obj,name)

@contextmanager
def instrumentation():
    with ExitStack() as stack:
        for obj,name,label in (
            (scaler,'prepare','prepare'),(scaler,'composite','composite'),
            (model,'forward','model_inclusive'),(model,'_validate_rgb','shape_validation'),
            (model,'_forward_front','body_adapter_inclusive'),(adapter,'_validate','constant_guard_inclusive'),
            (adapter,'_constants','constant_signature'),(adapter,'_complete','body_completion_wait'),
            (temporal,'warp_history_square','history_warp'),
            (temporal,'reset_front_features','front'),(temporal,'zero_motion_front_features','front')):
            stack.enter_context(timed_patch(obj,name,label))
        yield

def tensors(i):
    row=manifest['frames'][i]
    pixels=np.asarray(Image.open(inputs/row['file']).convert('RGB'),dtype='f4')/255
    flow=np.fromfile(inputs/row['motion_file'],'<f4').reshape(1080,1920,2)
    return pixels,flow,torch.from_numpy(pixels).to('xpu'),torch.from_numpy(flow).to('xpu')

try:
    torch.set_num_threads(2)
    model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',
        EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    lut=register(model);constant=Constant(model);provider=K8TiledMatrices(model);provider.select('fp16_xmx')
    adapter=GraphFront(model,arithmetic=provider);warp=GraphHistoryWarp(model);scaler=ResidualScale(256)
    components=[provider,FusedSwin(provider),HeadLayout(model,provider),adapter,
        FusedBatched(model,provider,workload='small'),FusedSplit(model,provider),FusedC32(model,provider),
        FusedVitProjection(model,provider),FusedFront(model),warp]
    with torch.inference_mode(),ExitStack() as stack:
        for c in components:stack.enter_context(c.installed())
        for i in (0,1):
            _,_,rgb,motion=tensors(i);canvas,flow=scaler.prepare(rgb,motion)
            with use_arithmetic_backend('triton'):low=model(canvas,flow,reset=i==0)
            assert low.cpu().numpy().tobytes()==expected[i][1].tobytes()
        model.reset();print('Warmed both body states and history warp',flush=True)
        for repetition,instrumented in enumerate((False,True,True,False,True)):
            with instrumentation() if instrumented else ExitStack():
                for i,spec in enumerate(manifest['frames']):
                    pixels,motion_np,rgb,motion=tensors(i)
                    torch.xpu.synchronize();stats={} if instrumented else None
                    started=time.perf_counter()
                    canvas,flow=scaler.prepare(rgb,motion)
                    with use_arithmetic_backend('triton') as dispatch:low=model(canvas,flow,reset=spec['reset'])
                    value=scaler.composite(rgb,canvas,low)
                    before_sync=time.perf_counter();torch.xpu.synchronize();finish=time.perf_counter()
                    if stats is not None:stats['final_sync']=finish-before_sync
                    seconds=finish-started;parts=dict(stats) if stats is not None else None;stats=None
                    a,b=value.cpu().numpy(),low.cpu().numpy()
                    assert a.tobytes()==expected[i][0].tobytes() and b.tobytes()==expected[i][1].tobytes()
                    assert model._previous.cpu().numpy().tobytes()==b.tobytes()
                    assert model.next_seed==(1 if spec['reset'] else i+1)
                    effective=dict(dispatch);assert effective.pop('xpu_graph_replay')==1
                    for k,v in adapter.last_entry.dispatch.items():
                        if k!='backend':effective[k]=effective.get(k,0)+v
                    assert effective==prior['runs']['k8_tiled'][i]['effective_dispatch']
                    if parts is not None:
                        parts['model_validation_and_commit_other']=parts['model_inclusive']-sum(parts.get(n,0.) for n in ('shape_validation','history_warp','front','body_adapter_inclusive'))
                        parts['body_submission_and_clone_other']=parts['body_adapter_inclusive']-parts['constant_guard_inclusive']-parts['body_completion_wait']
                        parts['pipeline_other']=seconds-sum(parts[n] for n in ('prepare','model_inclusive','composite','final_sync'))
                        assert min(parts.values())>=0
                    report['runs'].append(dict(round=repetition,frame=i,reset=spec['reset'],instrumented=instrumented,
                        seconds=seconds,parts_seconds=parts,full_byte_equal_prior=True,low_byte_equal_prior=True,
                        private_byte_equal_low=True,next_seed=model.next_seed,output=old[i]['output'],low_nr=old[i]['low_nr']))
                    if repetition==0 and not spec['reset']:
                        assert len(warp.entries)==1
                        e=next(iter(warp.entries.values()))
                        assert bool(e.outputs[2])
                        captured={n:arrays.save(t.cpu().numpy()) for n,t in [('image',e.image),('motion',e.motion),('numerator',e.outputs[0]),('reciprocal',e.outputs[1])]}
                        report['warp_cases'].append(dict(frame=i,arrays=captured,valid=True))
                        report['warp_capture_raw_bytes']+=sum(m['raw_bytes'] for m in captured.values())
                    value.zero_();low.zero_();assert model._previous.cpu().numpy().tobytes()==b.tobytes()
                    assert rgb.cpu().numpy().tobytes()==pixels.tobytes() and motion.cpu().numpy().tobytes()==motion_np.tobytes()
            save();print(json.dumps(dict(round=repetition,instrumented=instrumented,all_equal=True)),flush=True)
        rows=[r for r in report['runs'] if r['instrumented'] and not r['reset']]
        report['host_inclusive_mean_ms']={k:statistics.mean(r['parts_seconds'].get(k,0.) for r in rows)*1000 for k in rows[0]['parts_seconds']}
        report['temporal_mean_ms']={str(flag):statistics.mean(r['seconds'] for r in report['runs'] if r['instrumented']==flag and not r['reset'])*1000 for flag in (False,True)}
        report['constant_buffer_count']=len(adapter.constants)
        before=adapter._constants();samples=[]
        for _ in range(5):
            started=time.perf_counter()
            for _ in range(100):assert adapter._constants()==before
            samples.append((time.perf_counter()-started)/100)
        report['isolated_host_constant_scan_seconds']=samples
        assert constant.require() is lut and lut.cpu().numpy().tobytes()==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
        assert len(report['warp_cases'])==11
        report.update(passed=True,full_outputs_verified=65,lut_unchanged=True,graphs=adapter.metadata())
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    if warp is not None:warp.close()
    if adapter is not None:adapter.close()
    assert all(sha(p)==h for p,h in sources.items());authenticate_main();save()
print(json.dumps({k:report[k] for k in ('passed','host_inclusive_mean_ms','temporal_mean_ms','constant_buffer_count','isolated_host_constant_scan_seconds')}),flush=True)
