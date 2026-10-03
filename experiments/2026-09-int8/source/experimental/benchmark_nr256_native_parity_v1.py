"""Measure B580 NR alone against the native4060 resident-input benchmark scope.

Use the exact native half input texture from the same256 fixture, zero motion,
default SDR controls,180 reset plus180 temporal calls per B580 variant. Compare
previous versus native-half attention output bytes on every call. Host timing
includes complete NR, state commit and completion, excluding residual scaling.
Additional static body event intervals are diagnostic, not a complete NR GPU
measurement and must not be used as the cross-device parity score.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from contextlib import ExitStack,contextmanager
from pathlib import Path,PureWindowsPath
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'results/nr256-native-parity-v1';assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-sm89-v1');sys.path[:0]=[str(R),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main()
paired_path=D/'results/native-half-residual256-v2/validation.json';paired=js(paired_path)
native_path=D/'experimental/native-steady-bench-v1/validation.json';native=js(native_path)
assert sha(native_path)=='70a86900dbde71d285f187cedc42a123bc7706cf27f21f0231e515b4d964659d'
assert paired['passed'] and js(paired_path.parent.with_suffix('.log.lease.json'))['returncode']==0 and native['passed']
assert all(sha(p)==h for p,h in paired['sources'].items()) and all(sha(p)==h for p,h in native['sources'].items())
old_path=R/'4060-real-flow-256x256-sequence-v3.json';old=js(old_path)
folder=R/'results'/PureWindowsPath(old['runDirectory']).name/'output'
input_path=folder/'frame00.png_input.rgba32f.bin';motion_path=folder/'frame00.png_motion.rg32f.bin'
reference_path=folder/'frame00.png_output.rgba32f.bin'
for p in (input_path,motion_path,reference_path):assert sha(p)==next(f['sha256'].lower() for f in old['outputs'] if f['name']==p.name)
sources={**paired['sources'],**native['sources']}
for p in (Path(__file__),HERE/'Run-Nr256NativeParityV1.cmd',paired_path,native_path,old_path,input_path,motion_path,reference_path):sources[str(p)]=sha(p)
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
import torch
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from strided_batched_v2 import StridedMatrices
from batched_branched_mlp_v2 import FusedBatched
from fused_split_ffwd_v2 import FusedSplit
from fused_c32_mlp_lut_v1 import FusedC32
from cubic_lut_constant_v1 import register,Constant,TABLE
from fused_vit_projection_v3 import FusedVitProjection
from projected_head_layout_v1 import ProjectedHeadLayout
from native_half_head_layout_v1 import HeadLayout
from fused_swin_scheduling_v1 import FusedSwin as OldSwin
from fused_swin_native_half_v1 import FusedSwin as NewSwin
from fused_dynamic_front_v1 import FusedFront
from graph_front_v6 import GraphFront
from graph_history_warp_v2 import GraphHistoryWarp
from shared_model_buffers_v1 import share_identical_buffers
from serial_graph_workspace_v1 import share_before_capture
import compressed_arrays_v1 as arrays
OUT.mkdir();report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,
    measured={'previous':[],'native_half':[]},complete_migration=False,
    parity_target='Same256x256 native NR input, resident textures, complete host call+completion; no residual scaling.',
    native_4060_reference=native['cases']['256x256']['statistics'],
    precision='B580 already-approved FP16 fast arithmetic; only scheduling/native-half attention changes versus previous.',
    body_event_timing_is_not_complete_nr_gpu_time=True)
models={};adapters={};warps={};contexts={};held={}
raw=lambda t:t.cpu().numpy().tobytes()
digest=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
@contextmanager
def installed(name):
    with ExitStack() as stack:
        for c in contexts[name]:stack.enter_context(c.installed())
        yield
def run(name,reset):
    with installed(name),use_arithmetic_backend('triton') as dispatch:
        torch.xpu.synchronize();started=time.perf_counter()
        value=models[name](rgb,motion,reset=reset)
        torch.xpu.synchronize();seconds=time.perf_counter()-started
    effective=dict(dispatch);assert effective.pop('xpu_graph_replay')==1
    for key,count in adapters[name].last_entry.dispatch.items():
        if key!='backend':effective[key]=effective.get(key,0)+count
    return value,seconds,effective
try:
    torch.set_num_threads(2)
    pixels=np.fromfile(input_path,'<f4').reshape(256,256,4)[...,:3].copy()
    flow=np.fromfile(motion_path,'<f4').reshape(256,256,2).copy();assert not flow.any()
    ref=np.fromfile(reference_path,'<f4').reshape(256,256,4)[...,:3].copy()
    rgb=torch.from_numpy(pixels).to('xpu');motion=torch.from_numpy(flow).to('xpu')
    report.update(input_rgb_sha256=digest(pixels),input_motion_sha256=digest(flow),b580_device=torch.xpu.get_device_name())
    for name in ('previous','native_half'):
        model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
        register(model)
        if models:report['shared_constants']=share_identical_buffers(model,models['previous'])
        models[name]=model;provider=StridedMatrices();provider.select('fp16_xmx')
        adapters[name]=GraphFront(model,arithmetic=provider);warps[name]=GraphHistoryWarp(model)
        contexts[name]=[provider,(OldSwin if name=='previous' else NewSwin)(provider),
            (ProjectedHeadLayout if name=='previous' else HeadLayout)(model,provider),adapters[name],
            FusedBatched(model,provider,workload='small'),FusedSplit(model,provider),FusedC32(model,provider),
            FusedVitProjection(model,provider),FusedFront(model),warps[name]]
    report['shared_transient_pool']=share_before_capture(adapters.values())
    with torch.inference_mode():
        first={}
        for name in models:
            v,_,_=run(name,True);a=v.cpu().numpy();first[name]=a.copy()
            held[name]=(v,raw(v));run(name,False)
            print('Prewarmed '+name,flush=True)
        assert first['previous'].tobytes()==first['native_half'].tobytes()
        report['reset_output']=arrays.save(first['native_half'])
        error=first['native_half'].astype('f4')-ref
        report['existing_fast_difference_from_native_reset']=dict(mean_abs=float(np.abs(error).mean()),max_abs=float(np.abs(error).max()),
            byte_equal=first['native_half'].astype('f4').tobytes()==ref.tobytes())
        for r in range(3):
            for position in range(2):
                mode=('reset','temporal')[(position+r)%2];reset=mode=='reset'
                for name in models:
                    run(name,True)
                    for i in range(20):warm,_,_=run(name,reset)
                    if name=='previous':warm_bytes=raw(warm)
                    else:assert raw(warm)==warm_bytes
                for i in range(60):
                    order=list(models);shift=(r+i)%2;order=order[shift:]+order[:shift];values={};dispatches={}
                    for name in order:
                        value,seconds,dispatch=run(name,reset);a=value.cpu().numpy()
                        assert a.shape==(256,256,3) and a.dtype==np.dtype('f2') and np.isfinite(a).all()
                        assert raw(models[name]._previous)==a.tobytes() and models[name].next_seed==(1 if reset else 22+i)
                        values[name]=a.copy();dispatches[name]=dispatch
                        report['measured'][name].append(dict(round=r,mode=mode,sample=i,host_ms=seconds*1000,
                            output_raw_sha256=digest(a),seed=models[name].next_seed,private_equal=True,caller_independent=True))
                        value.zero_();assert raw(models[name]._previous)==a.tobytes()
                    assert values['previous'].tobytes()==values['native_half'].tobytes()
                    assert dispatches['previous']==dispatches['native_half']
                for name,(v,data) in held.items():assert raw(v)==data
                save();print(json.dumps(dict(round=r,mode=mode,all60outputs_match=True,
                    host_mean_ms={name:statistics.mean(x['host_ms'] for x in report['measured'][name][-60:]) for name in models})),flush=True)
        report['static_body_diagnostic']={}
        for name,adapter in adapters.items():
            report['static_body_diagnostic'][name]=[]
            for entry in adapter.entries.values():
                with entry.stream:entry.graph.replay()
                torch.xpu.synchronize();target=raw(entry.output);samples=[]
                begin=torch.xpu.Event(enable_timing=True);end=torch.xpu.Event(enable_timing=True)
                for i in range(10):
                    torch.xpu.synchronize()
                    with entry.stream:
                        begin.record();started=time.perf_counter();entry.graph.replay();end.record();end.synchronize()
                        wall=(time.perf_counter()-started)*1000;event=begin.elapsed_time(end)
                    assert 0<event<10000 and raw(entry.output)==target
                    samples.append(dict(event_interval_ms=event,host_ms=wall))
                report['static_body_diagnostic'][name].append(dict(temporal=entry.inputs['previous'] is not None,samples=samples,output_unchanged=True))
        for name in models:
            value,_,_=run(name,True);assert raw(value)==first[name].tobytes() and models[name].next_seed==1
            assert raw(Constant(models[name]).require())==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
        assert raw(rgb)==pixels.tobytes() and raw(motion)==flow.tobytes()
        report['summary']={name:{mode:dict(mean_host_ms=statistics.mean(s['host_ms'] for s in rows if s['mode']==mode),
            median_host_ms=statistics.median(s['host_ms'] for s in rows if s['mode']==mode),
            round_mean_host_ms=[statistics.mean(s['host_ms'] for s in rows if s['mode']==mode and s['round']==r) for r in range(3)])
            for mode in ('reset','temporal')} for name,rows in report['measured'].items()}
        report.update(passed=True,complete_outputs_compared=720,inputs_unchanged=True,reset_reproduces_first=True,
            all_candidate_outputs_byte_equal=True,held_outputs_unchanged=True,lut_bytes_unchanged=True,
            graphs={name:adapter.metadata() for name,adapter in adapters.items()})
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for warp in warps.values():warp.close()
    for adapter in adapters.values():adapter.close()
    assert all(sha(p)==h for p,h in sources.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],summary=report.get('summary'))),flush=True)
