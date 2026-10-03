"""One process of an ABBA compiler comparison, using unchanged native-half code.

Run stock3.7.2, isolated3.8, isolated3.8, stock3.7.2 serially under the GPU lease.
Same resident256 native input, zero motion, default SDR, full NR host completion.
Each process measures180 reset and180 temporal calls, checks complete half output
hashes against the frozen3.7.2 run, and maintains its own model and history.
No C32 tail fusion is installed. Excludes IO/upload/JIT/validation/game contention.
"""
import argparse,hashlib,json,os,statistics,sys,time,traceback,urllib.request
from contextlib import ExitStack,contextmanager
from pathlib import Path,PureWindowsPath
p=argparse.ArgumentParser();p.add_argument('runtime',choices=['372','380']);p.add_argument('run',type=int,choices=[0,1]);args=p.parse_args()
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=D/f'results/nr256-compiler-{args.runtime}-r{args.run}-v1';assert not OUT.exists()
TOOLCHAIN=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
os.environ['TRITON_CACHE_DIR']=str(D/('triton-cache-c32-triton38-v1' if args.runtime=='380' else 'triton-cache-sm89-v1'))
sys.path[:0]=([str(TOOLCHAIN/'site')] if args.runtime=='380' else [])+[str(R),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main()
legacy_path=D/'results/nr256-native-parity-v1/validation.json'
assert sha(legacy_path)=='5008dfcf5316aabe984e1e30d1db1c933bb3536c5e4a54a3020390c4845ff007'
legacy=js(legacy_path)
assert legacy['passed'] and js(legacy_path.parent.with_suffix('.log.lease.json'))['returncode']==0
assert all(sha(p)==h for p,h in legacy['sources'].items())
legacy_hashes={(r['round'],r['mode'],r['sample']):r['output_raw_sha256'] for r in legacy['measured']['native_half']}
provision_path=TOOLCHAIN/'provision-v1.json'
assert sha(provision_path)=='e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
if args.runtime=='380':assert all(sha(TOOLCHAIN/'site'/p)==h for p,h in js(provision_path)['files'].items())
old_path=R/'4060-real-flow-256x256-sequence-v3.json';old=js(old_path)
folder=R/'results'/PureWindowsPath(old['runDirectory']).name/'output'
input_path=folder/'frame00.png_input.rgba32f.bin';motion_path=folder/'frame00.png_motion.rg32f.bin'
for p in (input_path,motion_path):assert sha(p)==next(f['sha256'].lower() for f in old['outputs'] if f['name']==p.name)
sources={**legacy['sources']}
for p in (Path(__file__),HERE/'Run-Nr256CompilerPairV1.cmd',provision_path,legacy_path,old_path,input_path,motion_path):sources[str(p)]=sha(p)
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
import torch
import triton
if args.runtime=='380':
    assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site') and triton.__version__.startswith('3.8.0')
else:
    assert triton.__version__=='3.7.2'
    assert Path(triton.__file__).is_relative_to(Path(sys.executable).parent/'Lib/site-packages')
    assert sha(Path(triton.__file__).parent/'_C/libtriton.pyd')=='e2a1cb8b04962fb86966d7e645b377bbacc084c315fd244ef33b90ba952315f8'
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from strided_batched_v2 import StridedMatrices
from batched_branched_mlp_v2 import FusedBatched
from fused_split_ffwd_v2 import FusedSplit
from fused_c32_mlp_lut_v1 import FusedC32
from cubic_lut_constant_v1 import register,Constant,TABLE
from fused_vit_projection_v3 import FusedVitProjection
from native_half_head_layout_v1 import HeadLayout
from fused_swin_native_half_v1 import FusedSwin
from fused_dynamic_front_v1 import FusedFront
from graph_front_v6 import GraphFront
from graph_history_warp_v2 import GraphHistoryWarp
import compressed_arrays_v1 as arrays
OUT.mkdir();report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,measured=[],complete_migration=False,
    runtime=dict(selector=args.runtime,run=args.run,triton_version=triton.__version__,triton_file=triton.__file__,torch_version=torch.__version__),
    native_4060_reference=legacy['native_4060_reference'])
model=adapter=warp=None
raw=lambda t:t.cpu().numpy().tobytes()
digest=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
@contextmanager
def installed():
    with ExitStack() as stack:
        for c in components:stack.enter_context(c.installed())
        yield
def run(reset):
    with installed(),use_arithmetic_backend('triton') as dispatch:
        torch.xpu.synchronize();started=time.perf_counter()
        value=model(rgb,motion,reset=reset)
        torch.xpu.synchronize();seconds=time.perf_counter()-started
    assert dispatch['xpu_graph_replay']==1
    return value,seconds
try:
    torch.set_num_threads(2)
    pixels=np.fromfile(input_path,'<f4').reshape(256,256,4)[...,:3].copy()
    flow=np.fromfile(motion_path,'<f4').reshape(256,256,2).copy();assert not flow.any()
    rgb=torch.from_numpy(pixels).to('xpu');motion=torch.from_numpy(flow).to('xpu')
    report.update(input_rgb_sha256=digest(pixels),input_motion_sha256=digest(flow),b580_device=torch.xpu.get_device_name())
    model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    register(model);provider=StridedMatrices();provider.select('fp16_xmx')
    adapter=GraphFront(model,arithmetic=provider);warp=GraphHistoryWarp(model)
    components=[provider,FusedSwin(provider),HeadLayout(model,provider),adapter,FusedBatched(model,provider,workload='small'),
        FusedSplit(model,provider),FusedC32(model,provider),FusedVitProjection(model,provider),FusedFront(model),warp]
    with torch.inference_mode():
        held,_=run(True);held_bytes=raw(held)
        assert held_bytes==arrays.load(legacy['reset_output']).tobytes()
        run(False);print('Prewarmed '+args.runtime,flush=True)
        for r in range(3):
            for position in range(2):
                mode=('reset','temporal')[(position+r)%2];reset=mode=='reset'
                run(True)
                for _ in range(20):run(reset)
                for i in range(60):
                    value,seconds=run(reset);a=value.cpu().numpy()
                    assert a.shape==(256,256,3) and a.dtype==np.dtype('f2') and np.isfinite(a).all()
                    assert digest(a)==legacy_hashes[(r,mode,i)]
                    assert raw(model._previous)==a.tobytes() and model.next_seed==(1 if reset else 22+i)
                    report['measured'].append(dict(round=r,mode=mode,sample=i,host_ms=seconds*1000,
                        output_raw_sha256=digest(a),seed=model.next_seed,private_equal=True,caller_independent=True))
                    value.zero_();assert raw(model._previous)==a.tobytes()
                assert raw(held)==held_bytes
                save();print(json.dumps(dict(runtime=args.runtime,run=args.run,round=r,mode=mode,
                    mean_host_ms=statistics.mean(s['host_ms'] for s in report['measured'][-60:]),all60outputs_match=True)),flush=True)
        body=[]
        for entry in adapter.entries.values():
            target=raw(entry.output);samples=[]
            for _ in range(5):
                torch.xpu.synchronize();started=time.perf_counter()
                for _ in range(10):entry.graph.replay()
                torch.xpu.synchronize();samples.append((time.perf_counter()-started)*100)
                assert raw(entry.output)==target
            body.append(dict(temporal=entry.inputs['previous'] is not None,samples_host_ms=samples,
                median_host_ms=statistics.median(samples),graph_replays=50,complete_output_readbacks=5))
        value,_=run(True);assert raw(value)==held_bytes and model.next_seed==1
        assert raw(Constant(model).require())==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
        assert raw(rgb)==pixels.tobytes() and raw(motion)==flow.tobytes()
        rows=report['measured']
        report.update(passed=True,full_outputs_verified=360,all_outputs_match_frozen_372=True,
            inputs_and_lut_unchanged=True,held_outputs_unchanged=True,reset_reproduces_first=True,
            static_body_diagnostic=body,graphs=adapter.metadata(),
            summary={m:dict(mean_host_ms=statistics.mean(s['host_ms'] for s in rows if s['mode']==m),
                median_host_ms=statistics.median(s['host_ms'] for s in rows if s['mode']==m),
                round_mean_host_ms=[statistics.mean(s['host_ms'] for s in rows if s['mode']==m and s['round']==j) for j in range(3)])
                for m in ('reset','temporal')})
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    if warp is not None:warp.close()
    if adapter is not None:adapter.close()
    assert all(sha(p)==h for p,h in sources.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],runtime=args.runtime,run=args.run,summary=report.get('summary'))),flush=True)
