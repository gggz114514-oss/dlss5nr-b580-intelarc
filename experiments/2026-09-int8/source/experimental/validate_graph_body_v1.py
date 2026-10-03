"""Complete480 graph smoke: reset, two changing temporal inputs and reset again."""
import hashlib,json,os,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=DREF/'results/graph-body-480-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1');sys.path[:0]=[str(R),str(ROOT/'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js,sha
gate=authenticate_main()
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
import immutable_artifacts_v1 as artifacts
inputs=R/'inputs/flow-full-864x480-v2';manifest=js(inputs/'manifest.json')
paths=[Path(__file__),HERE/'Run-GraphBodyV1.cmd',HERE/'graph_front_v1.py',HERE/'capture_body_v1.py',HERE/'swin_scheduling_v1.py',HERE/'fused_cached_matrices_v2.py',HERE/'fused_cached_matrices_v1.py',HERE/'fused_activation_int8_v1.py',HERE/'static_weight_cache_v2.py',HERE/'static_weight_cache_v1.py',HERE/'fast_matrices_v3.py',HERE/'immutable_artifacts_v1.py',prior_path,inputs/'manifest.json',Path(torch.xpu.__file__).with_name('graphs.py'),*sorted((ROOT/'backend/nr_backend').glob('*.py'))]
sources={str(p):sha(p) for p in paths}
OUT.mkdir()
report=dict(scope=__doc__,sources=sources,passed=False,runs=[],exact_gate=gate,complete_migration=False,timing_scope='Smoke only. Initial calls include warmup and capture; not a paired throughput benchmark.')
adapter=None
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
def tensors(i):
    spec=manifest['frames'][i];pixels=np.asarray(Image.open(inputs/spec['file']).convert('RGB'),dtype='f4')/255
    motion=np.fromfile(inputs/spec['motion_file'],'<f4').reshape(480,864,2)
    return torch.from_numpy(pixels).to('xpu'),torch.from_numpy(motion).to('xpu')
try:
    torch.set_num_threads(2)
    model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    experiment=FusedCachedMatrices();experiment.select('fp16_xmx');scheduler=SwinScheduling(enabled=True)
    adapter=GraphFront(model,arithmetic=experiment);held=[]
    with torch.inference_mode(),experiment.installed(),scheduler.installed(),adapter.installed():
        for step,i in enumerate((0,1,2,0)):
            rgb,motion=tensors(i);torch.xpu.synchronize()
            print(f'Step{step} source{i}: graph call',flush=True);started=time.perf_counter()
            with use_arithmetic_backend('triton') as dispatch:
                value=model(rgb,motion,reset=i==0);torch.xpu.synchronize()
            seconds=time.perf_counter()-started
            actual=value.cpu().numpy();meta=prior['runs']['fp16_xmx'][i]['actual']
            assert sha(Path(meta['path']))==meta['sha256']
            expected=np.load(meta['path'],allow_pickle=False).astype('f2');equal=actual.tobytes()==expected.tobytes()
            effective=dict(dispatch);assert effective.pop('xpu_graph_replay')==1
            for k,v in adapter.last_entry.dispatch.items():
                if k!='backend':effective[k]=effective.get(k,0)+v
            row=dict(step=step,source_frame=i,seconds=seconds,byte_equal_prior=equal,output=meta if equal else artifacts.array(actual),next_seed=model.next_seed,runtime_dispatch=dict(dispatch),captured_dispatch=adapter.last_entry.dispatch,effective_dispatch=effective,graphs=adapter.metadata(),peak_allocated=torch.xpu.max_memory_allocated())
            report['runs'].append(row);save()
            assert equal and actual.tobytes()==model._previous.cpu().numpy().tobytes(),step
            assert effective==prior['runs']['fp16_xmx'][i]['dispatches'] and model.next_seed==i+1
            for old,old_bytes in held:assert old.cpu().numpy().tobytes()==old_bytes
            held.append((value,actual.tobytes()))
            print(json.dumps(dict(step=step,byte_equal=True,seconds=seconds,graph_count=len(adapter.entries),effective_dispatch=effective)),flush=True)
        assert len(adapter.entries)==2 and adapter.replays==4
        previous=model._previous.cpu().numpy().tobytes();seed=model.next_seed
        value.zero_();assert model._previous.cpu().numpy().tobytes()==previous
        saved_scale=model.blend_scale.clone();model.blend_scale.add_(.125)
        rgb,motion=tensors(1)
        try:
            with use_arithmetic_backend('triton'):model(rgb,motion,reset=False)
        except RuntimeError as error:assert 'Model constants changed' in str(error)
        else:raise AssertionError('Mutated model reused stale graph')
        assert model.next_seed==seed and model._previous.cpu().numpy().tobytes()==previous
        model.blend_scale.copy_(saved_scale)
        report.update(passed=True,reset_and_temporal_graphs=2,changing_input_bytes_match=True,held_outputs_survive_replay=True,private_history_ownership=True,model_mutation_rejected_without_state_advance=True)
except Exception as error:
    report['error']=repr(error);report['traceback']=traceback.format_exc();raise
finally:
    if adapter is not None:adapter.close()
    assert all(sha(Path(p))==h for p,h in sources.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'])),flush=True)
