"""Parallel-part ViT projections, synthetic token counts and full-body checks.

The two projection families preserve four independent K partitions with half
merges. Complete operands/results are saved, not sampled. Larger token counts
use repeated small-frame operands and do not establish real large-frame fidelity.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=DREF/'experimental/fused-vit-projection-v2'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1')
sys.path[:0]=[str(R),str(ROOT/'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js,sha
gate=authenticate_main()
prior_path=DREF/'experimental/fused-body-stages-v2/validation.json'
assert sha(prior_path)=='85462097455e379845eb3c1244f22fa91329dd5ddaaf708d438424dd58f3efac'
prior=js(prior_path);assert prior['passed']
assert all(sha(Path(p))==h for p,h in prior['sources'].items())
assert js(prior_path.parent.with_suffix('.log.lease.json'))['returncode']==0
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as r:queue=json.load(r)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
import torch
import nr_backend.vit_block as vit
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from strided_batched_v2 import StridedMatrices
from fused_branched_pairs_v2 import FusedPairs
from fused_c32_mlp_v1 import FusedC32
from fused_split_ffwd_v2 import FusedSplit
from fused_swin_scheduling_v1 import FusedSwin
from qkv_head_layout_v1 import QKVHeadLayout
from fused_vit_projection_v2 import forward,FusedVitProjection
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
paths=[Path(__file__),HERE/'Run-VitProjectionV2.cmd',HERE/'fused_vit_projection_v1.py',HERE/'fused_vit_projection_v2.py',prior_path,*[Path(p) for p in prior['sources']]]
frozen={str(p):sha(p) for p in paths}
OUT.mkdir()
report=dict(scope=__doc__,sources=frozen,exact_gate=gate,passed=False,cases=[],candidates=[],body_checks=[],complete_migration=False,
            timing='Static graph replay, five rotated rounds x20 calls including same output copy in all variants; not complete-frame performance.')
cases=[];graphs=[]
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
def raw(t):return t.cpu().numpy().tobytes()
def graph_for(fn):
    stream=torch.xpu.Stream();torch.xpu.synchronize()
    with stream:
        for _ in range(2):warm=fn()
    torch.xpu.synchronize();out=torch.empty_like(warm);del warm
    graph=torch.xpu.XPUGraph()
    with torch.xpu.graph(graph,stream=stream):temporary=fn();out.copy_(temporary)
    del temporary
    graph.replay();torch.xpu.synchronize()
    return graph,out,stream
try:
    torch.set_num_threads(2)
    model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    provider=StridedMatrices();provider.select('fp16_xmx')
    pairs,c32=FusedPairs(model,provider),FusedC32(model,provider,bm=32,warps=4,stages=1)
    scheduler,split,layout=FusedSwin(provider),FusedSplit(model,provider),QKVHeadLayout(model,provider)
    names={id(getattr(m,role)):name+'.'+role for name,m in model.named_modules() if type(m) is vit.VitBlock for role in ('contract','projection')}
    original=vit.split_k_projection
    def capture(features,weight,initial,parts=4):
        assert parts==4 and id(weight) in names
        value=original(features,weight,initial,parts)
        cases.append((names[id(weight)],features.clone(),weight,initial.clone(),value.clone()))
        return value
    options=dict(sigmoid=model.sigmoid,blend_scale=model.blend_scale,return_float32=False)
    with torch.inference_mode(),provider.installed(),pairs.installed(),c32.installed(),split.installed(),scheduler.installed(),layout.installed(),body.installed(),use_arithmetic_backend('triton'):
        inputs={name:torch.from_numpy(arrays.load(meta)).to('xpu') for name,meta in prior['body_inputs'].items()}
        vit.split_k_projection=capture
        try:value=body.forward_front(model,**inputs,**options)
        finally:vit.split_k_projection=original
        assert raw(value)==arrays.load(prior['expected_output']).tobytes()
        assert len(cases)==len(names)==16 and len({c[0] for c in cases})==16
        for family in (0,1):
            name,x,w,initial,_=cases[family]
            for count in (19,96,128,640,960):
                xx=x.repeat((count+x.shape[0]-1)//x.shape[0],1)[:count].contiguous()
                ii=initial.repeat((count+x.shape[0]-1)//x.shape[0],1)[:count].contiguous()
                cases.append((f'synthetic_repeated_or_tail_K{x.shape[1]}_M{count}',xx,w,ii,original(xx,w,ii)))
        configs=[dict(bm=bm,bn=bn,stages=s) for bm,bn,s in ((16,32,1),(16,64,1),(32,32,1),(32,64,1),(16,64,2))]
        for index,(name,x,w,initial,expected) in enumerate(cases):
            expected_bytes=raw(expected)
            report['cases'].append(dict(name=name,shape=list(x.shape),synthetic=name.startswith('synthetic'),arrays={
                'x':arrays.save(x.cpu().numpy()),'w':arrays.save(w.cpu().numpy()),'initial':arrays.save(initial.cpu().numpy()),'expected':arrays.save(expected.cpu().numpy())}))
            for config in configs:
                actual,kernels=forward(x,w,initial,**config)
                equal=raw(actual)==expected_bytes
                report['candidates'].append(dict(case=index,config=config,byte_equal=equal,spills=max(k.n_spills for k in kernels),registers=[k.n_regs for k in kernels],kernel_spills=[k.n_spills for k in kernels]))
                assert equal,(name,config)
            save();print('Byte checked '+name,flush=True)
        report['timings']=[]
        for index in (0,1,14,15,18,20,23,25):
            name,x,w,initial,expected=cases[index];expected_bytes=raw(expected)
            functions=[lambda:original(x,w,initial)]+[lambda cfg=cfg:forward(x,w,initial,**cfg)[0] for cfg in configs]
            entries=[graph_for(fn) for fn in functions];graphs.extend(e[0] for e in entries);samples=[[] for _ in entries]
            for repetition in range(5):
                order=list(range(len(entries)));offset=repetition%len(order);order=order[offset:]+order[:offset]
                for candidate in order:
                    graph,out,_=entries[candidate]
                    torch.xpu.synchronize();started=time.perf_counter()
                    for _ in range(20):graph.replay()
                    torch.xpu.synchronize();samples[candidate].append((time.perf_counter()-started)/20)
                    assert raw(out)==expected_bytes
            medians=[statistics.median(s) for s in samples]
            report['timings'].append(dict(case=index,samples_seconds=samples,medians_seconds=medians,speedups=[medians[0]/t for t in medians[1:]]))
            for graph,_,_ in entries:graph.reset();graphs.remove(graph)
            save();print(json.dumps(dict(timed=name,medians=medians)),flush=True)
        for config in configs:
            fusion=FusedVitProjection(model,provider,**config)
            with fusion.installed(),use_arithmetic_backend('triton') as dispatch:value=body.forward_front(model,**inputs,**options)
            equal=raw(value)==arrays.load(prior['expected_output']).tobytes()
            report['body_checks'].append(dict(config=config,byte_equal=equal,module_calls=fusion.calls,dispatch=dict(dispatch)))
            assert equal and fusion.calls==16
        report['passed']=True
except Exception as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for graph in graphs:graph.reset()
    assert all(sha(Path(p))==h for p,h in frozen.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],candidates=len(report['candidates']))),flush=True)
