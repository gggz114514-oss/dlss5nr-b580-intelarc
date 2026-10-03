"""Real sixteen-module C512 group fusion checks, tuning and complete-body replay.

Temporal NR256 body inputs are authenticated from the prior stage profile.
One eager whole-body pass captures each actual projected group input and output.
All candidates compare complete tensors before static graph timing. Synthetic
19-row tails and repeated larger row counts are labeled, not real large frames.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=DREF/'experimental/fused-split-ffwd-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1')
sys.path[:0]=[str(R),str(ROOT/'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js,sha
gate=authenticate_main()
prior_path=DREF/'experimental/fused-body-stages-v1/validation.json'
assert sha(prior_path)=='cefd69d92cab568b733cb83b02a5611f0edd8461daf8b2df55d08c10ba58db8c'
prior=js(prior_path);assert prior['passed']
assert all(sha(Path(p))==h for p,h in prior['sources'].items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as r:queue=json.load(r)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
import torch
import nr_backend.split_block as split
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from strided_batched_v2 import StridedMatrices
from fused_branched_pairs_v2 import FusedPairs
from fused_c32_mlp_v1 import FusedC32
from fused_swin_scheduling_v1 import FusedSwin
from window_layout_v1 import WindowLayout
from fused_split_ffwd_v1 import forward,FusedSplit
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
paths=[Path(__file__),HERE/'Run-SplitFFWDV1.cmd',HERE/'fused_split_ffwd_v1.py',prior_path,*[Path(p) for p in prior['sources']]]
frozen={str(p):sha(p) for p in paths}
OUT.mkdir()
report=dict(scope=__doc__,sources=frozen,exact_gate=gate,passed=False,cases=[],candidates=[],body_checks=[],complete_migration=False,
            timing='Static graph replay: 5 rotated rounds x20 calls, graph creation/compilation and validation excluded; '
                   'not an entire frame or image-quality claim.')
cases=[];graphs=[]
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
def raw(t):return t.cpu().numpy().tobytes()
def original_groups(z,expand,reduce):
    parts=[]
    for i in range(8):
        hidden=split.cubic_quantize(split.dot(z[...,i*64:(i+1)*64],expand[i],chunk_k=16))
        parts.append(split.q(split.dot(hidden,reduce[i],chunk_k=16)))
    return torch.cat(parts,dim=-1)
def graph_for(fn):
    stream=torch.xpu.Stream();torch.xpu.synchronize()
    with stream:
        for _ in range(2):warm=fn()
    torch.xpu.synchronize()
    out=torch.empty_like(warm);del warm
    graph=torch.xpu.XPUGraph()
    with torch.xpu.graph(graph,stream=stream):
        temporary=fn();out.copy_(temporary)
    del temporary
    graph.replay();torch.xpu.synchronize()
    return graph,out,stream
try:
    torch.set_num_threads(2)
    model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    provider=StridedMatrices();provider.select('fp16_xmx')
    pairs,c32=FusedPairs(model,provider),FusedC32(model,provider,bm=32,warps=4,stages=1)
    scheduler,layout=FusedSwin(provider),WindowLayout()
    options=dict(sigmoid=model.sigmoid,blend_scale=model.blend_scale,return_float32=False)
    with torch.inference_mode(),provider.installed(),pairs.installed(),c32.installed(),scheduler.installed(),layout.installed(),body.installed(),use_arithmetic_backend('triton'):
        inputs={name:torch.from_numpy(arrays.load(meta)).to('xpu') for name,meta in prior['body_inputs'].items()}
        modules={id(m):name for name,m in model.named_modules() if type(m) is split.SplitFeedForward}
        original=split.SplitFeedForward.forward
        def capture(module,features):
            z=split.q(split.dot(split.q(features),module.linear,chunk_k=16))
            value=original_groups(z,module.expand,module.reduce)
            cases.append((modules[id(module)],z.clone(),module.expand,module.reduce,value.clone()))
            return value
        split.SplitFeedForward.forward=capture
        try:value=body.forward_front(model,**inputs,**options)
        finally:split.SplitFeedForward.forward=original
        assert raw(value)==arrays.load(prior['expected_output']).tobytes()
        assert len(cases)==16
        first=cases[0]
        for count in (19,1792,8640):
            z=first[1].reshape(-1,512).repeat((count+first[1].numel()//512-1)//(first[1].numel()//512),1)[:count].contiguous()
            cases.append((f'synthetic_repeated_or_tail_{count}',z,first[2],first[3],original_groups(z,first[2],first[3])))
        configs=[dict(bm=bm,bn=bn,stages=stage) for bm,bn,stage in ((16,64,1),(16,64,2),(32,64,1),(32,64,2),(16,32,1))]
        for index,(name,z,expand,reduce,expected) in enumerate(cases):
            expected_bytes=raw(expected)
            meta=dict(name=name,shape=list(z.shape),synthetic=name.startswith('synthetic'),arrays={
                'z':arrays.save(z.cpu().numpy()),'expand':arrays.save(expand.cpu().numpy()),
                'reduce':arrays.save(reduce.cpu().numpy()),'expected':arrays.save(expected.cpu().numpy())})
            report['cases'].append(meta)
            for config in configs:
                actual,kernel=forward(z,expand,reduce,**config)
                equal=raw(actual)==expected_bytes
                row=dict(case=index,config=config,byte_equal=equal,spills=kernel.n_spills,registers=kernel.n_regs)
                report['candidates'].append(row)
                assert equal,(name,config)
            print('Byte checked '+name,flush=True)
        # Representative actual module and synthetic large/tail workloads.
        report['timings']=[]
        for index in (0,7,8,15,16,17,18):
            name,z,expand,reduce,expected=cases[index]
            expected_bytes=raw(expected)
            functions=[lambda:original_groups(z,expand,reduce)] + [lambda cfg=cfg:forward(z,expand,reduce,**cfg)[0] for cfg in configs]
            entries=[graph_for(fn) for fn in functions];graphs.extend(e[0] for e in entries)
            samples=[[] for _ in entries]
            for repetition in range(5):
                order=list(range(len(entries)));offset=repetition%len(order);order=order[offset:]+order[:offset]
                for candidate in order:
                    graph,out,_=entries[candidate]
                    torch.xpu.synchronize();started=time.perf_counter()
                    for _ in range(20):graph.replay()
                    torch.xpu.synchronize();samples[candidate].append((time.perf_counter()-started)/20)
                    assert raw(out)==expected_bytes
            medians=[statistics.median(s) for s in samples]
            report['timings'].append(dict(case=index,samples_seconds=samples,medians_seconds=medians,speedups=[medians[0]/v for v in medians[1:]]))
            for graph,_,_ in entries:graph.reset();graphs.remove(graph)
            print(json.dumps(dict(timed=name,medians=medians)),flush=True)
        # Complete body with every candidate; all intermediate fused modules are real.
        for config in configs:
            fusion=FusedSplit(model,provider,**config)
            with fusion.installed():value=body.forward_front(model,**inputs,**options)
            equal=raw(value)==arrays.load(prior['expected_output']).tobytes()
            report['body_checks'].append(dict(config=config,byte_equal=equal,module_calls=fusion.calls))
            assert equal and fusion.calls==16
        report['passed']=True
except Exception as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for graph in graphs:graph.reset()
    assert all(sha(Path(p))==h for p,h in frozen.items())
    authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],candidates=len(report['candidates']))),flush=True)
