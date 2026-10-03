"""Real multihead projected tensors, adversarial half bits and full NR body.

Checks every Q/K/V element before graph timing. The larger repeated tensor is
synthetic; whole-video tests, not this primitive, establish temporal behavior.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=DREF/'experimental/fused-qkv-pack-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1')
sys.path[:0]=[str(R),str(ROOT/'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js,sha
gate=authenticate_main()
prior_path=DREF/'experimental/fused-body-stages-v1/validation.json'
assert sha(prior_path)=='cefd69d92cab568b733cb83b02a5611f0edd8461daf8b2df55d08c10ba58db8c'
prior=js(prior_path);assert prior['passed']
stack_path=DREF/'results/split-heads-residual256-v1/validation.json'
assert sha(stack_path)=='0a509e1160eb176849f9a13b9b87f8cd16c5e0cfbbee2020ff5b9858903fe1d3'
stack=js(stack_path);assert stack['passed']
assert all(sha(Path(p))==h for p,h in stack['sources'].items())
assert all(sha(Path(p))==h for p,h in prior['sources'].items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as r:queue=json.load(r)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
import torch
import nr_backend.multihead_block as multihead
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from strided_batched_v2 import StridedMatrices
from fused_branched_pairs_v2 import FusedPairs
from fused_c32_mlp_v1 import FusedC32
from fused_split_ffwd_v2 import FusedSplit
from fused_swin_scheduling_v1 import FusedSwin
from fused_head_layout_v1 import HeadLayout
from window_layout_v1 import pack
from fused_qkv_pack_v1 import forward
from qkv_head_layout_v1 import QKVHeadLayout
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
paths=[Path(__file__),HERE/'Run-QKVPackV1.cmd',HERE/'fused_qkv_pack_v1.py',HERE/'qkv_head_layout_v1.py',prior_path,stack_path,*[Path(p) for p in prior['sources']],*[Path(p) for p in stack['sources']]]
frozen={str(p):sha(p) for p in paths}
OUT.mkdir()
report=dict(scope=__doc__,sources=frozen,exact_gate=gate,passed=False,cases=[],candidates=[],body_checks=[],complete_migration=False,
            timing='Static graph replay: five rotated rounds x20 calls, including output copies in every variant. Not full-frame latency.')
cases=[];graphs=[]
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
def raw(t):return t.cpu().numpy().tobytes()
def original(z,scale,order):
    q=multihead.quantize_fp8((multihead.normalize_c32(z[:,:,:,0])*scale[None,None,:,None]).half())
    k=multihead.quantize_fp8(multihead.normalize_c32(z[:,:,:,1]))
    v=multihead.quantize_fp8(z[:,:,:,2])
    return tuple(pack(t,order) for t in (q,k,v))
def graph_for(fn):
    stream=torch.xpu.Stream();torch.xpu.synchronize()
    with stream:
        for _ in range(2):warm=fn()
    torch.xpu.synchronize();out=tuple(torch.empty_like(t) for t in warm);del warm
    graph=torch.xpu.XPUGraph()
    with torch.xpu.graph(graph,stream=stream):
        temporary=fn()
        for a,b in zip(out,temporary):a.copy_(b)
    del temporary
    graph.replay();torch.xpu.synchronize()
    return graph,out,stream
try:
    torch.set_num_threads(2)
    model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    provider=StridedMatrices();provider.select('fp16_xmx')
    pairs,c32=FusedPairs(model,provider),FusedC32(model,provider,bm=32,warps=4,stages=1)
    scheduler,split=FusedSwin(provider),FusedSplit(model,provider)
    modules={id(m):name for name,m in model.named_modules() if type(m) is multihead.MultiHeadAttention}
    class Capture(HeadLayout):
        def multi(self,module,features):
            z=multihead.sm89_f16_dot(multihead.quantize_fp8(features),module.qkv,chunk_k=16).reshape(*features.shape[:2],module.heads,3,32)
            cases.append((modules[id(module)],z.clone(),module.scale,module.pixel_order,original(z,module.scale,module.pixel_order)))
            return super().multi(module,features)
    layout=Capture(model,provider)
    options=dict(sigmoid=model.sigmoid,blend_scale=model.blend_scale,return_float32=False)
    with torch.inference_mode(),provider.installed(),pairs.installed(),c32.installed(),split.installed(),scheduler.installed(),body.installed(),use_arithmetic_backend('triton'):
        inputs={name:torch.from_numpy(arrays.load(meta)).to('xpu') for name,meta in prior['body_inputs'].items()}
        with layout.installed():value=body.forward_front(model,**inputs,**options)
        assert raw(value)==arrays.load(prior['expected_output']).tobytes()
        assert len(cases)==len(modules)>0 and len({c[0] for c in cases})==len(modules)
        report['real_module_count']=len(cases)
        first=cases[0]
        # Exact source bit coverage includes +/-zero, subnormals, infinities and
        # NaN payloads. Those nonfinite cases are primitive checks, not NR inputs.
        bits=np.resize(np.arange(65536,dtype='u2'),(8,32,4,3,32)).view('f2')
        z=torch.from_numpy(bits.copy()).to('xpu');scale=torch.tensor([0.,-1.,2.,.25],dtype=torch.float16,device='xpu')
        cases.append(('synthetic_all_half_bits',z,scale,first[3],original(z,scale,first[3])))
        z=first[1].repeat((144+first[1].shape[0]-1)//first[1].shape[0],(240+first[1].shape[1]-1)//first[1].shape[1],1,1,1)[:144,:240].contiguous()
        cases.append(('synthetic_repeated_144x240',z,first[2],first[3],original(z,first[2],first[3])))
        configs=[dict(rows=r) for r in (8,16,32,64)]
        for index,(name,z,scale,order,expected) in enumerate(cases):
            expected_bytes=tuple(raw(t) for t in expected)
            meta=dict(name=name,shape=list(z.shape),synthetic=name.startswith('synthetic'),nonfinite_test='half_bits' in name,
                arrays={'z':arrays.save(z.cpu().numpy()),'scale':arrays.save(scale.cpu().numpy()),'order':arrays.save(order.cpu().numpy()),
                        **{n:arrays.save(t.cpu().numpy()) for n,t in zip(('q','k','v'),expected)}})
            report['cases'].append(meta)
            for config in configs:
                actual,kernel=forward(z,scale,order,**config)
                matches=[raw(t)==b for t,b in zip(actual,expected_bytes)]
                report['candidates'].append(dict(case=index,config=config,byte_equal=all(matches),qkv_matches=matches,spills=kernel.n_spills,registers=kernel.n_regs))
                assert all(matches),(name,config,matches)
            save();print('Byte checked '+name,flush=True)
        report['timings']=[]
        indices=[]
        for heads in (2,4,8):indices.append(next(i for i,c in enumerate(cases) if not c[0].startswith('synthetic') and c[1].shape[2]==heads))
        indices.append(len(cases)-1)
        for index in indices:
            name,z,scale,order,expected=cases[index];expected_bytes=tuple(raw(t) for t in expected)
            functions=[lambda:original(z,scale,order)]+[lambda cfg=cfg:forward(z,scale,order,**cfg)[0] for cfg in configs]
            entries=[graph_for(fn) for fn in functions];graphs.extend(e[0] for e in entries);samples=[[] for _ in entries]
            for repetition in range(5):
                order_indices=list(range(len(entries)));offset=repetition%len(entries);order_indices=order_indices[offset:]+order_indices[:offset]
                for candidate in order_indices:
                    graph,out,_=entries[candidate]
                    torch.xpu.synchronize();started=time.perf_counter()
                    for _ in range(20):graph.replay()
                    torch.xpu.synchronize();samples[candidate].append((time.perf_counter()-started)/20)
                    assert tuple(raw(t) for t in out)==expected_bytes
            medians=[statistics.median(s) for s in samples]
            report['timings'].append(dict(case=index,samples_seconds=samples,medians_seconds=medians,speedups=[medians[0]/v for v in medians[1:]]))
            for graph,_,_ in entries:graph.reset();graphs.remove(graph)
            save();print(json.dumps(dict(timed=name,medians=medians)),flush=True)
        for config in configs:
            fusion=QKVHeadLayout(model,provider,**config)
            with fusion.installed(),use_arithmetic_backend('triton') as dispatch:value=body.forward_front(model,**inputs,**options)
            equal=raw(value)==arrays.load(prior['expected_output']).tobytes()
            report['body_checks'].append(dict(config=config,byte_equal=equal,layout_calls=fusion.calls,dispatch=dict(dispatch)))
            assert equal and fusion.calls['qkv_pack']==len(modules)
        report['passed']=True
except Exception as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for graph in graphs:graph.reset()
    assert all(sha(Path(p))==h for p,h in frozen.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],candidates=len(report['candidates']))),flush=True)
