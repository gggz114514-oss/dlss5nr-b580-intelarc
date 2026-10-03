"""Real C32 fronts, large repeated canvases and bounded scatter edge checks.

Complete Q/K/V matrices compared for every candidate. Synthetic large canvases
are repetitions, not native full-frame captures. The edge case covers all half
bits, partial chunks, scanline/window boundaries and untouched output guards.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=DREF/'experimental/c32-chunk-pack-v3'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1')
sys.path[:0]=[str(R),str(ROOT/'backend')]
from attention_weights_main_gate_v1 import authenticate_main
from analyze_auxiliary_captures import js,sha
gate=authenticate_main()
prior_path=DREF/'experimental/fused-body-stages-v2/validation.json'
assert sha(prior_path)=='85462097455e379845eb3c1244f22fa91329dd5ddaaf708d438424dd58f3efac'
prior=js(prior_path)
stack_path=DREF/'results/vit-projection-residual256-v1/validation.json'
assert sha(stack_path)=='fb17fe93c6bc93036fa11a79465e3e27e474a52574fff6aa9bd4a8c69330418d'
stack=js(stack_path)
for path,r in ((prior_path,prior),(stack_path,stack)):
    assert r['passed'] and js(path.parent.with_suffix('.log.lease.json'))['returncode']==0
    assert all(sha(Path(p))==h for p,h in r['sources'].items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as r:queue=json.load(r)
assert not queue['queue_running'] and not queue['queue_pending']
import numpy as np
import torch
import nr_backend.attention as attention
from nr_backend.temporal import MotionNR
from nr_backend.execution import use_arithmetic_backend
from strided_batched_v2 import StridedMatrices
from fused_branched_pairs_v2 import FusedPairs
from fused_c32_mlp_v1 import FusedC32
from fused_split_ffwd_v2 import FusedSplit
from fused_vit_projection_v3 import FusedVitProjection
from fused_swin_scheduling_v1 import FusedSwin
from qkv_head_layout_v1 import QKVHeadLayout
from window_layout_v1 import pack
from c32_chunk_layout_v2 import prepare,ChunkedHeadLayout
from fused_c32_qkv_scatter_v2 import scatter
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
paths=[Path(__file__),HERE/'Run-C32ChunkPackV3.cmd',HERE/'fused_c32_qkv_scatter_v2.py',HERE/'c32_chunk_layout_v2.py',prior_path,stack_path,*[Path(p) for p in prior['sources']],*[Path(p) for p in stack['sources']]]
frozen={str(p):sha(p) for p in paths}
OUT.mkdir()
report=dict(scope=__doc__,sources=frozen,exact_gate=gate,passed=False,cases=[],candidates=[],body_checks=[],edge_checks=[],complete_migration=False,
            timing='Static graph replay: five rotated rounds x10 calls, with matching copies of complete QKV outputs. Not model/frame latency.')
cases=[];graphs=[]
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
def raw(t):return t.cpu().numpy().tobytes()
def original(module,x):
    h,w=x.shape[:2];q,k,v=module.front(x)
    return tuple(pack(t.reshape(h,w,1,32),module.pixel_order)[0] for t in (q,k,attention.quantize_fp8(v)))
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
    scheduler,split,vit=FusedSwin(provider),FusedSplit(model,provider),FusedVitProjection(model,provider)
    modules={id(m):name for name,m in model.named_modules() if isinstance(m,attention.C32AttentionCore)}
    class Capture(QKVHeadLayout):
        def c32(self,module,features):
            cases.append((modules[id(module)],features.clone(),module,original(module,features)))
            return super().c32(module,features)
    layout=Capture(model,provider)
    options=dict(sigmoid=model.sigmoid,blend_scale=model.blend_scale,return_float32=False)
    with torch.inference_mode(),provider.installed(),pairs.installed(),c32.installed(),split.installed(),vit.installed(),scheduler.installed(),body.installed(),use_arithmetic_backend('triton'):
        inputs={name:torch.from_numpy(arrays.load(meta)).to('xpu') for name,meta in prior['body_inputs'].items()}
        with layout.installed():value=body.forward_front(model,**inputs,**options)
        assert raw(value)==arrays.load(prior['expected_output']).tobytes()
        assert len(cases)==len(modules)==10 and len({c[0] for c in cases})==10
        first=cases[0]
        for h,w in ((512,896),(1152,1920)):
            x=first[1].repeat((h+first[1].shape[0]-1)//first[1].shape[0],(w+first[1].shape[1]-1)//first[1].shape[1],1)[:h,:w].contiguous()
            cases.append((f'synthetic_repeated_{h}x{w}',x,first[2],original(first[2],x)))
        configs=[dict(rows=r) for r in (8,16,32,64)]
        for index,(name,x,module,expected) in enumerate(cases):
            expected_bytes=tuple(raw(t) for t in expected)
            report['cases'].append(dict(name=name,shape=list(x.shape),synthetic=name.startswith('synthetic'),chunks=(x.numel()//32+32767)//32768,
                arrays={'x':arrays.save(x.cpu().numpy()),'qkv_weight':arrays.save(module.front.qkv.cpu().numpy()),'scale':arrays.save(module.front.scale.cpu().numpy()),
                        'order':arrays.save(module.pixel_order.cpu().numpy()),'inverse':arrays.save(module.pixel_inverse.cpu().numpy()),
                        **{n:arrays.save(t.cpu().numpy()) for n,t in zip(('q','k','v'),expected)}}))
            for config in configs:
                with use_arithmetic_backend('triton') as dispatch:actual=prepare(module,x,**config)
                matches=[raw(t)==b for t,b in zip(actual,expected_bytes)]
                chunks=(x.numel()//32+32767)//32768
                assert dict(dispatch)==dict(backend='triton',dense=chunks,batched=0,attention_normalize_c32=2*chunks,fp8=2*chunks+1),dict(dispatch)
                report['candidates'].append(dict(case=index,config=config,byte_equal=all(matches),qkv_matches=matches,dispatch=dict(dispatch)))
                assert all(matches),(name,config,matches)
                del actual
            save();print('Byte checked '+name,flush=True)
        # Arbitrary chunk offsets, including non-window-aligned and tail rows.
        zcpu=np.resize(np.arange(65536,dtype='u2'),(768,96)).view('f2')
        z=torch.from_numpy(zcpu.copy()).to('xpu');module=first[2];scale=torch.tensor([-1.],dtype=torch.float16,device='xpu')
        q=attention.quantize_fp8((attention.normalize_c32(z[:,:32])*scale).half())
        k=attention.quantize_fp8(attention.normalize_c32(z[:,32:64]));v=attention.quantize_fp8(z[:,64:])
        expected=tuple(pack(t.reshape(24,32,1,32),module.pixel_order)[0].cpu().numpy() for t in (q,k,v))
        report['edge_arrays']={'z':arrays.save(zcpu),'scale':arrays.save(scale.cpu().numpy()),'inverse':arrays.save(module.pixel_inverse.cpu().numpy()),**{n:arrays.save(t) for n,t in zip(('q','k','v'),expected)}}
        inverse=module.pixel_inverse.cpu().numpy();shape=(3,4,64,32);numel=768*32;sentinel=0x7e55
        for config in configs:
            bases=[torch.full((numel+256,),sentinel,dtype=torch.int16,device='xpu') for _ in range(3)]
            outputs=tuple(t[128:-128].view(torch.float16).reshape(shape) for t in bases)
            start=0;covered=np.zeros(768,dtype=bool);checks=[]
            for count in (19,1,37,211,256,244):
                kernel=scatter(z[start:start+count],scale,module.pixel_inverse,outputs,start,**config)
                indices=np.arange(start,start+count);y,x=indices//32,indices%32
                slots=((y//8)*4+x//8)*64+inverse[(y%8)*8+x%8];covered[slots]=True
                for actual,base,target in zip(outputs,bases,expected):
                    observed=actual.cpu().numpy().view('u2').reshape(768,32)
                    wanted=np.full((768,32),sentinel,dtype='u2');wanted[covered]=target.view('u2').reshape(768,32)[covered]
                    assert observed.tobytes()==wanted.tobytes()
                    bits=base.cpu().numpy().view('u2');assert (bits[:128]==sentinel).all() and (bits[-128:]==sentinel).all()
                checks.append(dict(start=start,count=count,complete_buffer_matches=True,guards_unchanged=True,spills=kernel.n_spills));start+=count
            assert start==768 and covered.all()
            report['edge_checks'].append(dict(config=config,chunks=checks,all_half_bits=True))
        report['timings']=[]
        # Time full pre/post, two intermediate fronts and both larger canvases.
        for index in (0,1,8,9,10,11):
            name,x,module,expected=cases[index];expected_bytes=tuple(raw(t) for t in expected)
            functions=[lambda:original(module,x)]+[lambda cfg=cfg:prepare(module,x,**cfg) for cfg in configs]
            entries=[graph_for(fn) for fn in functions];graphs.extend(e[0] for e in entries);samples=[[] for _ in entries]
            for repetition in range(5):
                order=list(range(len(entries)));offset=repetition%len(order);order=order[offset:]+order[:offset]
                for candidate in order:
                    graph,out,_=entries[candidate];torch.xpu.synchronize();started=time.perf_counter()
                    for _ in range(10):graph.replay()
                    torch.xpu.synchronize();samples[candidate].append((time.perf_counter()-started)/10)
                    assert tuple(raw(t) for t in out)==expected_bytes
            medians=[statistics.median(s) for s in samples]
            report['timings'].append(dict(case=index,samples_seconds=samples,medians_seconds=medians,speedups=[medians[0]/t for t in medians[1:]]))
            for graph,_,_ in entries:graph.reset();graphs.remove(graph)
            del entries;torch.xpu.empty_cache()
            save();print(json.dumps(dict(timed=name,medians=medians)),flush=True)
        for config in configs:
            fusion=ChunkedHeadLayout(model,provider,c32_rows=config['rows'])
            with fusion.installed(),use_arithmetic_backend('triton') as dispatch:value=body.forward_front(model,**inputs,**options)
            equal=raw(value)==arrays.load(prior['expected_output']).tobytes()
            report['body_checks'].append(dict(config=config,byte_equal=equal,layout_calls=fusion.calls,dispatch=dict(dispatch)))
            assert equal and fusion.calls['c32']==10 and fusion.calls['c32_chunk_pack']==sum(c['chunks'] for c in report['cases'][:10])
        report['passed']=True
except Exception as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for graph in graphs:graph.reset()
    assert all(sha(Path(p))==h for p,h in frozen.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],candidates=len(report['candidates']))),flush=True)
