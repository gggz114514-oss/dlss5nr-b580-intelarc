"""Compare fused C32 projection/pack against real complete recorded QKV windows.

Reuse10 real NR256 cases and two explicitly synthetic larger canvases. Two new
small synthetic cases test a zero input and non-square window mapping. Every
full Q/K/V output is checked before timing and after graph replay. Complete body
checks use the current batched-MLP/C32-LUT stack. Timings include identical
pool-external output copies, and are not full-frame throughput.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/c32-projection-pack-v1'
assert not OUT.exists();os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-sm89-v1')
sys.path[:0]=[str(EXACT/'reference'),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main();sources={};records={}
for key,folder,digest in (
    ('cases','experimental/c32-chunk-pack-v3','fc384ac9ccb36e24ae389a2d4e047d2f329493d9ac5b1ad0ea44c797254c79e2'),
    ('pipeline','results/batched-branches-residual256-v1','e0627317e6b1abdf15b25dca116ab93b3ce7bcad38e6bfbca9f117630dcecb42'),
    ('body','experimental/small-branched-mlp-operands-v1','0c3ee88b397831266e6359e5eb24db5d77f3279f5bd9c67f9bda8ae11df4c11b')):
    p=D/folder/'validation.json';assert sha(p)==digest;r=js(p)
    assert r['passed'] and js(p.parent.with_suffix('.log.lease.json'))['returncode']==0
    assert all(sha(p)==h for p,h in r['sources'].items())
    records[key]=r;sources.update(r['sources']);sources[str(p)]=digest
for p in (Path(__file__),HERE/'Run-C32ProjectionPackV1.cmd',HERE/'fused_c32_projection_pack_v1.py'):sources[str(p)]=sha(p)
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
from fused_swin_scheduling_v1 import FusedSwin
from c32_chunk_layout_v2 import prepare,ChunkedHeadLayout
from fused_c32_projection_pack_v1 import forward,ProjectedHeadLayout
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
import immutable_artifacts_v1 as artifacts
configs=[dict(bm=bm,warps=warps,stages=stages) for bm in (16,32) for warps in (4,8) for stages in (1,2)]
OUT.mkdir();report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,
    cases=[],body_checks=[],complete_migration=False)
graphs=[]

def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

def raw(t):return t.cpu().numpy().tobytes()

def graph_for(fn,pool,stream):
    with stream:
        for _ in range(2):warm=fn()
    torch.xpu.synchronize();outputs=tuple(torch.empty_like(v) for v in warm);del warm
    graph=torch.xpu.XPUGraph()
    with torch.xpu.graph(graph,stream=stream,pool=pool):
        temporary=fn()
        for dst,src in zip(outputs,temporary):dst.copy_(src)
    del temporary
    segments=torch.xpu.memory_snapshot(pool)
    assert all(not any(s['address']<=v.data_ptr()<s['address']+s['total_size'] for s in segments) for v in outputs)
    return graph,outputs

try:
    torch.set_num_threads(2)
    model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    lut=register(model);constant=Constant(model);provider=StridedMatrices();provider.select('fp16_xmx')
    with torch.inference_mode(),provider.installed(),body.installed(),use_arithmetic_backend('triton'):
        cases=list(records['cases']['cases'])
        base=cases[0]['arrays']
        for name,shape in (('synthetic_zero_8x8',(8,8,32)),('synthetic_repeated_24x40',(24,40,32))):
            x=np.zeros(shape,dtype='f2') if 'zero' in name else np.resize(arrays.load(base['x']),shape)
            meta={n:m for n,m in base.items() if n not in ('x','q','k','v')};meta['x']=arrays.save(x)
            cases.append(dict(name=name,shape=list(shape),synthetic=True,arrays=meta))
        shapes=set()
        for ordinal,case in enumerate(cases):
            metadata=dict(case['arrays']);saved={n:arrays.load(m) for n,m in metadata.items()}
            tensors={n:torch.from_numpy(v).to('xpu') for n,v in saved.items() if n not in ('q','k','v')}
            module=SimpleNamespace(front=SimpleNamespace(qkv=tensors['qkv_weight'],scale=tensors['scale']),pixel_inverse=tensors['inverse'])
            previous=lambda:prepare(module,tensors['x'],rows=16)
            baseline=previous();expected=tuple(raw(v) for v in baseline)
            for name,value,data in zip(('q','k','v'),baseline,expected):
                if name in saved:assert saved[name].tobytes()==data
                else:metadata[name]=arrays.save(value.cpu().numpy())
            del baseline
            shape=tuple(case['shape']);timed=shape not in shapes;shapes.add(shape)
            row=dict(name=case['name'],shape=case['shape'],synthetic=case['synthetic'],arrays=metadata,
                     timed=timed,candidates=[]);report['cases'].append(row)
            functions=[previous]
            for config in configs:
                fn=lambda config=config:forward(tensors['x'],tensors['qkv_weight'],tensors['scale'],tensors['order'],**config)
                values,kernel=fn();equal=tuple(raw(v) for v in values)==expected
                ir=str(kernel.asm['ttgir']);assert 'ttig.dpas' in ir
                row['candidates'].append(dict(config=config,byte_equal=equal,spills=kernel.n_spills,ttgir=artifacts.text(ir,'ttgir')))
                assert equal,(case['name'],config)
                functions.append(lambda fn=fn:fn()[0])
            if timed:
                pool=torch.xpu.graph_pool_handle();stream=torch.xpu.Stream()
                entries=[graph_for(fn,pool,stream) for fn in functions];graphs.extend(g for g,_ in entries)
                samples=[[] for _ in entries];orders=[]
                for repetition in range(5):
                    order=list(range(len(entries)));shift=(2*repetition)%len(order);order=order[shift:]+order[:shift];orders.append(order)
                    for n in order:
                        graph,outputs=entries[n];torch.xpu.synchronize();started=time.perf_counter()
                        for _ in range(10):graph.replay()
                        torch.xpu.synchronize();samples[n].append((time.perf_counter()-started)/10)
                        assert tuple(raw(v) for v in outputs)==expected
                row.update(samples_seconds=samples,median_seconds=[statistics.median(s) for s in samples],orders=orders)
                for graph,_ in entries:graph.reset();graphs.remove(graph)
                del entries,pool,stream
                print(json.dumps(dict(name=case['name'],median_ms=[s*1000 for s in row['median_seconds']])),flush=True)
            assert all(raw(t)==saved[n].tobytes() for n,t in tensors.items())
            row['operands_unchanged']=True;save()
        real=[v for v in report['cases'] if v['timed'] and not v['synthetic']]
        selected=min(range(len(configs)),key=lambda i:statistics.mean(v['median_seconds'][i+1] for v in real))
        report['selected_config']=configs[selected]
        input_meta=records['body']['body_inputs'];inputs={n:torch.from_numpy(arrays.load(m)).to('xpu') for n,m in input_meta.items()}
        expected_output=arrays.load(records['body']['expected_output']).tobytes()
        split=FusedSplit(model,provider);c32=FusedC32(model,provider);batched=FusedBatched(model,provider,workload='small')
        vit=FusedVitProjection(model,provider);swin=FusedSwin(provider)
        with split.installed(),c32.installed(),batched.installed(),vit.installed(),swin.installed():
            for name,layout in (('previous',ChunkedHeadLayout(model,provider)),('fused',ProjectedHeadLayout(model,provider,**configs[selected]))):
                with layout.installed(),use_arithmetic_backend('triton') as dispatch:
                    value=body.forward_front(model,**inputs,sigmoid=model.sigmoid,blend_scale=model.blend_scale,return_float32=False)
                assert raw(value)==expected_output and dict(dispatch)==records['body']['dispatch']
                report['body_checks'].append(dict(name=name,byte_equal=True,dispatch=dict(dispatch),layout_calls=dict(layout.calls)))
            assert report['body_checks'][1]['layout_calls']['c32_projection_pack']==10
        assert constant.require() is lut and raw(lut)==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
        report.update(passed=True,lut_unchanged=True,primitive_comparisons=len(cases)*len(configs))
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for graph in graphs:graph.reset()
    assert all(sha(p)==h for p,h in sources.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],selected=report.get('selected_config'))),flush=True)
