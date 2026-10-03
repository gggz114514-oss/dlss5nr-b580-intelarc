"""Validate and time native-half FMA inside current C32 and multihead attention.

Complete captured operands, special encodings, all52 real multihead modules and
the current NR256 body are compared with their frozen previous implementation.
Body ablations keep all other kernels and FP16 precision unchanged. Primitive
and body replay timings include completion and pool-external output copies.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from contextlib import ExitStack
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/native-half-attention-v1'
assert not OUT.exists();os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-sm89-v1')
sys.path[:0]=[str(EXACT/'reference'),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main();sources={};records={}
for key,folder,h in [
    ('fma','experimental/native-half-fma-v1','f7572e55a7e964f527b263a58936047aeef61693f2ce7be8ce9f42c1f7987700'),
    ('c32','experimental/c32-projection-pack-v1','10e4cbed3bb4d503e99f304c19be4ef663d1ed9d328584452943407b646f7f89'),
    ('qkv','experimental/fused-qkv-pack-v1','d5f5c510e10b444054c1be75fce45adc149c7619b9aa9e97b9912096bbed154f'),
    ('swin','experimental/fused-swin-core-v2','bea955d638c61f294970e7c91772850be4b9fc8c35f83316469f42c855736be6'),
    ('body','experimental/small-branched-mlp-operands-v1','0c3ee88b397831266e6359e5eb24db5d77f3279f5bd9c67f9bda8ae11df4c11b'),
    ('pipeline','results/c32-projection-residual256-v1','0a1cb68232d96bf24c5f8f31f57b04d537acfa413a725ee7ca53b80bbc4009d0')]:
    p=D/folder/'validation.json';assert sha(p)==h;r=js(p)
    assert r['passed'] and js(p.parent.with_suffix('.log.lease.json'))['returncode']==0
    assert all(sha(q)==h for q,h in r['sources'].items())
    records[key]=r;sources.update(r['sources']);sources[str(p)]=sha(p)
for c in records['fma']['cases'][1:]:assert c['candidates'][1]['byte_equal']
derivation=D/'experimental/native-half-attention-derivation-v1/source-derivation.json'
for c in js(derivation)['changes']:
    assert sha(c['source'])==c['source_sha256'] and sha(c['output'])==c['output_sha256']
    sources[c['source']]=c['source_sha256'];sources[c['output']]=c['output_sha256']
for p in [Path(__file__),HERE/'Run-NativeHalfAttentionV1.cmd',HERE/'native_half_attention_fma_v1.py',
          HERE/'native_half_head_layout_v1.py',HERE/'prepare_native_half_attention_v1.py',derivation]:sources[str(p)]=sha(p)
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
from fused_c32_projection_pack_v1 import forward as old_c32
from fused_c32_projection_native_half_v1 import forward as new_c32
from fused_qkv_pack_v1 import forward as old_pack
from fused_qkv_pack_native_half_v1 import forward as new_pack
from fused_swin_core_v2 import forward as old_swin
from fused_swin_core_native_half_v1 import forward as new_swin
from fused_swin_heads_v1 import forward as old_heads
from fused_swin_heads_native_half_v1 import forward as new_heads
import capture_body_v1 as body
import compressed_arrays_v1 as arrays
import immutable_artifacts_v1 as artifacts
OUT.mkdir();report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,c32=[],qkv=[],swin=[],heads=[],body=[],complete_migration=False)
graphs=[]
raw=lambda t:t.cpu().numpy().tobytes()
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')

def graph_for(fn,pool,stream):
    with stream:
        for _ in range(2):warm=fn()
    torch.xpu.synchronize();outputs=tuple(torch.empty_like(v) for v in warm);del warm
    g=torch.xpu.XPUGraph()
    with torch.xpu.graph(g,stream=stream,pool=pool):
        temporary=fn()
        for dst,src in zip(outputs,temporary):dst.copy_(src)
    del temporary
    segments=torch.xpu.memory_snapshot(pool)
    assert all(not any(s['address']<=v.data_ptr()<s['address']+s['total_size'] for s in segments) for v in outputs)
    graphs.append(g);return g,outputs

def time_functions(functions,expected):
    pool=torch.xpu.graph_pool_handle();stream=torch.xpu.Stream()
    entries=[graph_for(fn,pool,stream) for fn in functions];samples=[[] for _ in entries];orders=[]
    for r in range(5):
        order=list(range(len(entries)));shift=r%len(entries);order=order[shift:]+order[:shift];orders.append(order)
        for i in order:
            g,values=entries[i];torch.xpu.synchronize();started=time.perf_counter()
            for _ in range(10):g.replay()
            torch.xpu.synchronize();samples[i].append((time.perf_counter()-started)/10)
            assert tuple(raw(v) for v in values)==expected
    for g,_ in entries:g.reset();graphs.remove(g)
    return dict(samples_seconds=samples,median_seconds=[statistics.median(s) for s in samples],orders=orders)

def compare(section,label,meta,old,new,timed):
    before,old_kernel=old();expected=tuple(raw(v) for v in before)
    values,compiled=new();assert tuple(raw(v) for v in values)==expected,(section,label)
    row=dict(name=label,arrays=meta,expected=[arrays.save(v.cpu().numpy()) for v in before],byte_equal=True,timed=timed,
             spills=compiled.n_spills,ir={k:artifacts.text(v,k) for k,v in compiled.asm.items() if isinstance(v,str) and k in ('ttgir','llir')})
    report[section].append(row)
    if timed:row.update(time_functions([lambda:old()[0],lambda:new()[0]],expected))
    save()
    if timed:print(json.dumps(dict(section=section,name=label,median_ms=[s*1000 for s in row['median_seconds']])),flush=True)
    return expected

try:
    torch.set_num_threads(2)
    model=MotionNR.from_assets(EXACT/'model-assets/sf-v2/WEIGHTS_HT.bin',EXACT/'model-assets/noise-sm89-v2',EXACT/'model-assets/sigmoid-sm89-v1').to('xpu').eval()
    lut=register(model);provider=StridedMatrices();provider.select('fp16_xmx');modules=dict(model.named_modules())
    with torch.inference_mode(),provider.installed(),body.installed(),use_arithmetic_backend('triton'):
        shapes=set()
        for c in records['c32']['cases']:
            meta=c['arrays'];t={k:torch.from_numpy(arrays.load(v)).to('xpu') for k,v in meta.items() if k not in ('q','k','v')}
            args=(t['x'],t['qkv_weight'],t['scale'],t['order']);shape=tuple(c['shape']);timed=shape not in shapes;shapes.add(shape)
            expected=compare('c32',c['name'],meta,lambda:old_c32(*args,bm=32),lambda:new_c32(*args,bm=32),timed)
            assert expected==tuple(arrays.load(meta[k]).tobytes() for k in ('q','k','v'))
            assert all(raw(v)==arrays.load(meta[k]).tobytes() for k,v in t.items())
        shapes=set()
        for c in records['qkv']['cases']:
            meta=c['arrays'];t={k:torch.from_numpy(arrays.load(v)).to('xpu') for k,v in meta.items() if k not in ('q','k','v')}
            args=(t['z'],t['scale'],t['order']);shape=tuple(c['shape']);timed=shape not in shapes;shapes.add(shape)
            expected=compare('qkv',c['name'],meta,lambda:old_pack(*args),lambda:new_pack(*args),timed)
            assert expected==tuple(arrays.load(meta[k]).tobytes() for k in ('q','k','v'))
            q,k,v=[torch.from_numpy(arrays.load(meta[n])).to('xpu') for n in ('q','k','v')]
            if not c['synthetic']:bias=modules[c['name']].bias
            else:bias=next(m.bias for m in modules.values() if hasattr(m,'bias') and tuple(m.bias.shape)==(q.shape[0],64,64))
            head_meta={n:meta[n] for n in ('q','k','v')};head_meta['bias']=arrays.save(bias.cpu().numpy())
            def old_head_call():value,compiled=old_heads(q,k,v,bias);return (value,),compiled
            def new_head_call():value,compiled=new_heads(q,k,v,bias);return (value,),compiled
            compare('heads',c['name'],head_meta,old_head_call,new_head_call,timed)
            assert all(raw(v)==arrays.load(meta[k]).tobytes() for k,v in t.items())
        for i,c in enumerate(records['swin']['cases']):
            meta=c['operands'];t={k:torch.from_numpy(arrays.load(v)).to('xpu') for k,v in meta.items()}
            args=(t['query'],t['key'],t['value'],t['bias'])
            def old():v,k=old_swin(*args,bm=32);return (v,),k
            def new():v,k=new_swin(*args,bm=32);return (v,),k
            compare('swin',str(i),meta,old,new,True)
            assert all(raw(v)==arrays.load(meta[k]).tobytes() for k,v in t.items())
        inputs={n:torch.from_numpy(arrays.load(m)).to('xpu') for n,m in records['body']['body_inputs'].items()}
        expected=(arrays.load(records['body']['expected_output']).tobytes(),)
        common=[FusedBatched(model,provider,workload='small'),FusedSplit(model,provider),FusedC32(model,provider),FusedVitProjection(model,provider)]
        variants=[('previous',ProjectedHeadLayout(model,provider),OldSwin(provider)),
            ('normalize',HeadLayout(model,provider,normalize=True,swin=False),OldSwin(provider)),
            ('swin',HeadLayout(model,provider,normalize=False,swin=True),NewSwin(provider)),
            ('both',HeadLayout(model,provider),NewSwin(provider))]
        fns=[]
        for name,layout,swin in variants:
            def fn(layout=layout,swin=swin):
                with ExitStack() as stack:
                    for component in [*common,layout,swin]:stack.enter_context(component.installed())
                    with use_arithmetic_backend('triton') as dispatch:
                        value=body.forward_front(model,**inputs,sigmoid=model.sigmoid,blend_scale=model.blend_scale,return_float32=False)
                    assert dict(dispatch)==records['body']['dispatch']
                    return (value,)
            values=fn();assert tuple(raw(v) for v in values)==expected,name
            report['body'].append(dict(name=name,byte_equal=True,dispatch=records['body']['dispatch'],layout_calls=dict(layout.calls)))
            fns.append(fn)
        report['body_timing']=time_functions(fns,expected)
        print(json.dumps(dict(body_names=[x[0] for x in variants],body_median_ms=[v*1000 for v in report['body_timing']['median_seconds']])),flush=True)
        assert raw(Constant(model).require())==np.load(TABLE,allow_pickle=False).view('i2').tobytes()
        assert all(raw(v)==arrays.load(records['body']['body_inputs'][k]).tobytes() for k,v in inputs.items())
        report.update(passed=True,lut_bytes_unchanged=True,operands_unchanged=True,primitive_comparisons=sum(len(report[k]) for k in ('c32','qkv','heads','swin')))
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for g in graphs:g.reset()
    assert all(sha(p)==h for p,h in sources.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],body=report.get('body_timing'))),flush=True)
