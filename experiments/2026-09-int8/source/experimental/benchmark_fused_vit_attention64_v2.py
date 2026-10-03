"""Fuse full ViT64 attention with native-half exponent FMA; compare all boundaries and all half exponent inputs.

Eight actual QKV input sets plus synthetic finite/extreme/layout cases. Preserve
the selected FP16 provider, original exponent, numerator-before-normalization,
and half boundaries. Five rotated timing rounds, four calls per graph, ten
replays per sample. These local attention timings are not complete NR latency.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/fused-vit-attention64-v2';assert not OUT.exists()
TOOLCHAIN=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
sys.path[:0]=[str(TOOLCHAIN/'site'),str(EXACT/'reference'),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main();sources={};records={}
for name,folder,pin in [('qkv','experimental/fused-vit-qkv-v1','0b69ebfe1d836a30724dfd4a92794cd3cbf4bfd29e647486958736b4d053c8e6'),
                      ('current','results/native-cubic-residual256-v1','089c9dbef37d2461b1525b16e63a0fd56470209503b5332628e6d9c35def47f4')]:
    p=D/folder/'validation.json';assert sha(p)==pin
    r=js(p);assert r['passed'] and js(p.parent.with_suffix('.log.lease.json'))['returncode']==0
    assert all(sha(p)==h for p,h in r['sources'].items());sources.update(r['sources']);sources[str(p)]=pin;records[name]=r
provision=TOOLCHAIN/'provision-v1.json'
assert sha(provision)=='e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN/'site'/p)==h for p,h in js(provision)['files'].items())
for p in [Path(__file__),HERE/'Run-FusedVitAttention64V2.cmd',HERE/'fused_vit_attention64_v2.py',HERE/'fused_vit_attention64_v1.py',HERE/'native_half_attention_fma_v1.py',HERE/'strided_batched_v2.py']:
    sources[str(p)]=sha(p)
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
import torch,triton
import nr_backend.vit_block as vb
from nr_backend.execution import use_arithmetic_backend
from strided_batched_v2 import StridedMatrices
from fused_vit_attention64_v2 import forward,_exponential
from fused_vit_attention64_v1 import forward as compensated
from nr_backend.triton_attention_exp import exponential
import triton.language as tl
import compressed_arrays_v1 as arrays
import immutable_artifacts_v1 as artifacts
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
OUT.mkdir();report=dict(scope=__doc__,passed=False,complete_migration=False,sources=sources,exact_gate=gate,cases=[],rounds=5,calls_per_graph=4,replays_per_sample=10)
report['runtime']=dict(torch=torch.__version__,triton=triton.__version__,triton_file=triton.__file__,isolated=True)
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
def raw(t):return t.cpu().numpy().tobytes()
def previous(q,k,v,debug=False):
    if not debug:return vb.vit_attention(q,k,v)
    scores=vb.sm89_f16_batched_dot(q,k.transpose(-1,-2))
    e=vb.vit_exponential(scores)
    num=vb.sm89_f16_batched_dot(vb.q(e),v)
    den=vb.attention_row_sum64(e)
    rcp=den.clamp(min=.00006198883056640625).float().reciprocal().half()
    output=vb.q((num*rcp).half())
    return output,dict(scores=scores,exponential=e,numerator=num,denominator=den,reciprocal=rcp)
configs=[dict(bm=16,warps=4,stages=1),dict(bm=32,warps=4,stages=1),dict(bm=64,warps=4,stages=1)]
@triton.jit
def exponent_probe(X,Y,N:tl.constexpr):
    i=tl.program_id(0)*512+tl.arange(0,512)
    x=tl.load(X+i,i<N,other=0)
    tl.store(Y+i,_exponential(x),i<N)

graphs=[]
try:
    torch.set_num_threads(2);provider=StridedMatrices();provider.select('fp16_xmx')
    with torch.inference_mode(),provider.installed(),use_arithmetic_backend('triton'):
        half_inputs=np.arange(65536,dtype='u2').view('f2')
        score_inputs=torch.from_numpy(half_inputs.copy()).to('xpu');old_exponential=exponential(score_inputs,vit=True)
        native_exponential=torch.empty_like(score_inputs)
        compiled=exponent_probe[(128,)](score_inputs,native_exponential,65536,num_warps=4,enable_fp_fusion=False)
        assert raw(native_exponential)==raw(old_exponential) and raw(score_inputs)==half_inputs.tobytes()
        assert 'llvm.fma.f16' in str(compiled.asm['llir'])
        report['all_half_exponent']=dict(byte_equal=True,inputs=arrays.save(half_inputs),expected=arrays.save(old_exponential.cpu().numpy()),actual=arrays.save(native_exponential.cpu().numpy()),llir=artifacts.text(str(compiled.asm['llir']),'llir'))
        save();print('All65536 half exponential inputs match',flush=True)
        cases=[]
        for row in records['qkv']['cases']:
            values=[torch.from_numpy(arrays.load(meta).copy()).to('xpu').transpose(0,1) for meta in row['outputs']]
            cases.append((f"block{row['index']}",values,True))
        reference=cases[0][1];shape=(3,64,32)
        cases.append(('zero',[torch.zeros(shape,dtype=torch.float16,device='xpu') for _ in range(3)],False))
        cases.append(('signed_zero',[torch.full(shape,-0. if i==0 else 0.,dtype=torch.float16,device='xpu') for i in range(3)],False))
        cases.append(('positive_saturation',[torch.full(shape,448.,dtype=torch.float16,device='xpu') for _ in range(3)],False))
        rng=np.random.Generator(np.random.PCG64(0x580A7764))
        mixed=[]
        for _ in range(3):
            a=rng.choice(np.array([-448.,-1.,-0.001953125,-0.,0.,.001953125,1.,448.],dtype='f2'),size=shape)
            mixed.append(torch.from_numpy(a).to('xpu'))
        cases.append(('mixed_fp8_extremes',mixed,False))
        cases.append(('contiguous_actual',[t.contiguous() for t in reference],False))
        strided=[]
        for t in reference:
            backing=torch.empty((t.shape[0],64,64),dtype=t.dtype,device=t.device);backing[:,:,::2]=t
            strided.append(backing[:,:,::2])
        cases.append(('strided_channels',strided,False))
        for name,values,timed in cases:
            q,k,v=values;expected,debug=previous(q,k,v,debug=True);expected_bytes=raw(expected)
            assert raw(previous(q,k,v))==expected_bytes
            row=dict(name=name,timed=timed,operands={n:arrays.save(t.cpu().numpy()) for n,t in zip(('q','k','v'),values)},strides=[list(t.stride()) for t in values],expected=arrays.save(expected.cpu().numpy()),boundaries={n:arrays.save(t.cpu().numpy()) for n,t in debug.items()},candidates={},samples_ms={},orders=[])
            report['cases'].append(row);functions={'previous':lambda:previous(q,k,v),'fused_compensated':lambda:compensated(q,k,v,bm=32,warps=4,stages=1)[0]}
            for index,config in enumerate(configs):
                output,compiled,bounds=forward(q,k,v,debug=True,**config)
                matches={n:raw(bounds[n])==raw(t) for n,t in debug.items()}
                plain,kernel,_=forward(q,k,v,**config)
                equal=raw(output)==expected_bytes and raw(plain)==expected_bytes and all(matches.values())
                label=str(index);row['candidates'][label]=dict(config=config,byte_equal=equal,boundaries=matches,debug_and_production_match=raw(output)==raw(plain),llir=artifacts.text(str(kernel.asm['llir']),'llir'),spills=kernel.n_spills)
                assert 'ttig.dpas' in str(kernel.asm['ttgir'])
                if equal:functions[label]=lambda config=config:forward(q,k,v,**config)[0]
                else:row['candidates'][label]['actual']=arrays.save(output.cpu().numpy())
            if timed:
                entries={}
                for label,fn in functions.items():
                    stream=torch.xpu.Stream()
                    with stream:
                        for _ in range(2):warm=fn()
                    torch.xpu.synchronize();out=torch.empty_like(warm);del warm;g=torch.xpu.XPUGraph()
                    with torch.xpu.graph(g,stream=stream):
                        for _ in range(4):temporary=fn()
                        out.copy_(temporary)
                    del temporary;graphs.append(g);entries[label]=(g,out);row['samples_ms'][label]=[]
                    assert not any(s['address']<=t.data_ptr()<s['address']+s['total_size'] for s in torch.xpu.memory_snapshot(g.pool()) for t in [*values,out])
                    g.replay();torch.xpu.synchronize();assert raw(out)==expected_bytes
                labels=list(entries)
                for repeat in range(5):
                    offset=repeat%len(labels);order=labels[offset:]+labels[:offset]
                    if repeat%2:order.reverse()
                    row['orders'].append(order)
                    for label in order:
                        g,out=entries[label];torch.xpu.synchronize();started=time.perf_counter()
                        for _ in range(10):g.replay()
                        torch.xpu.synchronize();row['samples_ms'][label].append((time.perf_counter()-started)*1000/40)
                        assert raw(out)==expected_bytes
                row['median_ms']={label:statistics.median(ts) for label,ts in row['samples_ms'].items()}
                row.update(all_graph_outputs_match=True,persistent_io_outside_pool=True)
                for g in graphs:g.reset()
                graphs.clear()
            assert all(raw(t)==arrays.load(row['operands'][n]).tobytes() for n,t in zip(('q','k','v'),values));row['operands_unchanged']=True
            save();print(json.dumps(dict(name=name,matched={n:c['byte_equal'] for n,c in row['candidates'].items()},median_ms=row.get('median_ms'))),flush=True)
        report['all_candidates_byte_equal']=all(c['byte_equal'] for r in report['cases'] for c in r['candidates'].values())
        timed=[r for r in report['cases'] if r['timed']]
        report['sum_median_ms']={label:sum(r['median_ms'][label] for r in timed) for label in ['previous','fused_compensated','0','1','2'] if all(label in r['median_ms'] for r in timed)}
        report['passed']=report['all_candidates_byte_equal']
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for g in graphs:g.reset()
    try:assert all(sha(p)==h for p,h in sources.items());authenticate_main()
    except BaseException as error:report.update(passed=False,finalization_error=repr(error));save();raise
    save()
assert report['passed'],'ViT attention fusion does not meet all boundary comparisons'
print(json.dumps(dict(passed=True,sum_median_ms=report['sum_median_ms'])),flush=True)
