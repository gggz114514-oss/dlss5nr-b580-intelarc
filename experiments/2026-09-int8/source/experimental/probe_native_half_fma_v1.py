"""Probe Intel half FMA lowering and exact bits in attention arithmetic domains.

Two native spellings are compared with the current compensated helper. Enumerate
all half score encodings for both exp transforms, all half x against selected y
for normalization x*x+half(y*y), plus seeded general half triples. Differences
are evidence, not silently accepted model changes. No model adapter is changed.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/native-half-fma-v1'
assert not OUT.exists();os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-sm89-v1')
sys.path[:0]=[str(EXACT/'reference'),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main()
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
import torch,triton
import triton.language as tl
from triton.language.extra.intel import libdevice
from nr_backend.triton_cubic_fp8 import _half_fma_value
from nr_backend.triton_attention_normalize import _nan_left
import compressed_arrays_v1 as arrays
import immutable_artifacts_v1 as artifacts

@triton.jit
def fma(a,b,c,METHOD:tl.constexpr):
    if METHOD==0:return _half_fma_value(a,b,c)
    elif METHOD==1:return tl.fma(a.to(tl.float16),b.to(tl.float16),c.to(tl.float16)).to(tl.float16)
    else:return libdevice.fma(a.to(tl.float16),b.to(tl.float16),c.to(tl.float16))

@triton.jit
def kernel(X,Y,Z,OUT,N:tl.constexpr,DOMAIN:tl.constexpr,METHOD:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<N
    if DOMAIN==0:
        a=tl.load(X+i,valid,other=0);b=tl.load(Y+i,valid,other=0);c=tl.load(Z+i,valid,other=0)
        result=fma(a,b,c,METHOD)
    elif DOMAIN==1:
        x=(i%65536).to(tl.uint16).to(tl.float16,bitcast=True)
        y=tl.load(Y+i//65536,valid,other=0)
        a=(y.to(tl.float32)*y.to(tl.float32)).to(tl.float16)
        result=_nan_left(fma(x,x,a,METHOD),x,a)
    else:
        x=(i%65536).to(tl.uint16).to(tl.float16,bitcast=True)
        if DOMAIN==2:
            result=fma(x,tl.full((),.044921875,tl.float32),tl.full((),1.30078125,tl.float32),METHOD)
            result=tl.minimum(tl.maximum(result.to(tl.float32),1.03125),1.5693359375).to(tl.float16)
            shift:tl.constexpr=5;offset:tl.constexpr=0x8000
        else:
            result=fma(x,tl.full((),.08953857421875,tl.float32),tl.full((),1.708984375,tl.float32),METHOD)
            result=tl.minimum(tl.maximum(result.to(tl.float32),1.439453125),1.9775390625).to(tl.float16)
            shift:tl.constexpr=4;offset:tl.constexpr=0x4000
        bits=result.to(tl.uint16,bitcast=True).to(tl.int32)
        inp=x.to(tl.uint16,bitcast=True).to(tl.int32)
        bits=tl.where((inp&0x7fff)>0x7c00,inp|0x200,bits)
        result=(((bits<<shift)+offset)&0xffff).to(tl.uint16).to(tl.float16,bitcast=True)
    tl.store(OUT+i,result,valid)

sources={str(p):sha(p) for p in [Path(__file__),HERE/'Run-NativeHalfFmaV1.cmd',
    ROOT/'backend/nr_backend/triton_cubic_fp8.py',ROOT/'backend/nr_backend/triton_attention_normalize.py',
    Path(libdevice.__file__),HERE/'compressed_arrays_v1.py',HERE/'immutable_artifacts_v1.py']}
OUT.mkdir();report=dict(scope=__doc__,passed=False,sources=sources,exact_gate=gate,cases=[],complete_migration=False)
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
graphs=[]
try:
    torch.set_num_threads(2)
    ys=np.unique(np.array([0,1,2,3,0x100,0x200,0x3fe,0x3ff,0x400,0x401,0x7ff,0x800,
        *range(0x1000,0x7c00,0x400),0x3bff,0x3c01,0x53ff,0x5401,0x5bff,0x5c00,0x5c01,
        0x7bfe,0x7bff,0x7c00,0x7c01,0x7dff,0x7e00,0x7fff,0x8000,0xfc00,0xfe00],dtype='u2')).view('f2')
    rng=np.random.Generator(np.random.PCG64(0x5804060))
    xyz=[rng.integers(0,65536,size=1048576,dtype=np.uint16).view('f2') for _ in range(3)]
    report['seed']=0x5804060;report['normalization_y_bits']=ys.view('u2').tolist()
    with torch.inference_mode():
        for name,domain,n,operands in [('raw_random',0,len(xyz[0]),xyz),('norm_square',1,65536*len(ys),[ys,ys,ys]),
                                     ('swin_exp_all_half',2,65536,[ys,ys,ys]),('vit_exp_all_half',3,65536,[ys,ys,ys])]:
            inputs=[torch.from_numpy(v).to('xpu') for v in operands]
            outputs=[torch.empty(n,dtype=torch.float16,device='xpu') for _ in range(3)]
            row=dict(name=name,domain=domain,elements=n,input_arrays=[arrays.save(v) for v in operands],candidates=[])
            report['cases'].append(row);reference=None
            active=[]
            for method in range(3):
                entry=dict(method=method);row['candidates'].append(entry)
                try:
                    compiled=kernel[(triton.cdiv(n,256),)](*inputs,outputs[method],n,domain,method,256,num_warps=4,enable_fp_fusion=False)
                    data=outputs[method].cpu().numpy();bits=data.view('u2')
                    if method==0:reference=data.copy()
                    different=np.flatnonzero(bits!=reference.view('u2'))
                    nan_equal=np.isnan(data)&np.isnan(reference)
                    finite_difference=(bits!=reference.view('u2'))&~nan_equal
                    entry.update(compiled=True,byte_equal=not len(different),different_values=int(len(different)),
                        differences_excluding_both_nan=int(finite_difference.sum()),
                        first_differences=[dict(index=int(i),expected_bits=int(reference.view('u2')[i]),actual_bits=int(bits[i])) for i in different[:12]],
                        output=arrays.save(data),spills=compiled.n_spills,
                        ir={k:artifacts.text(v,k) for k,v in compiled.asm.items() if isinstance(v,str) and k in ('ttir','ttgir','llir')})
                    active.append(method)
                except Exception as error:
                    entry.update(compiled=False,error=repr(error),traceback=traceback.format_exc())
                    if method==0:raise
            for method in active:
                stream=torch.xpu.Stream();g=torch.xpu.XPUGraph()
                with torch.xpu.graph(g,stream=stream):
                    kernel[(triton.cdiv(n,256),)](*inputs,outputs[method],n,domain,method,256,num_warps=4,enable_fp_fusion=False)
                graphs.append(g)
            timings={method:[] for method in active};orders=[]
            for repetition in range(5):
                order=active[repetition%len(active):]+active[:repetition%len(active)];orders.append(order)
                for method in order:
                    g=graphs[active.index(method)];torch.xpu.synchronize();started=time.perf_counter()
                    for _ in range(20):g.replay()
                    torch.xpu.synchronize();timings[method].append((time.perf_counter()-started)/20)
                    assert outputs[method].cpu().numpy().tobytes()==arrays.load(row['candidates'][method]['output']).tobytes()
            for method in active:
                row['candidates'][method].update(samples_seconds=timings[method],median_seconds=statistics.median(timings[method]))
            for g in graphs:g.reset()
            graphs.clear();row['timing_orders']=orders
            assert all(t.cpu().numpy().tobytes()==a.tobytes() for t,a in zip(inputs,operands))
            row['operands_unchanged']=True;save()
            print(json.dumps(dict(name=name,elements=n,results=[{k:v for k,v in e.items() if k in ('method','compiled','different_values','differences_excluding_both_nan','median_seconds','error')} for e in row['candidates']])),flush=True)
    report['passed']=True
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for g in graphs:g.reset()
    assert all(sha(p)==h for p,h in sources.items());authenticate_main();save()
print(json.dumps(dict(probe_completed=report['passed'],all_model_changes_pending=True)),flush=True)
