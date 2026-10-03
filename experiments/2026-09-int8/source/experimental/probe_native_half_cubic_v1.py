"""Exhaustive unary half cubic equivalence under isolated Triton3.8.

Compare native half FMA and the original compensated GPU helper against the
previously certified 65536-entry table. Preserve complete output receipts;
microtimings are not an entire model performance claim.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference'); OUT=D/'experimental/native-half-cubic-v1'
assert not OUT.exists()
TOOLCHAIN=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
sys.path[:0]=[str(TOOLCHAIN/'site'),str(EXACT/'reference'),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main()
provision=TOOLCHAIN/'provision-v1.json'
assert sha(provision)=='e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN/'site'/p)==h for p,h in js(provision)['files'].items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
import torch,triton
import triton.language as tl
from fused_branched_mlp_v1 import _cubic as previous
from native_half_cubic_v1 import cubic as candidate
from cubic_lut_constant_v1 import TABLE,TABLE_SHA
import compressed_arrays_v1 as arrays
import immutable_artifacts_v1 as artifacts
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site')
assert sha(TABLE)==TABLE_SHA
certificates=[]
for name,pin in [('cubic-fp8-fused-cpu-v3.json','e392b661add80f01b0db4de802bc0a50ed5ea05b098d48354c2a13e611654f63'),
                 ('cubic-fp8-fused-xpu-v3.json','c79a820724e0d7e467b8cd963971990fa163971da4490bc497ecdac4aca4ae1e')]:
    p=D/name;assert sha(p)==pin and js(p)['all_byte_equal'] and js(p)['table_file_sha256']==TABLE_SHA
    certificates.append(p)
table=np.load(TABLE,allow_pickle=False);assert table.shape==(65536,) and table.dtype==np.dtype('<f2')
paths=[Path(__file__),HERE/'Run-NativeHalfCubicV1.cmd',HERE/'native_half_cubic_v1.py',HERE/'fused_branched_mlp_v1.py',
       HERE/'cubic_lut_constant_v1.py',HERE/'compressed_arrays_v1.py',HERE/'immutable_artifacts_v1.py',
       ROOT/'backend/nr_backend/triton_cubic_fp8.py',ROOT/'backend/nr_backend/triton_fp8.py',provision,TABLE,*certificates]
sources={str(p):sha(p) for p in paths}

@triton.jit
def kernel(X,LUT,Y,N:tl.constexpr,B:tl.constexpr,METHOD:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B)
    x=tl.load(X+i,i<N,other=0)
    if METHOD==0:result=previous(x)
    elif METHOD==1:result=candidate(x)
    else:
        index=x.to(tl.uint16,bitcast=True).to(tl.int32)
        result=tl.load(LUT+index).to(tl.uint16).to(tl.float16,bitcast=True)
    tl.store(Y+i,result,i<N)

OUT.mkdir();report=dict(scope=__doc__,passed=False,complete_migration=False,sources=sources,exact_gate=gate,cases=[],timings=[])
report['runtime']=dict(torch=torch.__version__,triton=triton.__version__,triton_file=triton.__file__,isolated=True)
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
def raw(t):return t.cpu().numpy().tobytes()
graphs=[]
try:
    torch.set_num_threads(2)
    rng=np.random.Generator(np.random.PCG64(0x580C0B1C))
    ordered=np.arange(65536,dtype='u2')
    shuffled=np.concatenate([rng.permutation(ordered),np.array([0x7c01,0x8000,0xfc00],dtype='u2')])
    with torch.inference_mode():
        lut=torch.from_numpy(table.view('i2').copy()).to('xpu')
        for name,bits in [('all_half_ordered',ordered),('all_half_shuffled_tail',shuffled)]:
            x=torch.from_numpy(bits.view('f2').copy()).to('xpu'); expected=table[bits]
            for block in (256,512,1024):
                row=dict(name=name,elements=len(bits),block=block,input=arrays.save(bits.view('f2')),expected=arrays.save(expected),methods=[])
                report['cases'].append(row)
                for method in (0,1,2):
                    y=torch.empty_like(x)
                    compiled=kernel[(triton.cdiv(x.numel(),block),)](x,lut,y,x.numel(),block,method,num_warps=4,enable_fp_fusion=False)
                    output=y.cpu().numpy();different=np.flatnonzero(output.view('u2')!=expected.view('u2'))
                    row['methods'].append(dict(method=method,byte_equal=not len(different),differences=len(different),
                        first_differences=[dict(index=int(i),input_bits=int(bits[i]),expected_bits=int(expected.view('u2')[i]),actual_bits=int(output.view('u2')[i])) for i in different[:16]],
                        output=arrays.save(output),ir={k:artifacts.text(v,k) for k,v in compiled.asm.items() if k in ('ttir','ttgir','llir') and isinstance(v,str)}))
                assert raw(x)==bits.tobytes() and raw(lut)==table.tobytes()
                save();print(json.dumps(dict(name=name,block=block,results=[{k:v for k,v in m.items() if k in ('method','byte_equal','differences','first_differences')} for m in row['methods']])),flush=True)
        report['all_complete_outputs_equal']=all(m['byte_equal'] for c in report['cases'] for m in c['methods'])
        if report['all_complete_outputs_equal']:
            for n in (65536,524288):
                bits=np.tile(ordered,n//65536);x=torch.from_numpy(bits.view('f2').copy()).to('xpu');expected=table[bits].tobytes()
                row=dict(elements=n,block=512,orders=[],samples_ms={str(m):[] for m in range(3)})
                report['timings'].append(row);entries={}
                for method in range(3):
                    out=torch.empty_like(x);stream=torch.xpu.Stream()
                    with stream:
                        for _ in range(2):kernel[(triton.cdiv(n,512),)](x,lut,out,n,512,method,num_warps=4,enable_fp_fusion=False)
                    torch.xpu.synchronize();g=torch.xpu.XPUGraph()
                    with torch.xpu.graph(g,stream=stream):
                        for _ in range(4):kernel[(triton.cdiv(n,512),)](x,lut,out,n,512,method,num_warps=4,enable_fp_fusion=False)
                    graphs.append(g);entries[method]=(g,out)
                    assert not any(s['address']<=t.data_ptr()<s['address']+s['total_size'] for s in torch.xpu.memory_snapshot(g.pool()) for t in (x,lut,out))
                    g.replay();torch.xpu.synchronize();assert raw(out)==expected
                for repeat in range(5):
                    order=[(repeat+i)%3 for i in range(3)]
                    if repeat%2:order.reverse()
                    row['orders'].append(order)
                    for method in order:
                        g,out=entries[method];torch.xpu.synchronize();start=time.perf_counter()
                        for _ in range(20):g.replay()
                        torch.xpu.synchronize();row['samples_ms'][str(method)].append((time.perf_counter()-start)*1000/80)
                        assert raw(out)==expected
                row['median_ms']={m:statistics.median(v) for m,v in row['samples_ms'].items()}
                row.update(all_graph_outputs_equal=True,persistent_io_outside_pool=True)
                assert raw(x)==bits.tobytes() and raw(lut)==table.tobytes()
                for g in graphs:g.reset()
                graphs.clear();save();print(json.dumps(row),flush=True)
        report['passed']=report['all_complete_outputs_equal']
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for g in graphs:g.reset()
    try:
        assert all(sha(p)==h for p,h in sources.items());authenticate_main()
    except BaseException as error:
        report.update(passed=False,finalization_error=repr(error));save();raise
    save()
assert report['passed'], 'Native unary cubic did not meet complete-bit equivalence'
