"""Streaming two-matrix fusion versus original whole MLP on captured real operands.

First full geometry for each C64/C128/C256 plus a 19-row tail. Five alternating
graph timing rounds; each graph contains four calls and is replayed twelve times.
Primitive speed only. Different-byte candidates are retained but never selected.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=DREF/'experimental/fused-branched-pairs-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1');sys.path.insert(0,str(ROOT/'backend'))
import numpy as np
import torch
from nr_backend.multihead_block import BranchedMLP
from nr_backend.execution import use_arithmetic_backend
from strided_batched_v2 import StridedMatrices
from fused_branched_pairs_v1 import forward as fused
import compressed_arrays_v1 as arrays
import immutable_artifacts_v1 as artifacts
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
capture_path=DREF/'experimental/branched-mlp-operands-v2/validation.json';capture=js(capture_path)
assert capture['passed'] and capture['full_output_byte_equal_prior']
assert js(capture_path.parent.with_suffix('.log.lease.json'))['returncode']==0
assert all(sha(p)==h for p,h in capture['sources'].items())
names=['benchmark_branched_pairs_v1.py','Run-BranchedPairsV1.cmd','fused_mlp_pair_v1.py','fused_branched_pairs_v1.py','fused_branched_mlp_v1.py','strided_batched_v2.py','strided_batched_v1.py','fused_cached_matrices_v2.py','fused_cached_matrices_v1.py','fused_activation_int8_v1.py','static_weight_cache_v2.py','static_weight_cache_v1.py','fast_matrices_v3.py','compressed_arrays_v1.py','immutable_artifacts_v1.py']
paths=[*(HERE/n for n in names),capture_path,*sorted((ROOT/'backend/nr_backend').glob('*.py'))]
frozen={str(p):sha(p) for p in paths}
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:queue=json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
OUT.mkdir();report=dict(scope=__doc__,sources=frozen,passed=False,cases=[],complete_migration=False,configuration='FP16 matrices, FP32 K32 sums, exact old half/FP8 boundaries, BM16, warps4/8 and pipeline stages1/2')
raw=lambda t:t.cpu().numpy().tobytes()
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
try:
    torch.set_num_threads(2)
    provider=StridedMatrices();provider.select('fp16_xmx')
    with torch.inference_mode(),provider.installed(),use_arithmetic_backend('triton'):
        for c in (64,128,256):
            source=next(row for row in capture['cases'].values() if row['channels']==c and row['captured'])
            saved={name:arrays.load(meta) for name,meta in source['arrays'].items()}
            weights={name:torch.from_numpy(saved[name]).to('xpu') for name in ('expand','reduce','project','skip_scale')}
            module=SimpleNamespace(channels=c,**weights)
            for tail in (False,True):
                x=saved['features'].reshape(-1,c);target=saved['output'].reshape(-1,c)
                if tail:x=x[:19];target=target[:19]
                features=torch.from_numpy(x.copy()).to('xpu');expected=target.tobytes()
                operands={name:source['arrays'][name] for name in weights}
                operands['features']=arrays.save(x)
                row=dict(channels=c,rows=len(x),tail=tail,operands=operands,expected=arrays.save(target),candidates=[],samples_seconds={})
                report['cases'].append(row)
                original=lambda:BranchedMLP.forward_unquantized(module,features)
                assert raw(original())==expected
                functions={'original':original}
                for bm,warps,stages in ((16,4,1),(16,8,1),(16,4,2)):
                    name=f'pair_m{bm}_w{warps}_s{stages}';print(f'Compile {name} C{c} rows{len(x)}',flush=True)
                    value,kernel=fused(features,**weights,bm=bm,warps=warps,stages=stages)
                    actual=value.cpu().numpy();equal=actual.tobytes()==expected
                    item=dict(name=name,bm=bm,warps=warps,stages=stages,byte_equal=equal,max_abs=float(np.max(np.abs(actual.astype('f4')-target.astype('f4')))),
                              changed_half_components=int(np.count_nonzero(actual.view('u2')!=target.view('u2'))),
                              ttgir=artifacts.text(str(kernel.asm['ttgir']),'ttgir'),registers=getattr(kernel,'n_regs',None),spills=getattr(kernel,'n_spills',None))
                    row['candidates'].append(item)
                    assert 'ttig.dpas' in str(kernel.asm['ttgir'])
                    if equal:functions[name]=lambda bm=bm,warps=warps,stages=stages:fused(features,**weights,bm=bm,warps=warps,stages=stages)[0]
                    else:item['actual']=arrays.save(actual)
                    save()
                entries={}
                for name,fn in functions.items():
                    stream=torch.xpu.Stream()
                    with stream:
                        for _ in range(3):fn()
                    torch.xpu.synchronize();graph=torch.xpu.XPUGraph()
                    with torch.xpu.graph(graph,stream=stream):
                        for _ in range(4):output=fn()
                    graph.replay();torch.xpu.synchronize();assert raw(output)==expected
                    entries[name]=(graph,output);row['samples_seconds'][name]=[]
                names=list(entries);row['orders']=[]
                for repetition in range(5):
                    offset=repetition%len(names);order=names[offset:]+names[:offset]
                    if repetition%2:order.reverse()
                    row['orders'].append(order)
                    for name in order:
                        graph,output=entries[name];torch.xpu.synchronize();start=time.perf_counter()
                        for _ in range(12):graph.replay()
                        torch.xpu.synchronize();row['samples_seconds'][name].append((time.perf_counter()-start)/48)
                        assert raw(output)==expected
                for name,value in weights.items():assert raw(value)==saved[name].tobytes()
                assert raw(features)==x.tobytes()
                medians={name:statistics.median(values) for name,values in row['samples_seconds'].items()}
                row.update(median_seconds=medians,speedup={name:medians['original']/value for name,value in medians.items()},operands_unchanged=True)
                for graph,output in entries.values():assert raw(output)==expected;graph.reset()
                entries.clear();save();print(json.dumps(dict(c=c,rows=len(x),median_seconds=medians,speedup=row['speedup'])),flush=True)
        report['passed']=True
except Exception as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    assert all(sha(p)==h for p,h in frozen.items());save()
print(json.dumps(dict(passed=report['passed'],cases=len(report['cases']))),flush=True)
