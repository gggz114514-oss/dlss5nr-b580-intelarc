"""Test remaining frequent full-model shapes using synthetic finite half operands."""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=DREF/'experimental/fused-shapes-v2'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1')
sys.path.insert(0,str(ROOT/'backend'))
import numpy as np
import torch
from fast_matrices_v3 import dot as baseline,quantize
from fused_activation_int8_v1 import dot as fused
import immutable_artifacts_v1 as artifacts
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
full_path=DREF/'results/fused-full-480-v1/validation.json'
assert sha(full_path)=='bfe82987b75ca2e76df0d7026443521398ca0e53c7f79722db3a923273274998'
full=json.loads(full_path.read_text());assert full['passed']
sources={str(p):sha(p) for p in [Path(__file__),HERE/'Run-FusedShapesV2.cmd',HERE/'fast_matrices_v3.py',HERE/'fused_activation_int8_v1.py',HERE/'immutable_artifacts_v1.py',full_path]}
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:queue=json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
OUT.mkdir()
report=dict(scope=__doc__,sources=sources,passed=False,cases=[],operand_seed=9092602,timing_scope='Six cyclic/reversed rounds of60 prewarmed calls. Includes wrapper/allocation, activation quantizer and matrix dot plus synchronization; excludes weight packing, upload, JIT, validation and IO. Primitive results only.',torch_version=torch.__version__,device=torch.xpu.get_device_name())
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
raw=lambda x:x.cpu().numpy().tobytes()
rng=np.random.default_rng(report['operand_seed'])
try:
    torch.set_num_threads(2)
    with torch.inference_mode():
        for m,k,n,initialized in [(448,64,256,False),(448,256,64,False),(1792,32,256,True),(1792,256,128,False),(2560,256,128,False),(128,256,512,True),(2048,256,256,True),(28672,64,192,False)]:
            a=rng.uniform(-1,1,(m,k)).astype('f2');w=rng.uniform(-.125,.125,(k,n)).astype('f2')
            a[0]=0;w[:,0]=0
            initial=rng.uniform(-.2,.2,(m,n)).astype('f2') if initialized else None
            aa=torch.from_numpy(a).to('xpu');ww=torch.from_numpy(w).to('xpu');ini=None if initial is None else torch.from_numpy(initial).to('xpu')
            qw,sw,_=quantize(ww,columns=True);packed=(qw,sw)
            target,_=baseline(aa,ww,initial=ini,int8=True,packed=packed);expected=raw(target)
            row=dict(shape=[m,k,n],initialized=initialized,expected=artifacts.array(target.cpu().numpy()),modes=[],operands=dict(a=artifacts.array(a),weight=artifacts.array(w),initial=None if initial is None else artifacts.array(initial)))
            report['cases'].append(row)
            funcs={'cached':lambda:baseline(aa,ww,initial=ini,int8=True,packed=packed)}
            for bn in (32,64,128):
                mode=f'fused_n{bn}';print(f'{m}x{k}x{n}: {mode}',flush=True)
                funcs[mode]=lambda bn=bn:fused(aa,ww,initial=ini,packed=packed,bn=bn)
                value,kernel=funcs[mode]()
                equal=raw(value)==expected
                row['modes'].append(dict(mode=mode,byte_equal=equal,actual=artifacts.array(value.cpu().numpy()),ttgir=artifacts.text(str(kernel.asm['ttgir']),'ttgir'),llir=artifacts.text(str(kernel.asm['llir']),'llir')))
                save();assert equal
                assert 'ttig.dpas' in str(kernel.asm['ttgir']) and 'SubgroupMatrixMultiplyAccumulateINTEL' in str(kernel.asm['llir'])
            for fn in funcs.values():
                for _ in range(10):fn()
            samples={mode:[] for mode in funcs};orders=[];modes=list(funcs)
            for repetition in range(6):
                offset=repetition%4;order=modes[offset:]+modes[:offset]
                if repetition%2:order.reverse()
                orders.append(order)
                for mode in order:
                    fn=funcs[mode];torch.xpu.synchronize();started=time.perf_counter()
                    for _ in range(60):value,_=fn()
                    torch.xpu.synchronize();samples[mode].append((time.perf_counter()-started)/60)
                    assert raw(value)==expected
            medians={mode:statistics.median(times) for mode,times in samples.items()}
            row.update(samples_seconds=samples,timing_orders=orders,median_seconds=medians,speedup_vs_cached={mode:medians['cached']/seconds for mode,seconds in medians.items()})
            assert raw(aa)==a.tobytes() and raw(ww)==w.tobytes()
            if ini is not None:assert raw(ini)==initial.tobytes()
            row['operands_unchanged']=True;save()
            print(json.dumps(dict(shape=row['shape'],seconds=medians,speedup=row['speedup_vs_cached'])),flush=True)
    report['passed']=True
except Exception as error:
    report['error']=repr(error);report['traceback']=traceback.format_exc();raise
finally:
    assert all(sha(p)==h for p,h in sources.items());save()
print(json.dumps(dict(passed=report['passed'],cases=len(report['cases']))),flush=True)
