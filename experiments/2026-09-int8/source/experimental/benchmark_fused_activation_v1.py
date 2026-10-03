"""Byte checks and bounded benchmarks against cached W8A8, including its quantizer."""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=DREF/'experimental/fused-activation-int8-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1')
sys.path.insert(0,str(ROOT/'backend'))
import numpy as np
import torch
from fast_matrices_v3 import dot as baseline,quantize
from fused_activation_int8_v1 import dot as fused
import immutable_artifacts_v1 as artifacts
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources={str(p):sha(p) for p in [Path(__file__),HERE/'Run-FusedActivationV1.cmd',HERE/'fused_activation_int8_v1.py',HERE/'fast_matrices_v3.py',HERE/'immutable_artifacts_v1.py']}
prior_path=DREF/'experimental/f16-xmx-v3/validation.json'
manifest=json.loads((DREF/'experimental/f16-xmx-ablation-v1/source-manifest.json').read_text())
assert sha(prior_path)==manifest['prior_report_sha256']
prior=json.loads(prior_path.read_text())
capture=next(c for c in prior['cases'] if c['name']=='actual-2-8x16w1')
sources[str(prior_path)]=sha(prior_path)
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:queue=json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
OUT.mkdir()
report=dict(scope=__doc__,sources=sources,passed=False,cases=[],timing_scope='Six cyclic/reversed rounds of40 prewarmed calls; includes allocation, activation quantization, dispatch and synchronization. Excludes weight packing, uploads, JIT, checks and IO. Same packed weights in all modes. Median per-call wall time; primitive results do not establish model speed.',native_byte_alignment_claim=False)
configs={'fused_n32':32,'fused_n64':64,'fused_n128':128}
rng=np.random.default_rng(9092601)
raw=lambda x:x.cpu().numpy().tobytes()
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
def run_case(name,a,w,initial=None,bench=False,noncontiguous=False,provenance=None):
    before=[a.copy(),w.copy(),None if initial is None else initial.copy()]
    aa=torch.from_numpy(a).to('xpu');ww=torch.from_numpy(w).to('xpu')
    ini=None if initial is None else torch.from_numpy(initial).to('xpu')
    if noncontiguous:
        aa=aa.transpose(-1,-2).contiguous().transpose(-1,-2)
        ww=ww.T.contiguous().T
        if ini is not None:ini=ini.transpose(-1,-2).contiguous().transpose(-1,-2)
    qw,sw,_=quantize(ww,columns=True);packed=(qw,sw)
    expected,_=baseline(aa,ww,initial=ini,int8=True,packed=packed)
    target=raw(expected)
    row=dict(name=name,a_shape=list(a.shape),w_shape=list(w.shape),initial=ini is not None,noncontiguous=noncontiguous,provenance=provenance,expected=artifacts.array(expected.cpu().numpy()),modes=[])
    report['cases'].append(row)
    funcs={'cached':lambda:baseline(aa,ww,initial=ini,int8=True,packed=packed)}
    for mode,bn in configs.items():
        print(f'{name}: compile and compare {mode}',flush=True)
        funcs[mode]=lambda bn=bn:fused(aa,ww,initial=ini,packed=packed,bn=bn)
        out,kernel=funcs[mode]()
        actual=raw(out)
        same=actual==target
        details=dict(mode=mode,byte_equal=same,actual=artifacts.array(out.cpu().numpy()),mismatched_values=int(np.count_nonzero(np.frombuffer(actual,'u2')!=np.frombuffer(target,'u2'))),ttgir=artifacts.text(str(kernel.asm['ttgir']),'ttgir'),llir=artifacts.text(str(kernel.asm['llir']),'llir'))
        details['dpas']='ttig.dpas' in str(kernel.asm['ttgir']) and 'SubgroupMatrixMultiplyAccumulateINTEL' in str(kernel.asm['llir'])
        row['modes'].append(details);save()
        assert same and details['dpas'],(name,mode,details['mismatched_values'])
    if bench:
        for fn in funcs.values():
            for _ in range(8):fn()
        torch.xpu.synchronize()
        samples={mode:[] for mode in funcs};orders=[];modes=list(funcs)
        for repetition in range(6):
            offset=repetition%len(modes);order=modes[offset:]+modes[:offset]
            if repetition%2:order.reverse()
            orders.append(order)
            for mode in order:
                fn=funcs[mode]
                torch.xpu.synchronize();started=time.perf_counter()
                for _ in range(40):value,_=fn()
                torch.xpu.synchronize();samples[mode].append((time.perf_counter()-started)/40)
                assert raw(value)==target
        medians={mode:statistics.median(times) for mode,times in samples.items()}
        row.update(samples_seconds=samples,timing_orders=orders,median_seconds=medians,speedup_vs_cached={mode:medians['cached']/seconds for mode,seconds in medians.items()})
        print(json.dumps(dict(name=name,median_seconds=medians,speedup=row['speedup_vs_cached'])),flush=True)
    assert raw(aa)==before[0].tobytes() and raw(ww)==before[1].tobytes()
    if ini is not None:assert raw(ini)==before[2].tobytes()
    row['operands_unchanged']=True;save()

try:
    torch.set_num_threads(2)
    with torch.inference_mode():
        for m,k,n in [(17,16,37),(33,48,65),(35,128,31),(19,256,129)]:
            a=rng.uniform(-1,1,(m,k)).astype('f2');w=rng.uniform(-1,1,(k,n)).astype('f2')
            a[0]=0;a[1]=np.float16(-0.);w[:,0]=0
            initial=rng.uniform(-.2,.2,(m,n)).astype('f2')
            run_case(f'tails-{m}-{k}-{n}',a,w,initial,noncontiguous=True)
        # All finite half magnitudes, tiny scales, signed zeros and mixed leading dims.
        bits=np.arange(65536,dtype='u2');finite=bits[(bits&0x7c00)!=0x7c00].view('f2')
        a=np.resize(finite,33*256).reshape(33,256).copy()
        a[:]=rng.choice(finite,size=a.shape);a[0]=np.float16(2**-24);a[1]=np.float16(-0.)
        w=rng.uniform(-.001,.001,(256,37)).astype('f2')
        run_case('finite-half-range',a,w)
        run_case('leading-dimensions',rng.uniform(-1,1,(2,17,32)).astype('f2'),rng.uniform(-1,1,(32,64)).astype('f2'))
        arrays={}
        for name,meta in capture['capture']['capture_sources'].items():
            assert sha(meta['path'])==meta['sha256']
            arrays[name]=np.load(meta['path'],allow_pickle=False).copy()
        run_case('actual-captured-32768-32-128',arrays['a'],arrays['weight'],arrays.get('initial'),bench=True,provenance=capture['capture'])
        for m,k,n in [(32768,128,32),(32768,32,96),(8192,64,64),(8192,128,128),(2048,256,256)]:
            a=rng.uniform(-1,1,(m,k)).astype('f2');w=rng.uniform(-.1,.1,(k,n)).astype('f2')
            initial=rng.uniform(-.1,.1,(m,n)).astype('f2')
            run_case(f'synthetic-{m}-{k}-{n}',a,w,initial,bench=True)
    report['passed']=True
except Exception as error:
    report['error']=repr(error);report['traceback']=traceback.format_exc();raise
finally:
    assert all(sha(p)==digest for p,digest in sources.items());save()
print(json.dumps(dict(passed=report['passed'],cases=len(report['cases']))),flush=True)
