"""Check that XPU graph capture replays mixed Triton/PyTorch work on NEW inputs."""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=DREF/'experimental/xpu-graph-probe-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1');sys.path.insert(0,str(ROOT/'backend'))
import numpy as np
import torch
from fused_activation_int8_v1 import dot
from fast_matrices_v3 import quantize
from nr_backend.triton_fp8 import quantize_fp8
import immutable_artifacts_v1 as artifacts
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sources={str(p):sha(p) for p in [Path(__file__),HERE/'Run-XpuGraphProbeV1.cmd',HERE/'fused_activation_int8_v1.py',HERE/'fast_matrices_v3.py',HERE/'immutable_artifacts_v1.py',ROOT/'backend/nr_backend/triton_fp8.py',Path(torch.xpu.__file__).with_name('graphs.py')]}
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:queue=json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
OUT.mkdir()
report=dict(scope=__doc__,sources=sources,passed=False,cases=[],complete_model_captured=False,torch_version=torch.__version__,device=torch.xpu.get_device_name(),timing_scope='Primitive only, paired prewarmed 60-call batches. Does not prove complete model speed, seed/history semantics or production graph safety.')
graph=None
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
try:
    torch.set_num_threads(2)
    with torch.inference_mode():
        rng=np.random.default_rng(9092603)
        host=rng.uniform(-1,1,(448,64)).astype('f2');weight=rng.uniform(-.1,.1,(64,256)).astype('f2')
        a=torch.from_numpy(host).to('xpu');w=torch.from_numpy(weight).to('xpu');qw,sw,_=quantize(w,columns=True)
        def compute():
            product,_=dot(a,w,packed=(qw,sw),bn=64)
            return (quantize_fp8(product)+.125).half()
        torch.xpu.synchronize();stream=torch.xpu.Stream()
        with stream:
            for _ in range(3):compute()
        torch.xpu.synchronize();graph=torch.xpu.XPUGraph()
        with torch.xpu.graph(graph,stream=stream):captured=compute()
        for i in range(5):
            host=rng.uniform(-1,1,(448,64)).astype('f2') if i else np.zeros((448,64),dtype='f2')
            a.copy_(torch.from_numpy(host));torch.xpu.synchronize()
            expected=compute().cpu().numpy();graph.replay();actual=captured.cpu().numpy()
            equal=actual.tobytes()==expected.tobytes()
            report['cases'].append(dict(input=artifacts.array(host),expected=artifacts.array(expected),actual=artifacts.array(actual),byte_equal=equal));save()
            assert equal,i
        assert len({r['actual']['raw_sha256'] for r in report['cases']})==5
        for _ in range(10):graph.replay();compute()
        samples={'ordinary':[],'graph':[]}
        for i in range(6):
            for name in (('ordinary','graph') if i%2==0 else ('graph','ordinary')):
                torch.xpu.synchronize();started=time.perf_counter()
                for _ in range(60):
                    if name=='graph':graph.replay()
                    else:compute()
                torch.xpu.synchronize();samples[name].append((time.perf_counter()-started)/60)
        medians={k:statistics.median(v) for k,v in samples.items()}
        report.update(passed=True,changed_inputs_replayed=True,samples_seconds=samples,median_seconds=medians,primitive_speedup=medians['ordinary']/medians['graph'])
        graph.reset();graph=None
except Exception as error:
    report['error']=repr(error);report['traceback']=traceback.format_exc();raise
finally:
    assert all(sha(p)==h for p,h in sources.items());save()
print(json.dumps({k:report[k] for k in ('passed','median_seconds','primitive_speedup')}),flush=True)
