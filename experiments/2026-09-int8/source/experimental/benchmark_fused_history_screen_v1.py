"""First complete actual-frame screen of the fused history sampler.

Compare both complete float32 numerator and reciprocal tensors to frozen current
pipeline captures, then measure original/fused whole history adapter host calls.
Includes original status readback, output clones and completion. No NR body time.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/fused-history-screen-v1';assert not OUT.exists()
TOOLCHAIN=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
sys.path[:0]=[str(TOOLCHAIN/'site'),str(EXACT/'reference'),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main();prior_path=D/'experimental/k8-outer-profile-v2/validation.json'
assert sha(prior_path)=='aa574c05863f78a4e09439e290c18534de5775d7194314df532b2ca0f426f1e0'
prior=js(prior_path);assert prior['passed'] and js(prior_path.parent.with_suffix('.log.lease.json'))['returncode']==0
sources=dict(prior['sources']);sources[str(prior_path)]=sha(prior_path)
provision_path=TOOLCHAIN/'provision-v1.json'
assert sha(provision_path)=='e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN/'site'/p)==h for p,h in js(provision_path)['files'].items())
for p in (Path(__file__),HERE/'Run-FusedHistoryScreenV1.cmd',HERE/'fused_square_history_v1.py',HERE/'fused_graph_history_warp_v1.py'):sources[str(p)]=sha(p)
assert all(sha(p)==h for p,h in sources.items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
import torch
import triton
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site') and triton.__version__.startswith('3.8.0')
from nr_backend.reciprocal import NativeReciprocalTable
from nr_backend.execution import use_arithmetic_backend
from graph_history_warp_v2 import GraphHistoryWarp
from fused_graph_history_warp_v1 import FusedHistoryWarp
import compressed_arrays_v1 as arrays
OUT.mkdir();report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,cases=[],complete_migration=False)
raw=lambda t:t.cpu().numpy().tobytes();adapters=[]
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
try:
    torch.set_num_threads(2)
    table=NativeReciprocalTable.from_directory(EXACT/'model-assets/reciprocal-sm89-v1').to('xpu')
    model=SimpleNamespace(reciprocal=table)
    adapters=[('previous',GraphHistoryWarp(model)),('b32',FusedHistoryWarp(model,block=32)),('b64',FusedHistoryWarp(model,block=64))]
    case=prior['warp_cases'][0];meta=case['arrays'];report['reference']=case
    image=torch.from_numpy(arrays.load(meta['image'])).to('xpu');motion=torch.from_numpy(arrays.load(meta['motion'])).to('xpu')
    expected=[arrays.load(meta[k]) for k in ('numerator','reciprocal')]
    expected_bytes=[a.tobytes() for a in expected];table_before=raw(table.values)
    def run(adapter):return adapter.apply(image,motion,return_components=True,reciprocal_source=table)
    with torch.inference_mode(),use_arithmetic_backend('triton'):
        for name,adapter in adapters:
            print('Compiling/warming history '+name,flush=True)
            values=run(adapter);actual=[v.cpu().numpy() for v in values];equal=[a.tobytes()==b for a,b in zip(actual,expected_bytes)]
            row=dict(name=name,byte_equal=all(equal),outputs=[dict(byte_equal=ok,raw_sha256=hashlib.sha256(a.tobytes()).hexdigest(),
                different_float_words=int(np.count_nonzero(a.view('u4')!=b.view('u4')))) for a,b,ok in zip(actual,expected,equal)])
            if not all(equal):row['actual']=[arrays.save(a) for a in actual]
            report['cases'].append(row);save();assert all(equal),row
            for _ in range(2):run(adapter)
        samples=[[] for _ in adapters];orders=[]
        for r in range(5):
            order=list(range(len(adapters)));order=order[r%len(order):]+order[:r%len(order)];orders.append(order)
            for i in order:
                torch.xpu.synchronize();started=time.perf_counter()
                for _ in range(10):values=run(adapters[i][1])
                torch.xpu.synchronize();samples[i].append((time.perf_counter()-started)/10)
                assert [raw(v) for v in values]==expected_bytes
            print(json.dumps(dict(round=r,history_ms=[s[-1]*1000 for s in samples])),flush=True)
        assert raw(image)==arrays.load(meta['image']).tobytes() and raw(motion)==arrays.load(meta['motion']).tobytes() and raw(table.values)==table_before
        report.update(passed=True,inputs_and_table_unchanged=True,timing=dict(labels=[n for n,_ in adapters],samples_seconds=samples,
            median_seconds=[statistics.median(s) for s in samples],orders=orders,adapter_calls_per_variant=50,complete_output_readbacks_per_variant=5))
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for _,adapter in adapters:adapter.close()
    assert all(sha(p)==h for p,h in sources.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],timing=report.get('timing'))),flush=True)
