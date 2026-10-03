"""Broaden selected exact K8 layouts beyond the measured temporal frame.

Use the two owned real model weights with small/ragged/strided inputs, signed
zeros and all finite half encodings in a deterministic shuffle. Compare complete
outputs against original K8 math, allowing matching nonfinite output words.
Check fallback for initialized calls, other weights and baseline precision.
These are finite input tests, not a universal NaN/HDR/control equivalence claim.
"""
import hashlib,json,os,sys,traceback,urllib.request
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/k8-tiled-boundaries-v1';assert not OUT.exists()
TOOLCHAIN=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
sys.path[:0]=[str(TOOLCHAIN/'site'),str(EXACT/'reference'),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main();prior_path=D/'experimental/k8-tiling38-v1/validation.json'
assert sha(prior_path)=='b5fef99364b540e8bd286790167ca011f62df9f291a7e468de2c857e6c5b80c3'
prior=js(prior_path);assert prior['passed'] and js(prior_path.parent.with_suffix('.log.lease.json'))['returncode']==0
provision_path=TOOLCHAIN/'provision-v1.json'
assert sha(provision_path)=='e94757afcb5962cacbe208cbb808b3516de210059335f1ee7ab7b40d6bfe2c07'
assert all(sha(TOOLCHAIN/'site'/p)==h for p,h in js(provision_path)['files'].items())
sources=dict(prior['sources']);sources[str(prior_path)]=sha(prior_path)
for p in (Path(__file__),HERE/'Run-K8TiledBoundariesV1.cmd',HERE/'k8_tiled_provider_v1.py'):sources[str(p)]=sha(p)
assert all(sha(p)==h for p,h in sources.items())
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response:q=json.load(response)
assert not q['queue_running'] and not q['queue_pending']
import numpy as np
import torch
import triton
assert Path(triton.__file__).is_relative_to(TOOLCHAIN/'site') and triton.__version__.startswith('3.8.0')
from nr_backend.triton_math import fused_dot
from k8_tiled_provider_v1 import K8TiledMatrices
import compressed_arrays_v1 as arrays
OUT.mkdir();report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,cases=[],complete_migration=False)
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
raw=lambda t:t.cpu().numpy().tobytes()
try:
    torch.set_num_threads(2);rng=np.random.default_rng(871251)
    weights={c['name']:torch.from_numpy(arrays.load(c['arrays']['w'])).to('xpu') for c in prior['cases']}
    model=SimpleNamespace(pre=SimpleNamespace(front_weight=weights['pre']),post=SimpleNamespace(head_weight=weights['post']))
    provider=K8TiledMatrices(model);provider.select('fp16_xmx')
    all_bits=np.arange(65536,dtype='u2');finite_bits=all_bits[(all_bits&0x7c00)!=0x7c00];assert len(finite_bits)==63488
    rng.shuffle(finite_bits)
    with torch.inference_mode():
        for name,w in weights.items():
            k,n=w.shape
            cases=[('rows'+str(m),rng.standard_normal((m,k)).astype('f2')) for m in (1,3,17,129)]
            cases.append(('all_finite_half_bits',finite_bits.view('f2').reshape(-1,k)))
            zeros=np.zeros((19,k),dtype='u2');zeros.flat[1::2]=0x8000;cases.append(('signed_zero',zeros.view('f2')))
            cases.append(('strided',rng.standard_normal((17,k*2)).astype('f2')))
            for label,cpu in cases:
                a=torch.from_numpy(cpu.copy()).to('xpu')
                if label=='strided':a=a[:,::2];assert not a.is_contiguous()
                saved=raw(a);expected=fused_dot(a,w,chunk_k=8)
                before=dict(provider.k8_calls);value=provider.dense(a,w,chunk_k=8)
                actual=value.cpu().numpy();target=expected.cpu().numpy();equal=actual.tobytes()==target.tobytes()
                row=dict(weight=name,name=label,input=arrays.save(a.cpu().numpy()),weight_meta=next(c['arrays']['w'] for c in prior['cases'] if c['name']==name),
                    expected=arrays.save(target),byte_equal=equal,different_half_words=int(np.count_nonzero(actual.view('u2')!=target.view('u2'))),
                    output_raw_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),nonfinite_output_words=int(np.count_nonzero(~np.isfinite(actual))))
                if not equal:row['actual']=arrays.save(actual)
                report['cases'].append(row);save();assert equal,(name,label,row['different_half_words'])
                assert provider.k8_calls[name]==before.get(name,0)+1 and raw(a)==saved
            probe=torch.from_numpy(rng.standard_normal((17,k)).astype('f2')).to('xpu')
            initial=torch.from_numpy(rng.standard_normal((17,n)).astype('f2')).to('xpu')
            for label,mode,weight,ini in [('initialized','fp16_xmx',w,initial),('unowned_weight','fp16_xmx',w.clone(),None),('baseline_mode','baseline',w,None)]:
                provider.select(mode);before=dict(provider.k8_calls)
                expected=fused_dot(probe,weight,chunk_k=8,initial=ini)
                actual=provider.dense(probe,weight,chunk_k=8,initial=ini)
                assert raw(actual)==raw(expected) and provider.k8_calls==before
                report['cases'].append(dict(weight=name,name=label,fallback=True,byte_equal=True));save()
            provider.select('fp16_xmx');print('Validated finite encoding, ragged, stride and fallback cases for '+name,flush=True)
        assert all(raw(w)==arrays.load(next(c['arrays']['w'] for c in prior['cases'] if c['name']==name)).tobytes() for name,w in weights.items())
        report.update(passed=True,weights_unchanged=True,full_outputs_compared=len(report['cases']),finite_half_input_encodings_per_weight=63488)
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    assert all(sha(p)==h for p,h in sources.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],cases=len(report['cases']))),flush=True)
