"""Full front tensor byte checks: real RGB/history, dynamic seeds and uint32 wrap.

FP16 reduced256 and float32 full480 inputs. Compare original front to fused
front for reset and temporal use. Primitive timing includes Python submission
and completion, but excludes validation/readback and is not full NR timing.
"""
import hashlib, json, os, statistics, sys, time, traceback, urllib.request
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; EXACT=ROOT.parent/'nr-b580'; R=EXACT/'reference'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference'); OUT=DREF/'experimental/fused-dynamic-front-v1'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1')
sys.path[:0]=[str(R),str(ROOT/'backend')]
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main()
import numpy as np
from PIL import Image
import torch
from nr_backend.front import reset_front_features,zero_motion_front_features
from nr_backend.noise import NativeNoiseTable
from fused_dynamic_front_v1 import forward
import compressed_arrays_v1 as arrays
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
reduced_path=DREF/'results/residual-scale-fp16_xmx-256-v1/validation.json'
full_path=DREF/'results/fast-precision-864x480-v1/validation.json'
assert sha(reduced_path)=='056251b4e2a839fcc44845d9646d2cd8ae1e9ed7f99fd1858dc7e3d51e3e50f9'
assert sha(full_path)=='7a43ff73d452cabe3e68ea1a15db1537417384a57c9a2443fcb16b095d5a5110'
reduced,full=js(reduced_path),js(full_path)
inputs=R/'inputs/flow-full-864x480-v2'; manifest=js(inputs/'manifest.json')
paths=[Path(__file__),HERE/'Run-DynamicFrontV1.cmd',HERE/'fused_dynamic_front_v1.py',HERE/'compressed_arrays_v1.py',
       HERE/'immutable_artifacts_v1.py',reduced_path,full_path,inputs/'manifest.json',*sorted((ROOT/'backend/nr_backend').glob('*.py'))]
frozen={str(p):sha(p) for p in paths}
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response: queue=json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
OUT.mkdir(); report=dict(scope=__doc__,passed=False,sources=frozen,exact_gate=gate,cases=[],complete_migration=False)
def save(): (OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
try:
    torch.set_num_threads(2)
    noise=NativeNoiseTable.from_directory(EXACT/'model-assets/noise-sm89-v2').to('xpu')
    datasets=[(arrays.load(reduced['primitive_checks']['first_canvas']),arrays.load(reduced['runs'][0]['low_nr']),(320,320)),
              (np.asarray(Image.open(inputs/manifest['frames'][0]['file']).convert('RGB'),dtype='f4')/255,
               np.load(full['runs']['fp16_xmx'][0]['actual']['path'],allow_pickle=False).astype('f2'),(512,896))]
    with torch.inference_mode():
        for pixels,history,pad in datasets:
            rgb,prev=torch.from_numpy(pixels).to('xpu'),torch.from_numpy(history).to('xpu')
            for temporal in (False,True):
                for seed in (0,1,243,0xfffffffe,0xffffffff):
                    options=dict(padded_size=pad,seed=seed,noise_source=noise)
                    old=lambda:zero_motion_front_features(rgb,prev,**options) if temporal else reset_front_features(rgb,**options)
                    new=lambda:forward(rgb,previous=prev if temporal else None,**options)
                    expected=old().cpu().numpy();actual=new().cpu().numpy()
                    equal=actual.tobytes()==expected.tobytes()
                    row=dict(shape=list(pixels.shape),input_dtype=str(pixels.dtype),temporal=temporal,seed=seed,
                             expected=arrays.save(expected),byte_equal=equal,
                             changed_half_components=int(np.count_nonzero(actual.view('u2')!=expected.view('u2'))))
                    if not equal: row['actual']=arrays.save(actual)
                    report['cases'].append(row);save();assert equal
                    if seed==1:
                        samples={'original':[],'fused':[]}
                        for repeat in range(5):
                            for name in (('original','fused') if repeat%2==0 else ('fused','original')):
                                fn=old if name=='original' else new
                                torch.xpu.synchronize();start=time.perf_counter()
                                for _ in range(8):value=fn()
                                torch.xpu.synchronize();samples[name].append((time.perf_counter()-start)/8)
                                assert value.cpu().numpy().tobytes()==expected.tobytes()
                        row['samples_seconds']=samples
                        row['median_seconds']={name:statistics.median(v) for name,v in samples.items()}
                    print(json.dumps({k:v for k,v in row.items() if k!='expected'}),flush=True)
            assert rgb.cpu().numpy().tobytes()==pixels.tobytes() and prev.cpu().numpy().tobytes()==history.tobytes()
        report.update(passed=True,caller_inputs_unchanged=True,full_fronts_verified=20)
except Exception as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    assert all(sha(p)==h for p,h in frozen.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],cases=len(report['cases']))),flush=True)
