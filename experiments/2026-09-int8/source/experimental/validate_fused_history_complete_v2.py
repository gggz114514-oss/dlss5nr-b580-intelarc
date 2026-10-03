"""Validate complete fused history outputs, finite edges and adapter guards.

Eleven actual temporal operands retain frozen numerator/reciprocal references.
Synthetic square, coefficient-edge, signed-zero, finite-half and dtype cases
compare against the unchanged original history adapter. Complete outputs are
read back; errors and table changes are checked before any model integration.
"""
import hashlib,json,os,statistics,sys,time,traceback,urllib.request
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580'
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=D/'experimental/fused-history-complete-v2';assert not OUT.exists()
TOOLCHAIN=D/'toolchains/triton-xpu-3.8.0-git1e2d42a0'
os.environ['TRITON_CACHE_DIR']=str(D/'triton-cache-c32-triton38-v1')
sys.path[:0]=[str(TOOLCHAIN/'site'),str(EXACT/'reference'),str(ROOT/'backend')]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main();screen_path=D/'experimental/fused-history-screen-v1/validation.json'
assert sha(screen_path)=='21e11d308898c383a424e6ce85e32eed17e9316c986fab2a6cfc1581af19653a'
screen=js(screen_path);assert screen['passed'] and js(screen_path.parent.with_suffix('.log.lease.json'))['returncode']==0
prior=js(D/'experimental/k8-outer-profile-v2/validation.json')
sources=dict(screen['sources']);sources[str(screen_path)]=sha(screen_path)
provision_path=TOOLCHAIN/'provision-v1.json'
assert all(sha(TOOLCHAIN/'site'/p)==h for p,h in js(provision_path)['files'].items())
for p in (Path(__file__),HERE/'Run-FusedHistoryCompleteV2.cmd',HERE/'fused_square_history_v2.py',HERE/'fused_graph_history_warp_v2.py',EXACT/'model-assets/reciprocal-sm89-v1/reciprocal.f32.bin'):sources[str(p)]=sha(p)
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
from fused_graph_history_warp_v2 import FusedHistoryWarp
import compressed_arrays_v1 as arrays
OUT.mkdir();report=dict(scope=__doc__,sources=sources,exact_gate=gate,passed=False,cases=[],guards=[],timing=[],complete_migration=False)
raw=lambda t:t.cpu().numpy().tobytes();adapters=[]
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
try:
    torch.set_num_threads(2);rng=np.random.default_rng(640320)
    table=NativeReciprocalTable.from_directory(EXACT/'model-assets/reciprocal-sm89-v1').to('xpu')
    model=SimpleNamespace(reciprocal=table)
    old=GraphHistoryWarp(model);new=FusedHistoryWarp(model,block=64);adapters=[old,new]
    table_before=raw(table.values);held=[]
    cases=[dict(name='actual'+str(c['frame']),arrays=c['arrays'],actual=True) for c in prior['warp_cases']]
    base=arrays.load(cases[0]['arrays']['image']);base_motion=arrays.load(cases[0]['arrays']['motion'])
    def synthetic(name,image,motion):
        cases.append(dict(name=name,actual=False,arrays=dict(image=arrays.save(image),motion=arrays.save(motion))))
    synthetic('zero_motion',base,np.zeros((256,256,2),dtype='f2'))
    finite=np.arange(65536,dtype='u2');finite=finite[(finite&0x7c00)!=0x7c00];rng.shuffle(finite)
    mixed=np.resize(finite,256*256*3).view('f2').reshape(256,256,3)
    synthetic('all_finite_half_colors',mixed,rng.uniform(-2,2,(256,256,2)).astype('f2'))
    signed=np.zeros((256,256,3),dtype='u2');signed.reshape(-1)[1::2]=0x8000
    synthetic('signed_zeros',signed.view('f2'),rng.uniform(-2,2,(256,256,2)).astype('f2'))
    fractions=np.asarray([0,1/512,3/512,127/256,.5,255/256,1-1/512,-1/512,-.5],dtype='f2')
    synthetic('coefficient_half_steps',base,np.resize(fractions,256*256*2).reshape(256,256,2))
    border=np.asarray([-65504,65504,-512,512,-256,256,-1.5,1.5],dtype='f2')
    synthetic('border_clamp',base,np.resize(border,256*256*2).reshape(256,256,2))
    synthetic('square512',np.tile(base,(2,2,1)),np.tile(base_motion,(2,2,1)))
    f32=np.resize(np.asarray([.50024414,.5002442,.5002441,-.50024414,-.5002442,-.5002441],dtype='f4'),256*256*2).reshape(256,256,2)
    synthetic('float_motion_rounding',base,f32)
    synthetic('float_history_fallback',base.astype('f4'),base_motion)
    def run(adapter,image,motion):return adapter.apply(image,motion,return_components=True,reciprocal_source=table)
    with torch.inference_mode(),use_arithmetic_backend('triton'):
        for index,case in enumerate(cases):
            meta=case['arrays'];image=torch.from_numpy(arrays.load(meta['image'])).to('xpu');motion=torch.from_numpy(arrays.load(meta['motion'])).to('xpu')
            reference=run(old,image,motion);expected=[v.cpu().numpy() for v in reference]
            if case['actual']:
                assert [a.tobytes() for a in expected]==[arrays.load(meta[k]).tobytes() for k in ('numerator','reciprocal')]
            else:
                meta.update({k:arrays.save(a) for k,a in zip(('numerator','reciprocal'),expected)})
            builds=new.fused_builds
            result=run(new,image,motion);actual=[v.cpu().numpy() for v in result]
            equal=[a.tobytes()==b.tobytes() for a,b in zip(actual,expected)]
            row=dict(name=case['name'],actual=case['actual'],arrays=meta,byte_equal=all(equal),
                different_float_words=[int(np.count_nonzero(a.view('u4')!=b.view('u4'))) for a,b in zip(actual,expected)],
                output_raw_sha256=[hashlib.sha256(a.tobytes()).hexdigest() for a in actual])
            if not all(equal):row['actual_outputs']=[arrays.save(a) for a in actual]
            report['cases'].append(row);save();assert all(equal),row
            if case['name']=='float_history_fallback':assert new.fused_builds==builds
            if index==0:held.append((result,[raw(v) for v in result]))
            else:
                for v in result:v.zero_()
                check=run(new,image,motion);assert [raw(v) for v in check]==[a.tobytes() for a in expected]
            for values,bytes_before in held:assert [raw(v) for v in values]==bytes_before
            assert raw(image)==arrays.load(meta['image']).tobytes() and raw(motion)==arrays.load(meta['motion']).tobytes()
            if case['name'] in ('actual1','square512'):
                for adapter in adapters:
                    for _ in range(2):run(adapter,image,motion)
                samples=[[],[]]
                for r in range(5):
                    for i in ([0,1] if r%2==0 else [1,0]):
                        torch.xpu.synchronize();started=time.perf_counter()
                        for _ in range(10):v=run(adapters[i],image,motion)
                        torch.xpu.synchronize();samples[i].append((time.perf_counter()-started)/10)
                        assert [raw(t) for t in v]==[a.tobytes() for a in expected]
                report['timing'].append(dict(name=case['name'],labels=['previous','fused64'],samples_seconds=samples,
                    median_seconds=[statistics.median(s) for s in samples],adapter_calls_per_variant=50,full_readbacks_per_variant=5))
            print('Validated complete history '+case['name'],flush=True)
        image=torch.from_numpy(base).to('xpu');motion=torch.from_numpy(base_motion).to('xpu')
        for invalid in (float('nan'),float('inf')):
            bad=motion.float().clone();bad[0,0,0]=invalid;errors=[]
            for adapter in adapters:
                print('Checking invalid '+str(invalid)+' '+type(adapter).__name__,flush=True)
                try:run(adapter,image,bad)
                except ValueError as error:
                    assert str(error)=='Input outside the validated native reciprocal interval';errors.append(str(error))
                else:raise AssertionError('Invalid domain accepted')
            report['guards'].append(dict(kind='nonfinite_motion',value=str(invalid),both_rejected=True,errors=errors))
        for values,bytes_before in held:assert [raw(v) for v in values]==bytes_before
        original=table.values
        with torch.inference_mode(False):replacement=original.clone()
        table.values=replacement
        try:
            for adapter in adapters:
                before=adapter.replays
                try:run(adapter,image,motion)
                except RuntimeError as error:assert 'reciprocal table changed' in str(error)
                else:raise AssertionError('Table replacement accepted')
                assert adapter.replays==before
        finally:table.values=original
        report['guards'].append(dict(kind='table_replacement',both_rejected=True,no_replay=True))
        with torch.no_grad():original.add_(0)
        for adapter in adapters:
            before=adapter.replays
            try:run(adapter,image,motion)
            except RuntimeError as error:assert 'reciprocal table changed' in str(error)
            else:raise AssertionError('In-place table change accepted')
            assert adapter.replays==before
        report['guards'].append(dict(kind='table_version',both_rejected=True,no_replay=True))
        assert raw(table.values)==table_before
        for adapter in adapters:
            adapter.close()
            try:run(adapter,image,motion)
            except RuntimeError as error:assert 'closed' in str(error)
            else:raise AssertionError('Closed session accepted')
        report['guards'].append(dict(kind='closed',both_rejected=True))
        report.update(passed=True,complete_cases=len(cases),fused_builds=new.fused_builds,inputs_and_table_bytes_unchanged=True,
            held_outputs_survive_replay=True,caller_owned_outputs=True,float_image_fallback=True)
except BaseException as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    for adapter in adapters:adapter.close()
    assert all(sha(p)==h for p,h in sources.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],cases=report.get('complete_cases'),timing=report['timing'])),flush=True)
