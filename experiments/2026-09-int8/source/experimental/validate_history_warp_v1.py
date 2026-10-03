"""Whole square warp bytes, invalid-domain recovery and ownership checks.

Uses saved NR history, real low motion, spatial gradients, zero and FP16-range
limits. NaN is passed directly to the internal helper to test its domain guard;
the public MotionNR API already rejects nonfinite motion before this helper.
"""
import hashlib,json,os,sys,traceback,urllib.request
from pathlib import Path
from types import SimpleNamespace
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;EXACT=ROOT.parent/'nr-b580';R=EXACT/'reference'
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference');OUT=DREF/'experimental/graph-history-warp-v2'
assert not OUT.exists()
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1');sys.path[:0]=[str(R),str(ROOT/'backend')]
from attention_weights_main_gate_v1 import authenticate_main
gate=authenticate_main()
import numpy as np
import torch
from nr_backend.reciprocal import NativeReciprocalTable
from nr_backend.execution import use_arithmetic_backend
from graph_history_warp_v2 import GraphHistoryWarp
import compressed_arrays_v1 as arrays
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
prior_path=DREF/'results/residual-scale-fp16_xmx-256-v1/validation.json';prior=js(prior_path)
assert sha(prior_path)=='056251b4e2a839fcc44845d9646d2cd8ae1e9ed7f99fd1858dc7e3d51e3e50f9'
paths=[Path(__file__),HERE/'Run-HistoryWarpV1.cmd',HERE/'graph_history_warp_v2.py',HERE/'graph_history_warp_v1.py',
       HERE/'compressed_arrays_v1.py',HERE/'immutable_artifacts_v1.py',prior_path,*sorted((ROOT/'backend/nr_backend').glob('*.py'))]
frozen={str(p):sha(p) for p in paths}
with urllib.request.urlopen('http://127.0.0.1:8188/queue',timeout=5) as response: queue=json.load(response)
assert not queue['queue_running'] and not queue['queue_pending']
OUT.mkdir();report=dict(scope=__doc__,passed=False,cases=[],sources=frozen,exact_gate=gate,complete_migration=False)
warp=None
def save():(OUT/'validation.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8',newline='\n')
try:
    torch.set_num_threads(2)
    table=NativeReciprocalTable.from_directory(EXACT/'model-assets/reciprocal-sm89-v1').to('xpu')
    warp=GraphHistoryWarp(SimpleNamespace(reciprocal=table))
    image_data=arrays.load(prior['runs'][0]['low_nr'])
    motion_data=arrays.load(prior['primitive_checks']['first_low_motion'])
    yy,xx=np.mgrid[:256,:256]
    gradient=np.stack(((xx-128)/13.,(yy-128)/17.),-1).astype('f2')
    image=torch.from_numpy(image_data).to('xpu');held=[]
    with torch.inference_mode(),use_arithmetic_backend('triton'):
        for name,flow in [('recorded',motion_data),('zero',np.zeros_like(motion_data)),('gradient',gradient),
                          ('limits',np.broadcast_to(np.array([65504,-65504],dtype='f2'),motion_data.shape).copy())]:
            motion=torch.from_numpy(flow).to('xpu')
            old=warp.original(image,motion,return_components=True,reciprocal_source=table)
            new=warp.apply(image,motion,return_components=True,reciprocal_source=table)
            expected=[t.cpu().numpy() for t in old];actual=[t.cpu().numpy() for t in new]
            assert all(a.tobytes()==b.tobytes() for a,b in zip(expected,actual))
            assert all(np.isfinite(a).all() for a in actual)
            for tensors,values in held:
                assert all(t.cpu().numpy().tobytes()==v for t,v in zip(tensors,values))
            if not held:held.append((new,[a.tobytes() for a in actual]))
            else:
                new[0].zero_()
                entry=next(iter(warp.entries.values()))
                assert entry.outputs[0].cpu().numpy().tobytes()==expected[0].tobytes()
            assert image.cpu().numpy().tobytes()==image_data.tobytes() and motion.cpu().numpy().tobytes()==flow.tobytes()
            report['cases'].append(dict(name=name,motion=arrays.save(flow),expected=[arrays.save(a) for a in expected],byte_equal=True))
        bad=torch.full_like(motion,float('nan'));errors=[]
        for fn in (warp.original,warp.apply):
            try:fn(image,bad,return_components=True,reciprocal_source=table)
            except ValueError as error:errors.append(str(error))
            else:raise AssertionError('Invalid reciprocal domain accepted')
        motion=torch.from_numpy(motion_data).to('xpu')
        recovered=warp.apply(image,motion,return_components=True,reciprocal_source=table)
        assert all(t.cpu().numpy().tobytes()==arrays.load(m).tobytes() for t,m in zip(recovered,report['cases'][0]['expected']))
        before=warp.replays;original_table=table.values
        table.values=table.values.clone()
        try:warp.apply(image,motion,return_components=True,reciprocal_source=table)
        except RuntimeError:pass
        else:raise AssertionError('Changed reciprocal table accepted')
        assert warp.replays==before
        table.values=original_table
        warp.close()
        try:warp.apply(image,motion,return_components=True,reciprocal_source=table)
        except RuntimeError:pass
        else:raise AssertionError('Closed session accepted')
        report.update(passed=True,invalid_domain_errors=errors,recovery_bytes_equal=True,table_replacement_rejected_before_replay=True,
                      closed_session_rejected=True,held_outputs_preserved=True,caller_mutation_does_not_touch_internal_output=True,
                      caller_inputs_unchanged=True,history=arrays.save(image_data))
except Exception as error:
    report.update(error=repr(error),traceback=traceback.format_exc());raise
finally:
    if warp is not None:warp.close()
    assert all(sha(p)==h for p,h in frozen.items());authenticate_main();save()
print(json.dumps(dict(passed=report['passed'],cases=len(report['cases']))),flush=True)
