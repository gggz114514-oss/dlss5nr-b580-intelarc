"""View identity, mutation invalidation, no dynamic-operand alias and byte equality."""
import hashlib,json,os,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=DREF/'experimental/static-weight-cache-v2'
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-sm89-v1')
sys.path.insert(0,str(HERE.parent/'backend'))
import torch
import numpy as np
from fast_matrices_v3 import dot
from static_weight_cache_v2 import StaticWeightCache
import immutable_artifacts_v1 as artifacts
assert not (OUT/'primitive.json').exists()
OUT.mkdir(parents=True,exist_ok=True)
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'static_weight_cache_v2.py',HERE/'fast_matrices_v3.py',HERE/'Run-StaticCacheV2.cmd']}
report=dict(scope=__doc__,sources=sources,passed=False,cases=[])
try:
    torch.set_num_threads(2)
    rng=np.random.default_rng(90610)
    model=torch.nn.Module()
    values=rng.uniform(-.9,.9,(3,32,64)).astype('f2')
    # Construct normal tensors so their mutation counter remains enforceable.
    model.register_buffer('branches',torch.from_numpy(values).to('xpu'))
    model.register_buffer('plain',torch.from_numpy(values[0].copy()).to('xpu'))
    cache=StaticWeightCache()
    with torch.inference_mode():
        cache.prepack(model)
        assert len(cache.entries)==4
        a=torch.from_numpy(rng.uniform(-1,1,(17,32)).astype('f2')).to('xpu')
        for i in range(3):
            w=model.branches[i]
            packed=cache.lookup(w)
            assert packed is not None
            actual,_=dot(a,w,int8=True,packed=packed)
            expected,_=dot(a,w,int8=True)
            av,ev=actual.cpu().numpy(),expected.cpu().numpy()
            assert av.tobytes()==ev.tobytes()
            report['cases'].append(dict(branch=i,byte_equal=True,actual=artifacts.array(av),expected=artifacts.array(ev)))
        assert cache.lookup(model.branches[0].clone()) is None
        assert cache.lookup(model.branches[0].T) is None
        assert cache.lookup(model.branches[0].reshape(16,128)) is None
        before=model.branches._version
        model.branches[1].mul_(.5)
        assert model.branches._version>before
        assert all(cache.lookup(model.branches[i]) is None for i in range(3))
        assert cache.lookup(model.plain) is not None
        cache.prepack(model)
        actual,_=dot(a,model.branches[1],int8=True,packed=cache.lookup(model.branches[1]))
        expected,_=dot(a,model.branches[1],int8=True)
        assert actual.cpu().numpy().tobytes()==expected.cpu().numpy().tobytes()
        inference_model=torch.nn.Module()
        inference_model.register_buffer('weight',model.plain.clone())
        try:cache.prepack(inference_model)
        except ValueError as error:assert 'immutability' in str(error)
        else:raise AssertionError('Implicit inference-weight immutability accepted')
        cache.prepack(inference_model,assume_immutable=True)
        assert cache.lookup(inference_model.weight) is not None
        report.update(passed=True,mutation_invalidates_all_sibling_views=True,unrelated_buffer_stays_valid=True,repack_updates_weights=True,dynamic_copy_shape_and_stride_misses=True,explicit_inference_immutability_required=True)
except Exception as error:
    report['error']=repr(error);report['traceback']=traceback.format_exc();raise
finally:
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sources.items())
    (OUT/'primitive.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in report.items() if k not in ('sources','cases')}),flush=True)
