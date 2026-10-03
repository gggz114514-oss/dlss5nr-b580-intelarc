"""Primitive checks against independently calculated ordinary/mixed precision math."""
import hashlib,json,os,sys,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=DREF/'experimental/fast-matrices-v1'
os.environ['TRITON_CACHE_DIR']=str(DREF/'triton-cache-fast-matrices-v1')
sys.path.insert(0,str(ROOT/'backend'))
import numpy as np
import torch
from fast_matrices_v1 import dot,quantize
import immutable_artifacts_v1 as artifacts
assert not (OUT/'primitive-v2.json').exists()
OUT.mkdir(parents=True,exist_ok=True)
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),HERE/'fast_matrices_v1.py',HERE/'immutable_artifacts_v1.py',HERE/'Run-MatricesV2.cmd',*sorted((ROOT/'backend/nr_backend').glob('*.py'))]}
report=dict(scope=__doc__,sources=sources,cases=[],passed=False,bit_exact_nr_claim=False)

def qcpu(x,columns=False):
    x=x.T if columns else x
    x=np.asarray(x,dtype='f4')
    maximum=np.max(np.abs(x),axis=1)
    scale=np.where(maximum>0,maximum/np.float32(127),np.float32(1))
    norm=x/scale[:,None]
    q=(np.minimum(np.floor(np.abs(norm)+np.float32(.5)),127)*np.where(norm<0,-1,1)).astype('i1')
    return q,scale

try:
    torch.set_num_threads(2)
    rng=np.random.default_rng(90609)
    with torch.inference_mode():
        for batched,m,k,n in [(False,17,16,37),(False,32,32,128),(False,65,128,32),(True,17,16,37),(True,64,64,32)]:
            shape=(2,m,k) if batched else (m,k)
            wshape=(2,k,n) if batched else (k,n)
            a=rng.uniform(-1,1,shape).astype('f2')
            w=rng.uniform(-1,1,wshape).astype('f2')
            a[...,0,:]=0
            initial=rng.uniform(-.2,.2,(*shape[:-1],n)).astype('f2')
            aa=torch.from_numpy(a).to('xpu')
            ww=torch.from_numpy(w).to('xpu')
            # Force noncontiguous operands, including a batched weight.
            aa=aa.transpose(-1,-2).contiguous().transpose(-1,-2)
            ww=ww.transpose(-1,-2).contiguous().transpose(-1,-2)
            for integer in ([False] if batched else [False,True]):
                packed=None
                if integer:
                    qa,sa=qcpu(a);qw,sw=qcpu(w,columns=True)
                    ag,sg,_=quantize(aa)
                    wg,tg,_=quantize(ww,columns=True)
                    assert np.array_equal(ag.cpu().numpy(),qa)
                    assert np.array_equal(wg.cpu().numpy(),qw)
                    assert np.array_equal(sg.cpu().numpy(),sa)
                    assert np.array_equal(tg.cpu().numpy(),sw)
                    packed=(wg,tg)
                    expected=(qa.astype('i4')@qw.astype('i4').T).astype('f4')*sa[:,None]*sw[None,:]
                else:expected=(a.astype('f8')@w.astype('f8')).astype('f4')
                for initialized in (False,True):
                    ini=torch.from_numpy(initial).to('xpu') if initialized else None
                    out,kernel=dot(aa,ww,initial=ini,batched=batched,int8=integer,packed=packed)
                    actual=out.cpu().numpy()
                    target=(expected+initial.astype('f4') if initialized else expected).astype('f2')
                    assert np.isfinite(actual).all()
                    np.testing.assert_allclose(actual.astype('f4'),target.astype('f4'),rtol=.001,atol=.001)
                    assert np.array_equal(aa.cpu().numpy(),a) and np.array_equal(ww.cpu().numpy(),w)
                    ir=str(kernel.asm['llir'])
                    assert 'SubgroupMatrixMultiplyAccumulateINTEL' in ir and 'ttig.dpas' in str(kernel.asm['ttgir']), 'Missing Intel matrix lowering'
                    row=dict(shape=list(shape),weight_shape=list(wshape),mode='int8_dense' if integer else 'fp16_xmx',initial=initialized,max_abs=float(np.max(np.abs(actual.astype('f4')-target.astype('f4')))),actual=artifacts.array(actual),expected=artifacts.array(target),llir=artifacts.text(ir,'llir'),dpas=True)
                    report['cases'].append(row)
                    print(json.dumps({key:row[key] for key in ('shape','mode','initial','max_abs','dpas')}),flush=True)
        report['passed']=True
except Exception as error:
    report['error']=repr(error)
    report['traceback']=traceback.format_exc()
    raise
finally:
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in sources.items())
    (OUT/'primitive-v2.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
