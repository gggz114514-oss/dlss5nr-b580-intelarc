"""Audit complete FMA probe arrays and exact half-domain coverage.

Native raw NaN payload differences remain explicitly recorded. CPU arithmetic
checks are limited to finite inputs of the two affine/clamped exp transforms
and normalization's nonnegative square sum, not general arbitrary FMA triples.
"""
import hashlib,json,statistics
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays
D=Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/native-half-fma-v1')
target=D/'saved-audit-v1.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
p=D/'validation.json';r=js(p);lease=D.with_suffix('.log.lease.json')
assert r['passed'] and not r['complete_migration'] and js(lease)['returncode']==0
assert all(sha(p)==h for p,h in r['sources'].items())
assert [c['name'] for c in r['cases']]==['raw_random','norm_square','swin_exp_all_half','vit_exp_all_half']
verified={};cpu_counts={}
def read(meta):
    a=arrays.load(meta);verified[meta['path']]=Path(meta['path']).stat().st_size;return a
for c in r['cases']:
    inputs=[read(m) for m in c['input_arrays']];outputs=[read(v['output']) for v in c['candidates']]
    assert all(a.size==c['elements'] and a.dtype.str=='<f2' for a in outputs)
    for i,(v,a) in enumerate(zip(c['candidates'],outputs)):
        ref=outputs[0];different=a.view('u2')!=ref.view('u2')
        assert v['method']==i and v['compiled'] and v['different_values']==int(different.sum())
        assert v['byte_equal']==(not different.any())
        count=int((different&~(np.isnan(a)&np.isnan(ref))).sum())
        assert v['differences_excluding_both_nan']==count==0
        assert v['median_seconds']==statistics.median(v['samples_seconds']) and len(v['samples_seconds'])==5
        for kind,meta in v['ir'].items():assert sha(meta['path'])==meta['sha256']
        llvm=Path(v['ir']['llir']['path']).read_text()
        if i==1:assert '@llvm.fma.f16' in llvm
        if i==2:assert '__spirv_ocl_fmaDF16_' in llvm
    if c['domain']==0:
        rng=np.random.Generator(np.random.PCG64(r['seed']))
        assert all(a.tobytes()==rng.integers(0,65536,size=1048576,dtype=np.uint16).tobytes() for a in inputs)
        assert c['candidates'][1]['different_values']==c['candidates'][2]['different_values']==2528
    elif c['domain']==1:
        ys=np.array(r['normalization_y_bits'],dtype='u2').view('f2')
        assert all(a.tobytes()==ys.tobytes() for a in inputs) and c['elements']==65536*len(ys)
        x=np.arange(65536,dtype='u2').view('f2');count=0
        with np.errstate(all='ignore'):
            for j,y in enumerate(ys):
                if not np.isfinite(y):continue
                a=np.float16(float(y)*float(y));valid=np.isfinite(x)
                expected=(x.astype('f8')**2+float(a)).astype('f2')
                got=outputs[0][j*65536:(j+1)*65536]
                assert got[valid].tobytes()==expected[valid].tobytes()
                count+=int(valid.sum())
        cpu_counts[c['name']]=count
    else:
        assert c['elements']==65536
        x=np.arange(65536,dtype='u2').view('f2');valid=np.isfinite(x)
        a,b,lo,hi,shift,offset=((.044921875,1.30078125,1.03125,1.5693359375,5,0x8000) if c['domain']==2
                              else (.08953857421875,1.708984375,1.439453125,1.9775390625,4,0x4000))
        with np.errstate(all='ignore'):
            affine=(x.astype('f8')*a+b).astype('f2')
            half=np.clip(affine.astype('f4'),lo,hi).astype('f2')
            expected=(((half.view('u2').astype('u4')<<shift)+offset)&65535).astype('u2').view('f2')
        assert outputs[0][valid].tobytes()==expected[valid].tobytes()
        cpu_counts[c['name']]=int(valid.sum())
    assert c['operands_unchanged']
audit=dict(passed=True,report_sha256=sha(p),auditor_sha256=sha(__file__),lease_sha256=sha(lease),
    unique_arrays_reread=len(verified),unique_array_bytes=sum(verified.values()),cpu_finite_domain_counts=cpu_counts,
    raw_nan_payload_difference_retained=2528,generic_fma_replacement_approved=False,limitation=__doc__,complete_migration=False)
target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');print(json.dumps(audit,indent=2),flush=True)
