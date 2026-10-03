"""Authenticate the complete current-stack capture and reread every saved array.

The complete body comparison took place on GPU; this is an artifact audit.
"""
import hashlib,json
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays

root=Path('D:/Codex-NR-Experiments/nr-b580/reference')
out=root/'experimental/current-dense-operands-v2'
path=out/'validation.json';target=out/'saved-audit-v1.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(path)=='46a16ba64e9f84425928d0a0fd93862e99574910b031ba098c789f0ca69e7290'
r=js(path);lease=out.with_suffix('.log.lease.json')
assert r['passed'] and js(lease)['returncode']==0 and not r['complete_migration']
assert r['byte_equal_prior'] and r['history_unchanged'] and r['vit_qkv_calls']==8
assert all(sha(p)==h for p,h in r['sources'].items())
prior=js(root/'results/residual-scale-fp16_xmx-256-v1/validation.json')['runs'][1]
assert r['output']==prior['low_nr'] and r['dispatch']==prior['captured_dispatch']
assert r['nr_input']==[256,256] and r['source_canvas']==[1920,1080]
assert len(r['cases'])==34 and sum(c['count'] for c in r['cases'].values())==r['matrix_calls']['fp16_dense']==167
verified={};raw_bytes=0
for row in r['cases'].values():
    assert row['count']>0 and row['initialized']==(row['operands']['initial'] is not None)
    m,k,n=row['shape']
    for name,meta in row['operands'].items():
        desc=row['descriptors'][name]
        if meta is None:assert desc is None;continue
        value=arrays.load(meta);assert np.isfinite(value).all() and value.dtype==np.dtype('f2')
        assert list(value.shape)==desc['shape'] and len(desc['stride'])==value.ndim
        assert desc['dtype']=='torch.float16' and all(s>=0 for s in desc['stride'])
        if name=='a':assert value.size==m*k and value.shape[-1]==k
        if name=='w':assert value.shape==(k,n)
        if name=='initial':assert value.size==m*n
        raw_bytes+=value.nbytes;verified[meta['path']]=meta['stored_bytes']
    meta=row['output'];value=arrays.load(meta)
    assert np.isfinite(value).all() and value.size==m*n and value.dtype==np.dtype('f2')
    raw_bytes+=value.nbytes;verified[meta['path']]=meta['stored_bytes']
assert raw_bytes==r['operand_raw_bytes']<=r['maximum_operand_raw_bytes']
value=arrays.load(r['output']);assert value.shape==(256,256,3) and np.isfinite(value).all()
audit=dict(passed=True,report_sha256=sha(path),lease_sha256=sha(lease),auditor_sha256=sha(__file__),
    geometries=34,dense_calls=167,unique_arrays_reread=len(verified),stored_bytes_reread=sum(verified.values()),
    raw_case_bytes=raw_bytes,limitation=__doc__,complete_migration=False)
target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');print(json.dumps(audit,indent=2))
