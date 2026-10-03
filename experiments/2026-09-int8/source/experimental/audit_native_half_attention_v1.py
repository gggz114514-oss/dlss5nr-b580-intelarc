"""Authenticate native-half attention comparisons and all reused complete tensors.

GPU-run outputs were compared against previous kernels before/after replay. CPU
audit rereads saved artifacts and coverage records, not an independent model run.
"""
import hashlib,json,statistics
from pathlib import Path
import compressed_arrays_v1 as arrays
D=Path('D:/Codex-NR-Experiments/nr-b580/reference');out=D/'experimental/native-half-attention-v1'
p=out/'validation.json';target=out/'saved-audit-v1.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
r=js(p);lease=out.with_suffix('.log.lease.json')
assert r['passed'] and not r['complete_migration'] and js(lease)['returncode']==0
assert all(sha(p)==h for p,h in r['sources'].items())
counts={'c32':14,'qkv':54,'heads':54,'swin':4};assert r['primitive_comparisons']==sum(counts.values())==126
old_c32=js(D/'experimental/c32-projection-pack-v1/validation.json')['cases']
old_qkv=js(D/'experimental/fused-qkv-pack-v1/validation.json')['cases']
old_swin=js(D/'experimental/fused-swin-core-v2/validation.json')['cases']
verified={}
def read(meta):
    a=arrays.load(meta);verified[meta['path']]=Path(meta['path']).stat().st_size;return a
for section,count in counts.items():
    rows=r[section];assert len(rows)==count
    for i,row in enumerate(rows):
        assert row['byte_equal']
        operands={k:read(m) for k,m in row['arrays'].items()}
        expected=[read(m) for m in row['expected']]
        if section in ('c32','qkv'):
            prior=(old_c32 if section=='c32' else old_qkv)[i]
            assert row['name']==prior['name'] and row['arrays']==prior['arrays']
            assert len(expected)==3
            assert all(a.tobytes()==operands[k].tobytes() for a,k in zip(expected,('q','k','v')))
        elif section=='heads':
            assert row['name']==old_qkv[i]['name'] and len(expected)==1
            assert all(row['arrays'][k]==old_qkv[i]['arrays'][k] for k in ('q','k','v'))
            assert expected[0].shape==operands['q'].shape
            assert operands['bias'].shape==(operands['q'].shape[0],64,64)
        else:
            assert row['arrays']==old_swin[i]['operands'] and len(expected)==1
            assert expected[0].shape==operands['query'].shape
            # Old reference output uses the same complete window order.
            assert expected[0].tobytes()==read(old_swin[i]['expected']).tobytes()
        for kind,m in row['ir'].items():assert sha(m['path'])==m['sha256']
        llvm=Path(row['ir']['llir']['path']).read_text();assert '@llvm.fma.f16' in llvm
        if row['timed']:
            assert len(row['samples_seconds'])==2 and all(len(s)==5 and min(s)>0 for s in row['samples_seconds'])
            assert row['median_seconds']==[statistics.median(s) for s in row['samples_seconds']]
            assert row['orders']==[[0,1],[1,0],[0,1],[1,0],[0,1]]
assert [c['name'] for c in r['body']]==['previous','normalize','swin','both']
body=js(D/'experimental/small-branched-mlp-operands-v1/validation.json')
assert all(c['byte_equal'] and c['dispatch']==body['dispatch'] for c in r['body'])
timing=r['body_timing'];assert len(timing['samples_seconds'])==4
assert timing['median_seconds']==[statistics.median(s) for s in timing['samples_seconds']]
assert timing['orders']==[[0,1,2,3],[1,2,3,0],[2,3,0,1],[3,0,1,2],[0,1,2,3]]
assert r['lut_bytes_unchanged'] and r['operands_unchanged']
summary=dict(passed=True,report_sha256=sha(p),auditor_sha256=sha(__file__),lease_sha256=sha(lease),
    primitive_comparisons=126,whole_body_variants=4,unique_arrays_reread=len(verified),unique_array_bytes=sum(verified.values()),
    body_median_ms=dict(zip([x['name'] for x in r['body']],[v*1000 for v in timing['median_seconds']])),
    limitation=__doc__,complete_migration=False)
target.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)
