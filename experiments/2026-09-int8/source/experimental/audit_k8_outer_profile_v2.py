"""Authenticate the completed host profile; no independent model arithmetic."""
import hashlib,json,statistics
from pathlib import Path
import compressed_arrays_v1 as arrays
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=D/'experimental/k8-outer-profile-v2';path=OUT/'validation.json'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(path)=='aa574c05863f78a4e09439e290c18534de5775d7194314df532b2ca0f426f1e0'
r=js(path);target=OUT/'saved-audit-v1.json';assert not target.exists()
lease=OUT.with_suffix('.log.lease.json')
assert r['passed'] and js(lease)['returncode']==0 and r['full_outputs_verified']==65 and len(r['runs'])==65
assert r['lut_unchanged'] and not r['complete_migration']
assert all(sha(p)==h for p,h in r['sources'].items())
old=js(D/'results/residual-scale-fp16_xmx-256-v1/validation.json')['runs'][:13]
verified=set()
for n,row in enumerate(r['runs']):
    j,i=divmod(n,13)
    assert (row['round'],row['frame'])==(j,i) and row['instrumented']==(j in (1,2,4))
    assert row['full_byte_equal_prior'] and row['low_byte_equal_prior'] and row['private_byte_equal_low']
    assert row['reset']==old[i]['reset'] and row['next_seed']==(1 if row['reset'] else i+1)
    for name in ('output','low_nr'):
        meta=row[name];assert meta==old[i][name]
        if meta['path'] not in verified:arrays.load(meta);verified.add(meta['path'])
    if row['instrumented']:
        p=row['parts_seconds'];assert min(p.values())>=0
        assert abs(row['seconds']-sum(p[n] for n in ('prepare','model_inclusive','composite','final_sync','pipeline_other')))<1e-10
    else:assert row['parts_seconds'] is None
for flag in (False,True):
    assert statistics.mean(v['seconds'] for v in r['runs'] if v['instrumented']==flag and not v['reset'])*1000==r['temporal_mean_ms'][str(flag)]
rows=[v for v in r['runs'] if v['instrumented'] and not v['reset']]
assert all(statistics.mean(v['parts_seconds'].get(k,0) for v in rows)*1000==x for k,x in r['host_inclusive_mean_ms'].items())
assert r['constant_buffer_count']==779 and len(r['isolated_host_constant_scan_seconds'])==5
assert len(r['warp_cases'])==11 and [c['frame'] for c in r['warp_cases']]==list(range(1,12))
warp_arrays=set();warp_bytes=0
for case in r['warp_cases']:
    assert case['valid']
    for name,meta in case['arrays'].items():
        a=arrays.load(meta);assert a.shape==((256,256,3) if name in ('image','numerator') else (256,256,2) if name=='motion' else (256,256))
        warp_arrays.add(meta['path']);warp_bytes+=meta['raw_bytes']
assert warp_bytes==r['warp_capture_raw_bytes']
audit=dict(warp_cases_verified=11,unique_warp_arrays_reread=len(warp_arrays),warp_raw_bytes=warp_bytes,passed=True,report_sha256=sha(path),auditor_sha256=sha(__file__),lease_sha256=sha(lease),
    unique_arrays_reread=len(verified),full_outputs_verified=65,temporal_mean_ms=r['temporal_mean_ms'],
    host_inclusive_mean_ms=r['host_inclusive_mean_ms'],complete_migration=False,scope=__doc__)
target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8');print(json.dumps(audit,indent=2))
