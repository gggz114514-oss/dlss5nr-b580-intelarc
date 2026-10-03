"""Authenticate the saved whole-window fusion experiment and complete references.

CPU reread validates artifacts and GPU comparison receipts, not an independent
CPU execution of the fused model. Timing is standalone block replay only.
"""
import argparse,hashlib,json,statistics
from pathlib import Path
import compressed_arrays_v1 as arrays
p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('report_sha256');args=p.parse_args()
D=Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental')
OUT=(D/args.folder).resolve();assert OUT.parent==D.resolve()
path=OUT/'validation.json';target=OUT/'saved-audit-v1.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(path)==args.report_sha256
r=js(path);lease=OUT.with_suffix('.log.lease.json');exitcode=js(lease)['returncode']
assert all(sha(p)==h for p,h in r['sources'].items())
assert not r['complete_migration']
verified={};comparisons=0;timings={};numeric_equal=True
def read(m):
    a=arrays.load(m);verified[m['path']]=m['stored_bytes'];return a
prior=js(D/'c32-window-blocks-v2/validation.json')
for i,c in enumerate(r['cases']):
    meta=c['arrays'];values={n:read(m) for n,m in meta.items()}
    if not c['synthetic']:
        assert c['name']==prior['cases'][i]['name'] and meta==prior['cases'][i]['arrays']
    expected=hashlib.sha256(values['output'].tobytes()).hexdigest()
    for candidate in c['candidates']:
        comparisons+=1;numeric_equal &= candidate['byte_equal']
        if candidate['byte_equal']:
            assert candidate['different_half_words']==0 and candidate['max_abs']==0
            assert candidate['output_sha256']==expected
        parts=candidate['ir'] if isinstance(candidate['ir'],list) else [candidate['ir']]
        for part in parts:
            for m in part.values():assert sha(m['path'])==m['sha256'] and Path(m['path']).stat().st_size==m['stored_bytes']
    if 'timing' in c:
        t=c['timing'];n=len(t['labels'])
        assert len(t['samples_seconds'])==n and all(len(s)==5 and min(s)>0 for s in t['samples_seconds'])
        assert t['median_seconds']==[statistics.median(s) for s in t['samples_seconds']]
        assert t['orders']==[list(range(n))[i%n:]+list(range(n))[:i%n] for i in range(5)]
        timings[c['name']]=dict(zip(t['labels'],[s*1000 for s in t['median_seconds']]))
if r['passed']:
    assert exitcode==0 and len(r['cases'])==10 and comparisons==40 and numeric_equal
    assert r['inputs_and_lut_unchanged']
else:
    assert exitcode!=0
summary=dict(audit_passed=True,experiment_passed=r['passed'],report_sha256=sha(path),auditor_sha256=sha(__file__),
    lease_sha256=sha(lease),returncode=exitcode,comparisons=comparisons,unique_arrays_reread=len(verified),
    stored_bytes_reread=sum(verified.values()),timing_median_ms=timings,scope=__doc__,complete_migration=False)
target.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2))
