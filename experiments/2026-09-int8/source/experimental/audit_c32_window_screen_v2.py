"""Audit the bounded first-block compiler screen, including an intentional stop.

All references and IR artifacts are reread. This does not replay the GPU model
on CPU and does not turn an interrupted broad experiment into a passing one.
"""
import argparse,hashlib,json,statistics
from pathlib import Path
import compressed_arrays_v1 as arrays
p=argparse.ArgumentParser();p.add_argument('folder');args=p.parse_args()
D=Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental');OUT=(D/args.folder).resolve()
assert OUT.parent==D.resolve()
path=OUT/'validation.json';target=OUT/'screen-audit-v1.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
r=js(path);lease=OUT.with_suffix('.log.lease.json');receipt=js(lease)
assert all(sha(p)==h for p,h in r['sources'].items())
assert not r['complete_migration'];verified={};comparisons=0
prior=js(D/'c32-window-blocks-v2/validation.json')
for i,c in enumerate(r['cases']):
    assert not c['synthetic'] and c['name']==prior['cases'][i]['name'] and c['arrays']==prior['cases'][i]['arrays']
    values={}
    for n,m in c['arrays'].items():
        values[n]=arrays.load(m);verified[m['path']]=m['stored_bytes']
    for candidate in c['candidates']:
        assert candidate['byte_equal'] and candidate['different_half_words']==0 and candidate['max_abs']==0
        assert candidate['output_sha256']==hashlib.sha256(values['output'].tobytes()).hexdigest()
        parts=candidate['ir'] if isinstance(candidate['ir'],list) else [candidate['ir']]
        for part in parts:
            for m in part.values():assert sha(m['path'])==m['sha256'] and Path(m['path']).stat().st_size==m['stored_bytes']
        comparisons+=1
    if 'timing' in c:
        t=c['timing'];n=len(t['labels'])
        assert len(t['samples_seconds'])==n and all(len(s)==5 and min(s)>0 for s in t['samples_seconds'])
        assert t['median_seconds']==[statistics.median(s) for s in t['samples_seconds']]
        assert t['orders']==[list(range(n))[i%n:]+list(range(n))[:i%n] for i in range(5)]
if receipt.get('reason')=='stop':
    assert args.folder=='c32-window-fusion-v5' and not r['passed']
    assert comparisons==4 and 'timing' in r['cases'][0]
    t=r['cases'][0]['timing'];assert min(t['median_seconds'][1:])>2*t['median_seconds'][0]
elif r['passed']:
    assert receipt['returncode']==0 and comparisons==2 and len(r['cases'])==1
    assert r['inputs_and_lut_unchanged']
else:assert receipt['returncode']!=0
summary=dict(audit_passed=True,experiment_passed=r['passed'],scope=__doc__,report_sha256=sha(path),
    auditor_sha256=sha(__file__),lease_sha256=sha(lease),completed_comparisons=comparisons,
    unique_arrays_reread=len(verified),stored_bytes_reread=sum(verified.values()),
    intentional_stop_after_regression=receipt.get('reason')=='stop',
    first_block_timing=r['cases'][0].get('timing'),complete_migration=False)
target.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='first_block_timing'},indent=2))
