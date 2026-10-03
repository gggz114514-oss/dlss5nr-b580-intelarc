"""Authenticate complete history sampler artifacts and GPU comparison receipts.

Audit success only authenticates the record; experiment_passed separately states
whether all GPU checks passed. This is not independent CPU sampler execution.
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
assert not r['complete_migration'] and all(sha(p)==h for p,h in r['sources'].items())
assert (exitcode==0)==r['passed']
verified={};count=0
def read(meta):
    a=arrays.load(meta);verified[meta['path']]=meta['stored_bytes'];return a
if 'reference' in r:
    meta=r['reference']['arrays'];values={n:read(m) for n,m in meta.items()}
    assert [c['name'] for c in r['cases']]==['previous','b32','b64']
    for c in r['cases']:
        assert c['byte_equal']
        for row,name in zip(c['outputs'],('numerator','reciprocal')):
            assert row['byte_equal'] and row['different_float_words']==0 and row['raw_sha256']==meta[name]['raw_sha256'];count+=1
    timings=[r['timing']]
    assert r['inputs_and_table_unchanged']
else:
    assert len(r['cases'])==19 and sum(c['actual'] for c in r['cases'])==11
    for c in r['cases']:
        meta=c['arrays'];values={n:read(m) for n,m in meta.items()}
        assert c['byte_equal'] and c['different_float_words']==[0,0]
        assert c['output_raw_sha256']==[meta[k]['raw_sha256'] for k in ('numerator','reciprocal')];count+=2
    timings=r['timing']
    if r['passed']:
        assert r['complete_cases']==19 and r['fused_builds']==3
        assert all(r[k] for k in ('inputs_and_table_bytes_unchanged','held_outputs_survive_replay','caller_owned_outputs','float_image_fallback'))
        assert [g['kind'] for g in r['guards']]==['nonfinite_motion','nonfinite_motion','table_replacement','table_version','closed']
        assert all(g['both_rejected'] for g in r['guards'])
        assert all(g['no_replay'] for g in r['guards'] if g['kind'].startswith('table_'))
    else:assert 'Invalid domain accepted' in r['error']
for t in timings:
    assert all(len(s)==5 and min(s)>0 for s in t['samples_seconds'])
    assert t['median_seconds']==[statistics.median(s) for s in t['samples_seconds']]
    assert t['adapter_calls_per_variant']==50
a=dict(passed=True,experiment_passed=r['passed'],report_sha256=sha(path),auditor_sha256=sha(__file__),lease_sha256=sha(lease),
    returncode=exitcode,complete_tensor_comparison_receipts=count,unique_arrays_reread=len(verified),stored_bytes_reread=sum(verified.values()),
    timing_median_ms=[dict(labels=t['labels'],values=[s*1000 for s in t['median_seconds']]) for t in timings],scope=__doc__,complete_migration=False)
target.write_text(json.dumps(a,indent=2)+'\n',encoding='utf-8');print(json.dumps(a,indent=2))
