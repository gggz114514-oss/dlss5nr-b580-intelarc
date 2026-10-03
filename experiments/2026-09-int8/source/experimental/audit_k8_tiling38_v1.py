"""Authenticate two complete stage captures and eight exact K8 tile receipts.

Rereads full saved operands/outputs; GPU comparisons are authenticated receipts,
not an independent CPU execution of the tiled K8 kernel or complete model.
"""
import hashlib,json,statistics
from pathlib import Path
import compressed_arrays_v1 as arrays
OUT=Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/k8-tiling38-v1')
p=OUT/'validation.json';target=OUT/'saved-audit-v1.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(p)=='b5fef99364b540e8bd286790167ca011f62df9f291a7e468de2c857e6c5b80c3'
r=js(p);lease=OUT.with_suffix('.log.lease.json')
assert r['passed'] and not r['complete_migration'] and js(lease)['returncode']==0
assert all(sha(p)==h for p,h in r['sources'].items())
assert r['inputs_and_lut_unchanged'] and r['history_not_advanced']
assert [c['name'] for c in r['cases']]==['pre','post']
assert all(s['complete_outputs_byte_equal'] for s in r['stage_checks'])
verified={};timing={}
for c in r['cases']:
    assert len(c['candidates'])==4 and c['arrays']['initial'] is None
    for m in c['arrays'].values():
        if m is not None:arrays.load(m);verified[m['path']]=m['stored_bytes']
    assert all(v['byte_equal'] and v['different_half_words']==0 for v in c['candidates'])
    t=c['timing'];assert len(t['labels'])==5 and all(len(s)==5 and min(s)>0 for s in t['samples_seconds'])
    assert t['median_seconds']==[statistics.median(s) for s in t['samples_seconds']]
    assert t['graph_replays_per_variant']==50 and t['full_output_readbacks_per_variant']==5
    assert t['orders']==[list(range(5))[i:]+list(range(5))[:i] for i in range(5)]
    timing[c['name']]=dict(zip(t['labels'],[s*1000 for s in t['median_seconds']]))
a=dict(passed=True,report_sha256=sha(p),auditor_sha256=sha(__file__),lease_sha256=sha(lease),
    whole_stages_verified=2,tiled_full_output_receipts=8,unique_arrays_reread=len(verified),
    stored_bytes_reread=sum(verified.values()),timing_median_ms=timing,limitation=__doc__,complete_migration=False)
target.write_text(json.dumps(a,indent=2)+'\n',encoding='utf-8');print(json.dumps(a,indent=2))
