"""Authenticate static body receipts; no independent CPU model execution.

The frozen benchmark's complete_replay_outputs_verified=100 counts replays.
It actually reads the complete owned output after each ten-replay group:
100 completed graph replays and 10 full output readbacks, plus 3 eager checks.
"""
import hashlib,json,statistics
from pathlib import Path
import compressed_arrays_v1 as arrays
D=Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental')
OUT=D/'c32-tail-body38-v1'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
p=OUT/'validation.json';target=OUT/'saved-audit-v1.json';assert not target.exists()
assert sha(p)=='43f28e3a5005c4d6ef044b97a1143648b4a498403cedc41d018f3a6c59243faf'
r=js(p);lease=OUT.with_suffix('.log.lease.json')
assert js(lease)['returncode']==0 and r['passed'] and not r['complete_migration']
assert all(sha(p)==h for p,h in r['sources'].items())
values=[arrays.load(m) for m in r['body_inputs'].values()]
expected=arrays.load(r['expected_output']);digest=hashlib.sha256(expected.tobytes()).hexdigest()
assert [c['name'] for c in r['body']]==['previous_cold_with_progress','previous','tail']
assert all(c['byte_equal'] and c['output_sha256']==digest for c in r['body'])
assert r['tail_modules']==['encoder.0.0','encoder.0.1','encoder.0.2','encoder.0.3','decoder.3.1','decoder.3.2','decoder.3.3']
assert all(r[k] for k in ['inputs_and_lut_unchanged','adapter_ownership_guards','constant_replacement_rejected','body_did_not_advance_history'])
t=r['timing'];assert t['labels']==['previous','tail']
assert len(t['samples_seconds'])==2 and all(len(s)==5 and min(s)>0 for s in t['samples_seconds'])
assert t['median_seconds']==[statistics.median(s) for s in t['samples_seconds']]
assert t['orders']==[[0,1],[1,0],[0,1],[1,0],[0,1]]
assert t['complete_replay_outputs_verified']==100
summary=dict(audit_passed=True,report_sha256=sha(p),auditor_sha256=sha(__file__),lease_sha256=sha(lease),
    returncode=0,raw_array_bytes_reread=sum(v.nbytes for v in values)+expected.nbytes,
    output_sha256=digest,eager_complete_output_checks=3,completed_graph_replays=100,
    complete_output_readbacks_after_replay_groups=10,benchmark_count_field_clarification=__doc__,
    body_median_ms=[s*1000 for s in t['median_seconds']],complete_migration=False)
target.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,indent=2))
