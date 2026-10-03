"""Authenticate the non-timing inventory and full saved stage boundaries.

Triton counts are Python launch requests from an eager equivalent of the body.
ATen counts include views/allocation and are not native kernel counts. This CPU
audit rereads saved arrays and receipts; it does not independently run NR math.
"""
from collections import Counter
import hashlib,json
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays

D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
out=D/'experimental/body-execution-inventory-v1'
p=out/'validation.json';target=out/'saved-audit-v1.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(p)=='a86af390cdd581486c85e4385669ce1ba7572ec2336de679db47abe5f773cef8'
r=json.loads(p.read_text(encoding='utf-8'));lp=out.with_suffix('.log.lease.json');lease=json.loads(lp.read_text(encoding='utf-8'))
assert r['passed'] and not r['complete_migration'] and lease['returncode']==0
assert r['all_instrumented_stage_outputs_match'] and r['assembled_byte_equal_prior']
assert r['stage_inputs_unchanged'] and r['history_unchanged'] and r['lut_unchanged']
assert r['vit_tokens']==64 and r['fused_history_builds']==1
assert all(sha(p)==h for p,h in r['sources'].items())
baseline=D/'experimental/current-body-stages-v2/validation.json'
assert sha(baseline)=='144d6f4aa8a28f37a4870af671821a3a30ab22b23614eea23f442c1e3aa1c448'
old=json.loads(baseline.read_text(encoding='utf-8'))
assert len(r['stages'])==len(old['stages'])==13
seen={};totals={k:Counter() for k in ('triton','aten')}
def reread(meta):
    if meta is not None and meta['path'] not in seen:
        a=arrays.load(meta);assert np.isfinite(a).all()
        seen[meta['path']]=meta['stored_bytes']
for row,previous in zip(r['stages'],old['stages']):
    assert row['name']==previous['name'] and row['instrumented_byte_equal']
    assert row['output_sha256']==previous['output_sha256']==[m['raw_sha256'] for m in row['outputs']]
    for meta in row['inputs']+row['outputs']:reread(meta)
    events=row['execution']['events'];summary=row['execution']['summary']
    for kind in totals:
        counts=Counter(e['name'] for e in events if e['kind']==kind)
        assert summary[kind]==dict(count=sum(counts.values()),by_name=dict(counts.most_common()))
        totals[kind].update(counts)
    for e in events:
        assert e['kind'] in totals
        if e['kind']=='triton':
            assert 1<=len(e['grid'])<=3 and min(e['grid'])>0
            tensors=e['arguments']
        else:tensors=e['inputs']+e['outputs']
        for t in tensors:assert len(t['shape'])==len(t['stride']) and t['logical_bytes']>=0
    assert summary['aten_output_logical_bytes']==sum(t['logical_bytes'] for e in events if e['kind']=='aten' for t in e['outputs'])
    assert summary['aten_argument_alias_output_logical_bytes']==sum(t['logical_bytes'] for e in events if e['kind']=='aten' for t in e['outputs'] if t['aliases_argument_storage'])
assert r['execution_totals']=={k:dict(count=sum(c.values()),by_name=dict(c.most_common())) for k,c in totals.items()}
for m in r['body_inputs'].values():reread(m)
reread(r['expected_output'])
assert r['expected_output']['raw_sha256']==r['stages'][-1]['output_sha256'][0]
audit=dict(passed=True,complete_migration=False,report_sha256=sha(p),auditor_sha256=sha(__file__),
    lease_sha256=sha(lp),baseline_stage_report_sha256=sha(baseline),sources_authenticated=len(r['sources']),
    unique_arrays_reread=len(seen),stored_bytes_reread=sum(seen.values()),execution_totals=r['execution_totals'],
    timing_measured=False,native_gpu_kernel_count_claim=False,limitation=__doc__)
target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit,indent=2),flush=True)
