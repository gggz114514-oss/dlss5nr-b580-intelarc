"""Reread saved full history-warp arrays and verify completed guard-test evidence."""
import hashlib,json
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays
DREF=Path('D:/Codex-NR-Experiments/nr-b580/reference')
out=DREF/'experimental/graph-history-warp-v2';path=out/'validation.json';dest=out/'saved-audit-v1.json'
assert not dest.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
r=js(path);lease=out.with_suffix('.log.lease.json')
assert r['passed'] and not r['complete_migration'] and js(lease)['returncode']==0
assert all(sha(p)==h for p,h in r['sources'].items())
flags=['recovery_bytes_equal','table_replacement_rejected_before_replay','closed_session_rejected',
       'held_outputs_preserved','caller_mutation_does_not_touch_internal_output','caller_inputs_unchanged']
assert all(r[k] for k in flags)
assert r['invalid_domain_errors']==['Input outside the validated native reciprocal interval']*2
assert len(r['cases'])==4
verified={}
def load(meta):
    a=arrays.load(meta);assert np.isfinite(a).all();verified[meta['path']]=meta['stored_bytes'];return a
assert load(r['history']).shape==(256,256,3)
for row in r['cases']:
    assert row['byte_equal'] and load(row['motion']).shape==(256,256,2)
    numerator,reciprocal=map(load,row['expected'])
    assert numerator.shape==(256,256,3) and reciprocal.shape==(256,256)
    assert numerator.dtype==reciprocal.dtype==np.dtype('f4')
audit=dict(passed=True,scope=__doc__,report_sha256=sha(path),auditor_sha256=sha(__file__),lease_sha256=sha(lease),
           process_exit_code=0,complete_arrays_reread=len(verified),stored_bytes=sum(verified.values()),
           runtime_guard_checks=flags,complete_migration=False)
dest.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8',newline='\n')
print(json.dumps(audit),flush=True)
