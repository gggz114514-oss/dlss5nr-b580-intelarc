"""Authenticate boundary GPU receipts and reread all complete saved tensors."""
import hashlib,json
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays
OUT=Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/k8-tiled-boundaries-v1')
p=OUT/'validation.json';target=OUT/'saved-audit-v1.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(p)=='081b26157a22ce05e13da54920275f3430a6b835fc8b9a0eb11317df9fb042e2'
r=js(p);lease=OUT.with_suffix('.log.lease.json')
assert r['passed'] and not r['complete_migration'] and js(lease)['returncode']==0
assert all(sha(p)==h for p,h in r['sources'].items())
assert r['weights_unchanged'] and len(r['cases'])==r['full_outputs_compared']==20
assert r['finite_half_input_encodings_per_weight']==63488
verified={};full=fallbacks=0;nonfinite=0
finite=np.arange(65536,dtype='u2');finite=finite[(finite&0x7c00)!=0x7c00]
for row in r['cases']:
    assert row['byte_equal']
    if row.get('fallback'):
        assert row['name'] in ('initialized','unowned_weight','baseline_mode');fallbacks+=1;continue
    full+=1;assert row['different_half_words']==0
    values={}
    for key in ('input','weight_meta','expected'):
        m=row[key];values[key]=arrays.load(m);verified[m['path']]=m['stored_bytes']
    assert np.isfinite(values['input']).all() and np.isfinite(values['weight_meta']).all()
    assert row['output_raw_sha256']==row['expected']['raw_sha256']
    assert row['nonfinite_output_words']==np.count_nonzero(~np.isfinite(values['expected']))
    nonfinite+=row['nonfinite_output_words']
    if row['name']=='all_finite_half_bits':
        assert np.array_equal(np.sort(values['input'].view('u2').reshape(-1)),finite)
assert full==14 and fallbacks==6
a=dict(passed=True,report_sha256=sha(p),auditor_sha256=sha(__file__),lease_sha256=sha(lease),
    full_output_receipts=20,complete_saved_comparisons=14,fallback_receipts=6,
    nonfinite_output_words_in_matching_receipts=nonfinite,unique_arrays_reread=len(verified),
    stored_bytes_reread=sum(verified.values()),limitation='CPU artifact audit, not independent K8 arithmetic execution.',complete_migration=False)
target.write_text(json.dumps(a,indent=2)+'\n',encoding='utf-8');print(json.dumps(a,indent=2))
