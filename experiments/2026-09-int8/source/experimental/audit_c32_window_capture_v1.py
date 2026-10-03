"""Reread complete saved C32 operands; this is not an independent CPU model run."""
import hashlib,json
from pathlib import Path
import compressed_arrays_v1 as arrays
D=Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT=D/'experimental/c32-window-blocks-v2';p=OUT/'validation.json';target=OUT/'saved-audit-v1.json'
assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(p)=='b99326949733fe839431341cc743ea5903abda8f95ef3ca795b23ed0f2ca08cd'
r=js(p);lease=OUT.with_suffix('.log.lease.json')
assert r['passed'] and not r['complete_migration'] and js(lease)['returncode']==0
assert all(sha(p)==h for p,h in r['sources'].items())
assert r['all_modules_captured'] and r['entire_body_byte_equal'] and r['inputs_and_lut_unchanged']
assert r['separate_contract_modules']==['decoder.3.0.body','post.body']
assert [c['name'] for c in r['cases']]==['encoder.0.0','encoder.0.1','encoder.0.2','encoder.0.3','decoder.3.1','decoder.3.2','decoder.3.3']
assert [c['shift'] for c in r['cases']]==[[0,0],[4,4],[0,4],[4,0],[4,4],[0,4],[4,0]]
verified={};total=0
def read(m):
    a=arrays.load(m);verified[m['path']]=m['stored_bytes'];return a
for c in r['cases']:
    assert c['shape']==[160,160,32]
    values={n:read(m) for n,m in c['arrays'].items()}
    assert values['features'].shape==values['output'].shape==(160,160,32)
    assert set(values['order'].tolist())==set(range(64))
    assert c['dispatch']==dict(dense=4,batched=2,fp8=7,cubic_fp8=1,attention_normalize_c32=2,attention_exp_swin=1,attention_weights=1)
    total+=sum(a.nbytes for a in values.values())
assert total==r['raw_bytes']<=r['maximum_raw_bytes']==128*2**20
prior=js(D/'experimental/small-branched-mlp-operands-v1/validation.json')
assert r['dispatch']==prior['dispatch'] and r['body_inputs']==prior['body_inputs'] and r['expected_output']==prior['expected_output']
for m in [*r['body_inputs'].values(),r['expected_output']]:read(m)
summary=dict(passed=True,report_sha256=sha(p),auditor_sha256=sha(__file__),lease_sha256=sha(lease),
    blocks=7,captured_raw_bytes=total,unique_arrays_reread=len(verified),stored_bytes_reread=sum(verified.values()),scope=__doc__)
target.write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2))
