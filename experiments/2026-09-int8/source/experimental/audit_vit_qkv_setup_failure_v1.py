"""Authenticate the V1 harness failure; this does not mark its run successful."""
import hashlib,json
from pathlib import Path

out=Path('D:/Codex-NR-Experiments/nr-b580/reference/results/vit-qkv-residual256-v1')
path=out/'validation.json';target=out/'saved-audit-v1.json'
assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
assert sha(path)=='0d3cc0aa1b82d15555f966d6a98d3c7b78f85f39a4a6e620cbd918144cb0a2ad'
r=js(path);lease=out.with_suffix('.log.lease.json')
assert not r['passed'] and js(lease)['returncode']==1 and not r['complete_migration']
assert 'AttributeError' in r['error'] and 'tobytes' in r['error']
assert len(r['vit_boundary_checks'])==8 and all(c['all_seven_byte_equal'] for c in r['vit_boundary_checks'])
assert len(r['vit_adapter_cases'])==5 and all(c['all_seven_byte_equal'] for c in r['vit_adapter_cases'])
assert all(sha(p)==h for p,h in r['sources'].items())
audit=dict(passed=True,experiment_passed=False,returncode=1,report_sha256=sha(path),
    auditor_sha256=sha(__file__),lease_sha256=sha(lease),
    error=r['error'],diagnosis='Boundary-reference variable overwrote the full-frame reference list; fixed in a separate V2 harness.',
    complete_migration=False)
target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit,indent=2))
