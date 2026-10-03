"""Preserve the failed V1 finalization without claiming a completed capture."""
import hashlib,json
from pathlib import Path

here=Path(__file__).resolve().parent
out=Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/current-dense-operands-v1')
target=out/'saved-audit-v1.json';assert not target.exists()
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
lease=out.with_suffix('.log.lease.json');log=out.with_suffix('.log')
record=json.loads(lease.read_text(encoding='utf-8'))
assert record['returncode']==1 and record['finished_unix']>=record['started_unix']
assert not (out/'validation.json').exists()
assert "AttributeError: 'str' object has no attribute 'open'" in log.read_text(encoding='utf-8')
audit=dict(passed=True,experiment_passed=False,returncode=1,validation_report_written=False,
    lease_sha256=sha(lease),log_sha256=sha(log),auditor_sha256=sha(__file__),
    source_sha256=sha(here/'capture_current_dense_operands_v1.py'),
    diagnosis='The Path-only final source hash helper received a string. V2 converts to Path and persists a failed report if finalization raises.',
    complete_migration=False)
target.write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit,indent=2))
