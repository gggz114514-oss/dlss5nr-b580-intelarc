"""Reread every saved CPU DIS diagnostic flow and authenticate its measurements."""
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
import compressed_arrays_v1 as arrays

OUT = Path('D:/Codex-NR-Experiments/nr-b580/reference/experimental/dis-budget-v1')
path = OUT / 'validation.json'
audit_path = OUT / 'saved-audit-v1.json'
assert not audit_path.exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
report = json.loads(path.read_text(encoding='utf-8'))
assert report['passed'] and len(report['measurements']) == 198 and len(report['quality']) == 66
assert all(sha(p) == digest for p, digest in report['sources'].items())
for name in report['variants']:
    rows = [r for r in report['measurements'] if r['variant'] == name]
    assert {(r['round'], r['frame']) for r in rows} == {(j, i) for j in range(3) for i in range(1, 12)}
    for row in rows:
        assert abs(row['prepare_seconds']+row['calc_seconds']+row['half_seconds']-row['cpu_path_seconds']) < 1e-9
        assert min(row['prepare_seconds'], row['calc_seconds'], row['half_seconds']) >= 0
    for i in range(1, 12):
        assert len({r['raw_half_sha256'] for r in rows if r['frame'] == i}) == 1
    for field in ('calc_seconds', 'cpu_path_seconds'):
        assert report['summary'][name][field]['mean'] == statistics.mean(r[field] for r in rows)
    if name.startswith('full_'):
        assert all(r['full_size_byte_equal_old'] for r in rows)
artifacts = {}
for row in report['quality']:
    value = arrays.load(row['low_motion'])
    assert value.shape == (256, 256, 2) and value.dtype == np.dtype('f2')
    assert np.isfinite(value).all() and not np.any(value[:56]) and not np.any(value[200:])
    artifacts[row['low_motion']['path']] = row['low_motion']['stored_bytes']
    if row['variant'].startswith('full_'):
        assert row['low_motion_mean_epe'] == 0
for index in range(1, 12):
    rows = [r for r in report['quality'] if r['frame'] == index and r['variant'].startswith('full_')]
    assert len({r['low_motion']['raw_sha256'] for r in rows}) == 1
assert report['semantics_changed'] == ['half_reuse_2t', '512_reuse_2t', '256_reuse_2t']
result = dict(passed=True, report_sha256=sha(path), audit_source_sha256=sha(__file__),
              complete_saved_flows_reread=66, unique_artifact_count=len(artifacts),
              unique_artifact_stored_bytes=sum(artifacts.values()), measurements_checked=198,
              limitation='Saved-array authentication and arithmetic checks, not an independent DIS rerun or NR visual acceptance')
audit_path.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
print(json.dumps(result, indent=2))
