"""Reread every small B580 result and compare its full RGB bytes to native."""
import hashlib
import json
from pathlib import Path

import numpy as np
import compressed_arrays_v1 as arrays

HERE = Path(__file__).resolve().parent
R = Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT = R / 'results/small-geometry-exact-v1'
TARGET = OUT / 'saved-audit-v1.json'
assert not TARGET.exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
path = OUT / 'validation.json'
report = js(path)
native_path = R / 'experimental/small-geometry-v1/native-audit-v1.json'
native = js(native_path)
assert report['passed'] and not report['complete_migration']
assert report['frames_verified'] == len(report['frames']) == 12
assert report['native_audit_sha256'] == sha(native_path)
assert js(OUT.with_suffix('.log.lease.json'))['returncode'] == 0
assert all(sha(p) == h for p, h in report['sources'].items())
assert all(sha(p) == h for p, h in native['sources'].items())
assert all(sha(p) == h for p, h in native['files'].items())
for key in ('caller_mutation_preserves_history', 'held_outputs_unchanged',
            'unsupported_geometry_preserves_state', 'reset_clears_history', 'all_bytes_equal_native'):
    assert report[key]
verified = {}
for case in native['cases']:
    w, h = map(int, case['dimension'].split('x'))
    rows = [v for v in report['frames'] if v['dimension'] == case['dimension']]
    assert len(rows) == 4
    for i, row in enumerate(rows):
        assert row['frame'] == i and row['reset'] == (i in (0, 3))
        assert row['next_seed'] == (1 if row['reset'] else i + 1)
        assert row['byte_equal_native'] and row['max_abs'] == row['half_components_different'] == 0
        assert row['private_equals_output'] and row['dispatch']['backend'] == 'triton'
        meta = row['output']
        assert sha(meta['path']) == meta['sha256']
        actual = arrays.load(meta)
        assert actual.shape == (h, w, 3) and actual.dtype == np.dtype('<f2') and np.isfinite(actual).all()
        expected_path = Path(case['native_dir']) / f'frame{i:02d}.png_output.rgba32f.bin'
        expected = np.fromfile(expected_path, '<f4').reshape(h, w, 4)[..., :3].copy()
        assert actual.astype('f4').tobytes() == expected.tobytes()
        verified[meta['path']] = Path(meta['path']).stat().st_size
        verified[str(expected_path)] = expected_path.stat().st_size
saved = dict(passed=True, auditor_sha256=sha(Path(__file__)), report_sha256=sha(path),
             native_audit_sha256=sha(native_path), lease_sha256=sha(OUT.with_suffix('.log.lease.json')),
             complete_frames_compared=12, all_rgb_bytes_equal_native=True,
             unique_arrays_reread=len(verified), unique_array_bytes=sum(verified.values()),
             complete_migration=False,
             limitation='Native/B580 output bytes independently reread; does not claim new dimensions have less body work or arbitrary controls are validated.')
TARGET.write_text(json.dumps(saved, indent=2) + '\n', encoding='utf-8')
print(json.dumps(saved, indent=2))
