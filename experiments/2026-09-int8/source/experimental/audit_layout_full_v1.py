"""Independently reread complete saved baseline artifacts and paired-run evidence."""
import argparse, hashlib, json, statistics
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays

DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
p = argparse.ArgumentParser()
p.add_argument('--scope', choices=['strided480', 'window480', 'window1080'], required=True)
p.add_argument('--mode', choices=['fp16_xmx', 'int8_fused_v2'])
a = p.parse_args()
if a.scope == 'window1080':
    assert a.mode is not None
    name = f'window-layout-full-1920x1080-{a.mode}-v1'
else:
    assert a.mode is None
    name = 'strided-full-480-v1' if a.scope == 'strided480' else 'window-layout-full-480-v1'
out = DREF / 'results' / name
path, audit_path = out / 'validation.json', out / 'saved-audit-v1.json'
assert not audit_path.exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
r = js(path)
assert r['passed'] and not r['complete_migration']
lease_path = Path(str(out) + '.log.lease.json')
assert js(lease_path)['returncode'] == 0
for source, digest in r['sources'].items():
    assert sha(source) == digest
prior_path = DREF / ('results/fast-precision-864x480-v1/validation.json' if a.scope != 'window1080'
                    else f'results/graph-native-1920x1080-{a.mode}-v4/validation.json')
prior = js(prior_path)

def load(meta):
    if meta.get('format') == 'npy+zlib':
        value = arrays.load(meta)
        assert value.dtype == np.dtype('f2')
        return value
    assert sha(meta['path']) == meta['sha256']
    value = np.load(meta['path'], allow_pickle=False)
    assert hashlib.sha256(value.tobytes()).hexdigest() == meta['raw_sha256']
    assert list(value.shape) == meta['shape'] and value.dtype.str == meta['dtype']
    assert value.astype('f2').astype(value.dtype).tobytes() == value.tobytes()
    return value.astype('f2')

count = 0
for mode, rows in r['runs'].items():
    assert len(rows) == 26
    canonical = prior['runs']['graph'] if a.scope == 'window1080' else prior['runs']['fp16_xmx' if mode.startswith('fp16') else 'int8_dense']
    for index, row in enumerate(rows):
        i = index % 13
        assert row['round'] == index // 13 and row['frame'] == i
        assert row['reset'] == (i in (0, 12)) and row['next_seed'] == (1 if row['reset'] else i + 1)
        actual = load(row['output'])
        old = canonical[i]
        expected = load(old['output'] if a.scope == 'window1080' else old['actual'])
        assert actual.tobytes() == expected.tobytes()
        assert row['byte_equal_prior'] and row['private_byte_equal_output']
        assert row['effective_dispatch'] == old['effective_dispatch' if a.scope == 'window1080' else 'dispatches']
        count += 1
means = {mode: statistics.mean(row['seconds'] for row in rows) for mode, rows in r['runs'].items()}
assert means == r['matching_mean_seconds']
assert r['held_outputs_survive_replay'] and r['caller_ownership_guards_passed']
assert all(row['byte_equal'] and len(row['marks']) == 13 for row in r['progress_fallback'])
audit = dict(passed=True, report_sha256=sha(path), auditor_sha256=sha(Path(__file__)),
             lease_sha256=sha(lease_path), process_exit_code=0, complete_frame_comparisons=count,
             saved_complete_arrays_match_prior=True, matching_mean_seconds=means,
             private_scope='Runtime private-byte checks share the canonical full output artifact; this audit rereads artifacts, not GPU memory.',
             complete_migration=False)
audit_path.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(audit), flush=True)
