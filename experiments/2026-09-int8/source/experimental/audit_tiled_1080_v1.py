"""Independently reread full saved1080 outputs and verify the paired-run records."""
import argparse, hashlib, json
from pathlib import Path
import compressed_arrays_v1 as arrays
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
p = argparse.ArgumentParser()
p.add_argument('--version', choices=['1', '2'], required=True)
a = p.parse_args()
out = DREF / f'results/dense-tiles-full1920x1080-int8_fused_v2-v{a.version}'
path, audit_path = out / 'validation.json', out / 'saved-audit-v1.json'
assert not audit_path.exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
r = js(path)
lease = out.with_suffix('.log.lease.json')
assert r['passed'] and js(lease)['returncode'] == 0 and not r['complete_migration']
assert all(sha(p) == digest for p, digest in r['sources'].items())
old = js(DREF / 'results/graph-native-1920x1080-int8_fused_v2-v4/validation.json')['runs']['graph'][:13]
count = 0
for mode, rows in r['runs'].items():
    assert len(rows) == 26
    for ordinal, row in enumerate(rows):
        i = ordinal % 13
        assert row['frame'] == i and row['round'] == ordinal // 13
        assert arrays.load(row['output']).tobytes() == arrays.load(old[i]['output']).tobytes()
        assert row['effective_dispatch'] == old[i]['effective_dispatch']
        assert row['byte_equal_prior'] and row['private_byte_equal_output']
        assert row['next_seed'] == (1 if row['reset'] else i + 1)
        count += 1
    assert len(r['graphs'][mode]) == 2
    assert all(g['persistent_inputs_and_output_verified_outside_pool'] for g in r['graphs'][mode])
assert len(r['progress_fallback']) == 2 and all(row['byte_equal'] for row in r['progress_fallback'])
assert r['held_outputs_survive_replay'] and r['caller_ownership_guards_passed']
if a.version == '2':
    assert r['all_four_graph_outputs_survive_other_graphs']
    assert len({g['pool'] for rows in r['graphs'].values() for g in rows}) == 1
audit = dict(passed=True, report_sha256=sha(path), auditor_sha256=sha(Path(__file__)),
             lease_sha256=sha(lease), process_exit_code=0, full_outputs_reread=count,
             complete_migration=False)
audit_path.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(audit), flush=True)
