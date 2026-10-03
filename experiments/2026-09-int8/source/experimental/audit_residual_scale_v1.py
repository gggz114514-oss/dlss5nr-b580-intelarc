"""CPU reread of all complete residual outputs, low NR frames and process evidence."""
import argparse, hashlib, json, statistics
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['fp16_xmx', 'int8_fused_v2'], required=True)
p.add_argument('--size', type=int, choices=[256, 512], required=True)
a = p.parse_args()
out = DREF / f'results/residual-scale-{a.mode}-{a.size}-v1'
path, audit_path = out / 'validation.json', out / 'saved-audit-v1.json'
assert not audit_path.exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
r = js(path)
lease = out.with_suffix('.log.lease.json')
assert r['passed'] and js(lease)['returncode'] == 0
assert r['complete_frames'] == len(r['runs']) == 26 and not r['new_quality_approved'] and not r['complete_migration']
assert all(sha(p) == h for p, h in r['sources'].items())
assert r['primitive_checks']['zero_residual_identity_bytes'] and r['primitive_checks']['constant_motion_units_scaled_exact_half']
assert r['caller_inputs_and_private_preserved'] and r['graph_eager_low_nr_reset_bytes_equal']
verified = {}
for row in r['runs']:
    i = row['frame']
    for key, shape, dtype in [('output', (1080,1920,3), 'f4'), ('low_nr', (a.size,a.size,3), 'f2')]:
        meta = row[key]
        value = arrays.load(meta)
        assert value.shape == shape and value.dtype == np.dtype(dtype) and np.isfinite(value).all()
        assert value.tobytes() == arrays.load(r['runs'][i][key]).tobytes()
        verified[meta['path']] = meta['stored_bytes']
    assert row['private_byte_equal_low_nr'] and row['next_seed'] == (1 if row['reset'] else i + 1)
    if row['round']:
        assert row['repeat_bytes_equal']
    assert row['runtime_dispatch']['xpu_graph_replay'] == 1
for key in ('first_canvas','first_low_motion'):
    arrays.load(r['primitive_checks'][key])
means = {key:statistics.mean(row[key] for row in r['runs']) for key in ('seconds','prepare_seconds','nr_seconds','composite_seconds')}
assert means == r['mean_seconds']
audit = dict(passed=True, report_sha256=sha(path), auditor_sha256=sha(Path(__file__)), lease_sha256=sha(lease),
             complete_full_resolution_outputs_reread=26, complete_low_nr_outputs_reread=26,
             unique_output_bytes=sum(verified.values()), process_exit_code=0, new_quality_approved=False,
             timing_scope=r['timing_scope'], complete_migration=False)
audit_path.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(audit), flush=True)
