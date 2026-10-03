"""Reread complete reused arrays and the process record for window-layout long replay."""
import argparse, hashlib, json
from pathlib import Path
import compressed_arrays_v1 as arrays

HERE = Path(__file__).resolve().parent
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['fp16_xmx', 'int8_fused_v2'], required=True)
a = p.parse_args()
out = DREF / f'results/{a.mode}-fused-pipeline-long-480-v3'
path = out / 'validation.json'
audit_path = out / 'saved-audit-v1.json'
assert not audit_path.exists()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
js = lambda path: json.loads(Path(path).read_text(encoding='utf-8-sig'))
r = js(path)
assert r['passed'] and r['frames_completed'] == len(r['frames']) == 243
assert not r['native_capture_for_this_sequence'] and not r['complete_migration']
lease_path = out.with_suffix('.log.lease.json')
assert js(lease_path)['returncode'] == 0
for name, digest in r['sources'].items():
    assert sha(name) == digest
old = js(DREF / 'results/long-precision-480-v1/validation.json')
old_mode = 'fp16_xmx' if a.mode == 'fp16_xmx' else 'int8_cached'
review_path = DREF / 'results/long-precision-480-v1/user-review-v1.json'
review = js(review_path)
assert review[old_mode + '_visually_accepted']
verified = {}
for i, row in enumerate(r['frames']):
    expected_frame = old['frames'][i]
    expected = expected_frame['runs'][old_mode]
    assert row['frame'] == i and row['reset'] == (i == 0) and row['next_seed'] == i + 1
    assert row['byte_equal_approved'] and row['private_byte_equal_output']
    assert row['input_rgb8_sha256'] == expected_frame['input_rgb8_sha256']
    value = arrays.load(row['output'])
    target = arrays.load(expected['output'])
    assert value.tobytes() == target.tobytes()
    flow = arrays.load(row['motion'])
    assert flow.tobytes() == arrays.load(expected_frame['motion']).tobytes()
    assert row['effective_dispatch'] == expected['dispatches']
    for meta in (row['output'], row['motion']):
        verified[meta['path']] = meta['stored_bytes']
assert len(r['graphs']) == 2 and len({g['pool'] for g in r['graphs']}) == 1
assert all(g['persistent_inputs_and_output_verified_outside_pool'] for g in r['graphs'])
audit = dict(passed=True, report_sha256=sha(path), auditor_sha256=sha(Path(__file__)),
             lease_sha256=sha(lease_path), process_exit_code=0, full_frames_verified=243,
             complete_output_and_motion_arrays_verified=486, approved_bytes_reused=True,
             user_review_sha256=sha(review_path), user_quote=review['user_response'],
             unique_reused_bytes=sum(verified.values()), new_output_bytes=0,
             private_scope='Runtime compared private bytes against the complete output before sharing its artifact.',
             complete_migration=False)
audit_path.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(audit), flush=True)
