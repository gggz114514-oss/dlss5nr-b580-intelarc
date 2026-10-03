"""Independently reread complete paired outputs and native RGB, including process exit."""
import argparse, hashlib, json, math, statistics
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as compressed

HERE = Path(__file__).resolve().parent
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
p = argparse.ArgumentParser()
p.add_argument('--dimension', required=True, choices=['864x480', '1920x1080'])
p.add_argument('--mode', required=True, choices=['baseline', 'fp16_xmx', 'int8_fused_v2'])
p.add_argument('--version', required=True, type=int)
a = p.parse_args()
out = DREF / f'results/graph-native-{a.dimension}-{a.mode}-v{a.version}'
path = out / 'validation.json'
audit_path = out / 'saved-audit-v1.json'
assert not audit_path.exists()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
js = lambda path: json.loads(Path(path).read_text(encoding='utf-8-sig'))
r = js(path)
assert r['passed'] and not r['complete_migration']
lease_path = Path(str(out) + '.log.lease.json')
lease = js(lease_path)
assert lease['returncode'] == 0, lease
for name, digest in r['sources'].items():
    assert sha(name) == digest
w, h = map(int, a.dimension.split('x'))
verified = {}

def read(meta):
    if meta['format'] == 'npy+zlib':
        value = compressed.load(meta)
        verified[meta['path']] = meta['stored_bytes']
    else:
        assert meta['format'] == 'native-rgba32f-select-rgb-to-f16'
        assert sha(meta['path']) == meta['sha256']
        value = np.fromfile(meta['path'], '<f4').reshape(h, w, 4)[..., :3].astype('<f2')
    assert value.shape == (h, w, 3) and value.dtype == np.dtype('<f2')
    assert hashlib.sha256(value.tobytes()).hexdigest() == meta['raw_sha256']
    return value

for mode, rows in r['runs'].items():
    assert len(rows) == 26
    for index, row in enumerate(rows):
        i = index % 13
        assert row['frame'] == i and row['round'] == index // 13
        assert row['reset'] == (i in (0, 12))
        assert row['next_seed'] == (1 if row['reset'] else i + 1)
        actual = read(row['output'])
        # Read native f32 bytes directly; don't infer native equality from a half hash.
        meta = row['native']
        assert sha(meta['path']) == meta['sha256']
        native = np.fromfile(meta['path'], '<f4').reshape(h, w, 4)[..., :3].copy()
        actual32 = actual.astype('<f4')
        assert np.isfinite(actual).all()
        assert hashlib.sha256(actual32.tobytes()).hexdigest() == row['rgb32_sha256']
        equal = actual32.tobytes() == native.tobytes()
        assert equal == row['native_byte_equal']
        if a.mode == 'baseline':
            assert equal
        peer = read(r['runs']['graph' if mode == 'eager' else 'eager'][index]['output'])
        first_round = read(rows[i]['output'])
        assert actual.tobytes() == peer.tobytes() == first_round.tobytes()
        if i == 12:
            assert actual.tobytes() == read(rows[0]['output']).tobytes()
        mse = float(np.square(actual32.astype('f8') - native.astype('f8')).mean())
        assert mse == row['mse']
        assert (None if mse == 0 else -10 * math.log10(mse)) == row['psnr_db']
        assert row['pair_and_repeat_byte_equal'] and row['private_byte_equal']
means = {mode: statistics.mean(row['seconds'] for row in rows) for mode, rows in r['runs'].items()}
assert means == r['matching_mean_seconds']
assert means['eager'] / means['graph'] == r['graph_speedup']
audit = dict(passed=True, report_sha256=sha(path), auditor_sha256=sha(Path(__file__)),
             process_exit_code=0, lease_sha256=sha(lease_path), full_output_comparisons=52,
             all_pairs_and_repeats_byte_equal=True, all_native_byte_equal=r['all_native_byte_equal'],
             matching_mean_seconds=means, graph_speedup=r['graph_speedup'],
             unique_saved_candidate_bytes=sum(verified.values()), unique_saved_candidate_arrays=len(verified),
             private_state_scope='Runtime checked private bytes against returned RGB and stored the same complete artifact; independent audit rereads that artifact, not device memory.',
             complete_migration=False)
audit_path.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(audit), flush=True)
