"""CPU reread of complete saved operands/results, source hashes and closed jobs.

GPU candidate equality was checked at execution; shared expected artifacts do
not constitute an independent CPU rerun of those candidate kernels.
"""
import argparse, hashlib, json, statistics
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays

DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
p = argparse.ArgumentParser()
p.add_argument('folder')
out = (DREF / p.parse_args().folder).resolve()
assert out.is_relative_to(DREF.resolve())
path, audit_path = out / 'validation.json', out / 'saved-audit-v1.json'
assert not audit_path.exists()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
r = js(path)
lease = out.with_suffix('.log.lease.json')
assert r['passed'] and not r['complete_migration'] and js(lease)['returncode'] == 0
assert all(sha(p) == h for p, h in r['sources'].items())
verified = {}


def load(meta):
    assert sha(meta['path']) == meta['sha256']
    value = arrays.load(meta) if meta.get('format') == 'npy+zlib' else np.load(meta['path'], allow_pickle=False)
    assert np.isfinite(value).all()
    verified[meta['path']] = Path(meta['path']).stat().st_size
    return value


if isinstance(r.get('cases'), dict):
    assert r['full_output_byte_equal_prior'] and r['private_equal'] and r['next_seed'] == 1
    assert sum(row['count'] for row in r['cases'].values()) == r['total_mlp_calls']
    total_bytes = 0
    for row in r['cases'].values():
        if not row['captured']:
            continue
        a = {name: load(meta) for name, meta in row['arrays'].items()}
        assert all(v.dtype == np.dtype('f2') for v in a.values())
        c = row['channels']
        assert a['features'].shape == a['output'].shape == tuple(row['shape'])
        assert a['features'].shape[-1] == c and a['skip_scale'].shape == (c,)
        if c == 32:
            assert a['expansion'].shape == (32, 128) and a['contraction'].shape == (128, 32)
        else:
            assert a['expand'].shape == (c // 32, c, 128)
            assert a['reduce'].shape == (c // 32, 128, 32)
            assert a['project'].shape == (c // 32, 32, c)
        total_bytes += sum(v.nbytes for v in a.values())
    assert total_bytes == r['raw_bytes']
    old = js(DREF / 'results/fast-precision-864x480-v1/validation.json')['runs']['fp16_xmx'][0]
    assert load(r['full_output']).astype('f2').tobytes() == load(old['actual']).astype('f2').tobytes()
    assert r['dispatch'] == old['dispatches']
elif isinstance(r.get('cases'), list):
    for row in r['cases']:
        a = {name: load(meta) for name, meta in row['operands'].items()}
        target = load(row['expected'])
        assert a['features'].shape == target.shape == (row['rows'], row['channels'])
        assert row['operands_unchanged']
        medians = {name: statistics.median(v) for name, v in row['samples_seconds'].items()}
        assert medians == row['median_seconds']
        assert row['speedup'] == {name: medians['original'] / v for name, v in medians.items()}
        for item in row['candidates']:
            ir = item['ttgir']
            assert sha(ir['path']) == ir['sha256'] and 'ttig.dpas' in Path(ir['path']).read_text()
            if item['name'] in medians:
                assert item['byte_equal'] and item['changed_half_components'] == 0 and item['max_abs'] == 0
            else:
                assert not item['byte_equal']
                assert load(item['actual']).tobytes() != target.tobytes()
elif 'full_outputs_verified' in r:
    old = js(DREF / 'results/fast-precision-864x480-v1/validation.json')['runs']['fp16_xmx']
    assert r['full_outputs_verified'] == 156 and len(r['runs']) == 4
    assert r['held_outputs_survive_replay'] and r['caller_ownership_guards_passed']
    pools = set()
    for name, rows in r['runs'].items():
        assert len(rows) == 39 and len(r['graphs'][name]) == 2
        pools.update(entry['pool'] for entry in r['graphs'][name])
        for ordinal, row in enumerate(rows):
            assert (row['round'], row['frame']) == divmod(ordinal, 13)
            assert load(row['output']).astype('f2').tobytes() == load(old[row['frame']]['actual']).astype('f2').tobytes()
            assert row['byte_equal_prior'] and row['private_byte_equal_output']
            assert row['effective_dispatch'] == old[row['frame']]['dispatches']
            assert row['next_seed'] == (1 if row['reset'] else row['frame'] + 1)
        assert statistics.mean(row['seconds'] for row in rows) == r['matching_mean_seconds'][name]
    assert len(pools) == 1 and len(r['progress_fallback']) == 4
    for row in r['progress_fallback']:
        assert row['byte_equal']
        assert bool(row['fused_pairs']) == (row['name'] in ('pairs', 'both'))
        assert bool(row['fused_c32']) == (row['name'] in ('c32', 'both'))
else:
    raise ValueError('Unknown report type')

audit = dict(scope=__doc__, passed=True, report_sha256=sha(path), auditor_sha256=sha(__file__),
             lease_sha256=sha(lease), process_exit_code=0, unique_complete_arrays_reread=len(verified),
             unique_array_bytes=sum(verified.values()), complete_migration=False)
audit_path.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(audit), flush=True)
