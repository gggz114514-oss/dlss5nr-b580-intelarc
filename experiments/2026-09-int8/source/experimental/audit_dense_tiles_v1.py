"""CPU audit of saved real operands, tile decisions, and complete-model comparisons.

Runtime GPU comparisons prove candidate equality. This auditor independently
rereads the saved full arrays and checks identities, selection math and process
completion; it does not claim to rerun GPU arithmetic on the CPU.
"""
import argparse, hashlib, json, statistics
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays
HERE = Path(__file__).resolve().parent
DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
p = argparse.ArgumentParser()
p.add_argument('--mode', choices=['fp16_xmx', 'int8_fused_v2'], required=True)
p.add_argument('--stage', choices=['capture', 'primitive', 'full'], required=True)
arg = p.parse_args()
folder = {'capture': f'experimental/dense-operands-{arg.mode}-v1',
          'primitive': f'experimental/dense-tiles-{arg.mode}-v1',
          'full': f'results/dense-tiles-full480-{arg.mode}-v1'}[arg.stage]
out = DREF / folder
path, audit_path = out / 'validation.json', out / 'saved-audit-v1.json'
assert not audit_path.exists()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
js = lambda path: json.loads(Path(path).read_text(encoding='utf-8-sig'))
r = js(path)
assert r['passed'] and not r['complete_migration']
lease = out.with_suffix('.log.lease.json')
assert js(lease)['returncode'] == 0
assert all(sha(name) == digest for name, digest in r['sources'].items())
verified = {}

def load(meta):
    value = arrays.load(meta) if meta.get('format') == 'npy+zlib' else np.load(meta['path'], allow_pickle=False)
    assert sha(meta['path']) == meta['sha256']
    verified[meta['path']] = Path(meta['path']).stat().st_size
    return value

if arg.stage == 'capture':
    assert r['byte_equal_prior'] and r['private_equal'] and r['next_seed'] == 1
    assert sum(row['count'] for row in r['cases'].values()) == 1262
    for row in r['cases'].values():
        if row['captured']:
            a, w, ini = (None if row['operands'][name] is None else load(row['operands'][name]) for name in ('a', 'w', 'initial'))
            m, k, n = row['shape']
            assert a.size == m * k and a.shape[-1] == k and w.shape == (k, n)
            assert ini is None or (ini.shape == (*a.shape[:-1], n))
            assert all(value is None or (value.dtype == np.dtype('f2') and np.isfinite(value).all()) for value in (a, w, ini))
    assert np.isfinite(load(r['output'])).all()
elif arg.stage == 'primitive':
    assert len(r['cases']) == 24
    for row in r['cases']:
        for meta in row['operands'].values():
            if meta is not None:
                load(meta)
        assert np.isfinite(load(row['expected'])).all()
        medians = {name: statistics.median(samples) for name, samples in row['samples_seconds'].items()}
        assert medians == row['median_seconds']
        base = row['baseline']
        eligible = [name for name in medians if name != base and medians[name] < medians[base] * .95
                    and all(b / c > 1.02 for b, c in zip(row['samples_seconds'][base], row['samples_seconds'][name]))]
        best = min(eligible, key=medians.get) if eligible else base
        assert best == row['selected'] and row['operands_unchanged']
        for item in row['candidates']:
            ir = item['ttgir']
            assert sha(ir['path']) == ir['sha256'] and 'ttig.dpas' in Path(ir['path']).read_text()
            if item['name'] in medians:
                assert item['byte_equal'] and item['graph_byte_equal']
            elif not item['byte_equal']:
                load(item['actual'])
else:
    old = js(DREF / 'results/fast-precision-864x480-v1/validation.json')
    targets = old['runs']['fp16_xmx' if arg.mode == 'fp16_xmx' else 'int8_dense']
    assert r['full_outputs_verified'] == 78
    assert r['held_outputs_survive_replay'] and r['caller_ownership_guards_passed']
    for name, rows in r['runs'].items():
        assert len(rows) == 39
        for ordinal, row in enumerate(rows):
            assert row['round'] == ordinal // 13 and row['frame'] == ordinal % 13
            target = targets[row['frame']]
            assert load(row['output']).astype('f2').tobytes() == load(target['actual']).astype('f2').tobytes()
            assert row['byte_equal_prior'] and row['private_byte_equal_output']
            assert row['effective_dispatch'] == target['dispatches']
            assert row['next_seed'] == (1 if row['reset'] else row['frame'] + 1)
        assert len(r['graphs'][name]) == 2 and len({entry['pool'] for entry in r['graphs'][name]}) == 1
    assert len(r['progress_fallback']) == 2 and all(row['byte_equal'] for row in r['progress_fallback'])
    assert next(row for row in r['progress_fallback'] if row['name'] == 'tiled')['matrix_calls']['tiled_dense'] > 0
audit = dict(scope=__doc__, stage=arg.stage, mode=arg.mode, passed=True,
             report_sha256=sha(path), auditor_sha256=sha(Path(__file__)), lease_sha256=sha(lease),
             process_exit_code=0, unique_complete_arrays_reread=len(verified),
             unique_array_bytes=sum(verified.values()), complete_migration=False)
audit_path.write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8', newline='\n')
print(json.dumps(audit), flush=True)
