"""Reread all v7 control graph display/private data and their native bytes."""
import hashlib
import json
from pathlib import Path
import numpy as np
import compressed_arrays_v1 as arrays

DREF = Path('D:/Codex-NR-Experiments/nr-b580/reference')
OUT = DREF / 'results/control-graph-v7-v1'
path = OUT / 'validation.json'
audit_path = OUT / 'saved-audit-v1.json'
assert not audit_path.exists()
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
js = lambda path: json.loads(Path(path).read_text(encoding='utf-8-sig'))
r = js(path)
assert r['passed'] and not r['complete_migration']
lease_path = Path(str(OUT) + '.log.lease.json')
assert js(lease_path)['returncode'] == 0
for source, digest in r['sources'].items():
    assert sha(source) == digest

def read(meta):
    if meta['format'] == 'npy+zlib':
        value = arrays.load(meta)
    else:
        assert meta['format'] == 'native-rgba-select-rgb-to-f16'
        assert sha(meta['path']) == meta['sha256']
        native = np.fromfile(meta['path'], meta['rgba_dtype']).reshape(256, 256, 4)[..., :3].copy()
        value = native.astype('f2')
        assert value.astype(native.dtype).tobytes() == native.tobytes()
    assert value.dtype == np.dtype('f2') and value.shape == (256, 256, 3)
    assert hashlib.sha256(value.tobytes()).hexdigest() == meta['raw_sha256']
    return value

assert [case['name'] for case in r['cases']] == ['intensity', 'tone', 'structure', 'style', 'auto-on', 'mask-add', 'mask-remove']
count = 0
for case in r['cases']:
    assert case['passed'] and len(case['frames']) == 8
    mask_case = case['name'].startswith('mask-')
    seeds = [0, 1, 2, 3] if mask_case else [0, 1, 2, 0] if case['name'] == 'intensity' else [0, 0, 1, 0]
    for index, row in enumerate(case['frames']):
        i = index % 4
        assert row['frame'] == i and row['round'] == index // 4
        assert row['native_seed'] == seeds[i] and row['next_seed'] == seeds[i] + 1
        assert row['native_display_byte_equal'] and row['native_private_byte_equal']
        assert row['runtime_dispatch']['xpu_graph_replay'] == 1 and row['graph_replays'] == index + 1
        actual, private = read(row['output']), read(row['private'])
        native_meta = row['native_output']
        assert sha(native_meta['path']) == native_meta['sha256']
        native = np.fromfile(native_meta['path'], '<f4').reshape(256, 256, 4)[..., :3].copy()
        assert actual.astype('f4').tobytes() == native.tobytes()
        assert private.tobytes() == read(row['native_private']).tobytes()
        if index >= 4:
            assert actual.tobytes() == read(case['frames'][i]['output']).tobytes()
            assert private.tobytes() == read(case['frames'][i]['private']).tobytes()
        count += 1
    assert case['invalid_input_preserves_state'] and case['held_outputs_survive'] and case['other_graph_outputs_survive']
    assert case['progress_fallback']['byte_equal'] and len(case['progress_fallback']['marks']) == 13
    assert case['has_float32_body_graph'] == mask_case
    assert len(case['graphs']) == (3 if mask_case else 2)
    assert len({graph['pool'] for graph in case['graphs']}) == 1
    assert all(graph['persistent_inputs_and_output_verified_outside_pool'] for graph in case['graphs'])
assert count == r['complete_native_display_comparisons'] == r['complete_native_private_comparisons'] == 56
assert r['progress_native_comparisons'] == 7 and r['class_entry_restored']
audit = dict(passed=True, report_sha256=sha(path), auditor_sha256=sha(Path(__file__)),
    lease_sha256=sha(lease_path), process_exit_code=0, complete_display_comparisons=56,
    complete_private_comparisons=56, runtime_progress_comparisons=7, complete_migration=False,
    scope='Reread complete native/candidate arrays; runtime history and progress guards remain explicitly scoped to the GPU run')
with audit_path.open('x', encoding='utf-8', newline='\n') as f:
    f.write(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit), flush=True)
