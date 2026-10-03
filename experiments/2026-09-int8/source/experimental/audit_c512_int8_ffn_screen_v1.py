"""Prewritten stored-result audit. No NumPy, model execution, or follow-up tests."""
import ast, hashlib, json, math, statistics, struct, sys, traceback, zlib
from datetime import datetime
from pathlib import Path

M = Path('D:/Codex-NR-Experiments/nr-b580/reference/results/c512-int8-ffn-screen-v1-monitor-luna-v1/request-from-main.json')
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
m = js(M)
cache = {}


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def file(path, expected=None, size=None):
    p = Path(path).resolve()
    key = str(p).casefold()
    if key not in cache:
        h, n = hashlib.sha256(), 0
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block)
                n += len(block)
        cache[key] = dict(path=str(p), sha256=h.hexdigest(), bytes=n)
    item = cache[key]
    assert expected is None or item['sha256'] == expected, p
    assert size is None or item['bytes'] == size, p
    return item


def scale_array(meta, width):
    file(meta['path'], meta['sha256'], meta['stored_bytes'])
    data = zlib.decompress(Path(meta['path']).read_bytes())
    assert hashlib.sha256(data).hexdigest() == meta['npy_sha256']
    assert data[:8] == b'\x93NUMPY\x01\x00'
    header_end = 10 + int.from_bytes(data[8:10], 'little')
    header = ast.literal_eval(data[10:header_end].decode('latin1').strip())
    assert header == dict(descr='<f4', fortran_order=False, shape=(1, width))
    value = data[header_end:]
    assert len(value) == meta['raw_bytes'] == width*4
    assert meta['shape'] == [1, width] and meta['dtype'] == '<f4'
    assert hashlib.sha256(value).hexdigest() == meta['raw_sha256']
    assert all(math.isfinite(v[0]) and v[0] > 0 for v in struct.iter_unpack('<f', value))


def metadata(value, shape, dtype):
    assert value['shape'] == shape and value['dtype'] == dtype
    assert value['bytes'] == math.prod(shape)*{'|i1': 1, '<f2': 2, '<f4': 4}[dtype]
    assert len(bytes.fromhex(value['raw_sha256'])) == 32


assert Path(m['log']).is_file(), 'Known-launched log missing'
if sys.argv[1:] == ['--startup']:
    assert not Path(m['startup']).exists()
    write(m['startup'], dict(read_at=datetime.now().astimezone().isoformat(),
          manifest=str(M), log=m['log'], log_bytes=Path(m['log']).stat().st_size))
    print('Startup recorded')
    sys.exit(0)
assert not sys.argv[1:] and not Path(m['handoff']).exists()
audit_result = dict(passed=False, phase='auditing', no_model_execution=True, no_rerun=True)
try:
    r, lease = js(m['result']), js(m['lease'])
    assert lease['owner'] == m['owner'] and lease['returncode'] == 0
    assert r['passed'] and r['phase'] == 'completed'
    assert not r.get('error') and not r.get('finalization_error')
    for key in ('candidate_promoted', 'full_candidate_model_test', 'new_video', 'visual_approval'):
        assert r[key] is False, key
    for key in ('new_quantization', 'default_unchanged', 'diagnostic_medians_are_not_additive',
                'baseline_history_seed_constants_and_graphs_unchanged', 'calibration_constants_unchanged'):
        assert r[key] is True, key
    for p, h in m['critical_sources'].items():
        file(p, h)
    for group in ('sources', 'exact_gate'):
        assert r[group]
        for p, h in r[group].items():
            file(p, h)
    prior_path = M.parents[1] / 'int8-ffn-range-repair-v1/validation.json'
    file(prior_path, 'bd53a2e2b818886e79fed05e836646bdd89814a434aa932ae5bdd4f7944ce2aa')
    prior = js(prior_path)
    expected_car = {v['frame']: v for v in prior['paired']['runs']['repaired_int8'] if v['round'] == 0}
    names = [f'{side}512.{i}.ffn' for side in ('encoder', 'decoder') for i in range(8)]
    calibration = ['car0', 'car5', 'car10', 'face48', 'face144', 'face242']
    heldout = ['car1', 'face96', 'face192']
    assert r['calibration_samples'] == calibration and r['heldout_samples'] == heldout
    assert r['calibration_margin'] == 1.25
    assert len(r['samples']) == 9 and {s['label'] for s in r['samples']} == set(calibration + heldout)
    by_sample = {}
    counts = dict(triton_calls=643, standalone_fp8=172, quantization_calls=395, elided_fp8=223)
    for sample in r['samples']:
        label, index = sample['label'], sample['frame']
        assert sample['calibration'] == (label in calibration) and sample['heldout'] == (label in heldout)
        for key in ('baseline_reference_byte_equal', 'eager_matches_frozen_graph',
                    'physical_sequence_unchanged', 'history_seed_and_inputs_unchanged'):
            assert sample[key]
        expected = expected_car[index] if sample['scene'] == 'car' else prior['frames'][index]
        low_hash = expected['low_raw_sha256'] if sample['scene'] == 'car' else expected['low']['raw_sha256']
        assert sample['low_raw_sha256'] == low_hash and sample['full_raw_sha256'] == expected['full_raw_sha256']
        assert sample['next_seed'] == expected['next_seed'] and sample['counts'] == counts
        assert [b['name'] for b in sample['boundaries']] == names
        for boundary in sample['boundaries']:
            metadata(boundary['input'], [144, 512], '<f2')
            metadata(boundary['output'], [144, 512], '<f2')
            assert boundary['triton_calls'] > 0
            by_sample[label, boundary['name']] = boundary
    assert len(r['calibration']) == 16 and [v['name'] for v in r['calibration']] == names
    for row in r['calibration']:
        assert row['fitted_only'] == calibration and row['no_calibration_clipping'] and row['staged_candidate_activations']
        for key, width in (('sz', 512), ('sh', 2048), ('sg', 512)):
            scale_array(row['scales'][key], width)
            assert row['scales'][key]['raw_sha256'] == row['packed_metadata'][key]['raw_sha256']
    assert len(r['cases']) == 144
    assert [(v['name'], v['sample']) for v in r['cases']] == [(n, s) for n in names for s in calibration + heldout]
    for row in r['cases']:
        boundary = by_sample[row['sample'], row['name']]
        assert row['input'] == boundary['input'] and row['reference'] == boundary['output']
        assert row['calibration'] == (row['sample'] in calibration) and row['heldout'] == (row['sample'] in heldout)
        for key in ('split_all_boundaries_match_cpu', 'fused_boundaries_match_cpu', 'nondebug_matches_cpu',
                    'fused_no_hidden_writes', 'debug_stores_disabled'):
            assert row[key]
        assert set(row['cpu']) == {'qx', 'sx', 'zraw', 'qz', 'hraw', 'qh', 'graw', 'qg', 'raw', 'out'}
        for key, value in row['cpu'].items():
            shape = [144] if key == 'sx' else [144, 2048 if key in ('qh', 'hraw') else 512]
            dtype = '<f4' if key == 'sx' else '|i1' if key in ('qx', 'qz', 'qh', 'qg') else '<f2'
            metadata(value, shape, dtype)
        assert set(row['clipping']) == {'linear', 'cubic', 'group_reduce'}
        for clip in row['clipping'].values():
            assert math.isfinite(clip['fraction']) and 0 <= clip['fraction'] <= 1
            assert math.isfinite(clip['maximum_range_multiple']) and clip['maximum_range_multiple'] >= 0
            if row['calibration']:
                assert clip['fraction'] == 0
        assert all(math.isfinite(v) and v >= 0 for v in row['error_vs_selected'].values())
    assert [v['name'] for v in r['timing']] == names
    labels = ['selected', 'split', 'fused']
    for row in r['timing']:
        for key in ('equal_output_copy', 'live_input_graph_checks', 'baseline_physical_sequence_and_fp8_elisions_equal',
                    'outputs_match_checked_eager', 'persistent_io_outside_pool', 'operands_unchanged', 'zero_entry_cpu_gpu_control'):
            assert row[key]
        assert row['captured_complete_segments'] == 8 and row['replays_per_round'] == 20
        assert len(row['orders']) == 7
        for i, order in enumerate(row['orders']):
            expected_order = labels[i % 3:] + labels[:i % 3]
            assert order == (expected_order[::-1] if i % 2 else expected_order)
        for label in labels:
            values = row['samples_ms'][label]
            assert len(values) == 7 and all(math.isfinite(v) and v > 0 for v in values)
            assert math.isclose(statistics.median(values), row['median_ms'][label], abs_tol=1e-9)
        for label in labels[1:]:
            assert math.isclose(row['change_percent'][label], (row['median_ms'][label]/row['median_ms']['selected'] - 1)*100, abs_tol=1e-9)
    assert set(r['resources']) == set(names)
    for group in r['resources'].values():
        assert len(group['kernels']) == len(group['selections']) == 11
        assert all(v['spills'] == 0 for v in group['kernels'].values())
        assert all(v['attempts'][-1]['spills'] == 0 and v['selected'] == v['attempts'][-1]['config'] for v in group['selections'].values())
    assert len(r['graphs']) == 2 and sum(v['replays'] for v in r['graphs']) == 16
    assert all(v['persistent_inputs_and_output_verified_outside_pool'] for v in r['graphs'])
    medians = {label: statistics.median(v['median_ms'][label] for v in r['timing']) for label in labels}
    assert medians == r['median_across_blocks_ms']
    final = json.loads(Path(m['log']).read_text(encoding='utf-8').splitlines()[-1])
    assert final == dict(passed=True, cases=144, timed_blocks=16, median_across_blocks_ms=medians, full_candidate_model_test=False)
    audit_result.update(passed=True, phase='completed', primary={key: file(m[key]) for key in ('result', 'log', 'lease')},
        unique_files_checked=len(cache), cases=144, timing=r['timing'], median_across_blocks_ms=medians,
        heldout_error_and_clipping=[{key: v[key] for key in ('name', 'sample', 'error_vs_selected', 'clipping')} for v in r['cases'] if v['heldout']],
        resources=r['resources'], candidate_promoted=False, full_candidate_model_test=False, visual_approval=False,
        scope='Independent CPU/GPU numerical contract screen and isolated FFN timing. Main assesses performance and approximation error; no quality acceptance.')
except BaseException:
    audit_result.update(passed=False, phase='failed', error=traceback.format_exc())
    raise
finally:
    write(m['handoff'], audit_result)
print(json.dumps(dict(passed=True, handoff=m['handoff'], cases=144)))
