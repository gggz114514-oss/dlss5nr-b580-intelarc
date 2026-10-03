"""Prewritten stored-result audit. No NumPy, model execution, or follow-up tests."""
import ast, hashlib, json, math, statistics, struct, sys, traceback, zlib
from datetime import datetime
from pathlib import Path

M = Path('D:/Codex-NR-Experiments/nr-b580/reference/results/c512-int8-range-v1-monitor-luna-v1/request-from-main.json')
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


def scale_array(meta, width, positive=True):
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
    values = [v[0] for v in struct.iter_unpack('<f', value)]
    assert all(math.isfinite(v) and (v > 0 if positive else v >= 0) for v in values)
    return values


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
    for key in ('candidate_promoted', 'full_candidate_model_test', 'new_video', 'visual_approval', 'runtime_calibration'):
        assert r[key] is False, key
    for key in ('new_quantization', 'default_unchanged', 'diagnostic_medians_are_not_additive',
                'baseline_history_seed_constants_and_graphs_unchanged', 'calibration_constants_unchanged', 'gpu_kernel_code_unchanged'):
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
    screen_path = M.parents[1] / 'c512-int8-ffn-screen-v2/validation.json'
    file(screen_path, '3e0ed164b48d7ed34fd914f6f2dc0c5cdaca5fa07ae7c6b66767ef43ab6c7ce7')
    screen = js(screen_path)
    old_cases = {(v['name'], v['sample']): v for v in screen['cases']}
    for name in ('c512_int8_ffn_oracle_v1.py', 'c512_int8_ffn_gpu_v1.py', 'c512_int8_ffn_capture_v1.py'):
        p = str(Path(m['audit_script']).parent / name)
        assert r['sources'][p] == screen['sources'][p]
        file(p, screen['sources'][p])
    guard = r['constant_guard']
    assert guard['excluded_state'] == ['_previous'] and guard['initial_history_none'] is True
    assert guard['initial_history_seed'] == 0 and guard['before'] == guard['after']
    assert set(guard['before']) == set(guard['graph_signature_names']) and len(guard['before']) == 819
    assert len(guard['graph_signature_names']) == len(set(guard['graph_signature_names']))
    assert '_previous' not in guard['before']
    assert len([n for n in guard['before'] if n.startswith('_int8_ffn_buffers.')]) == 40
    assert guard['difference'] == dict(added=[], removed=[], changed=[])
    for value in guard['before'].values():
        assert value['bytes'] > 0 and len(value['shape']) == len(value['stride'])
        assert len(bytes.fromhex(value['raw_sha256'])) == 32
    history = r['history_guard']
    assert history['before'] == history['after'] and history['same_object'] is True
    assert history['seed_before'] == history['seed_after'] == 217
    assert history['separately_checked_after_baseline'] is True and history['frozen_reference_frame'] == 216
    metadata(history['before'], [256, 256, 3], '<f2')
    assert history['before']['raw_sha256'] == prior['frames'][216]['low']['raw_sha256']
    decode = r['fresh_decode']
    assert decode['returncode'] == 0 and decode['output_bytes'] == 2*864*480*3
    assert decode['output_files_written'] is False and [v['frame'] for v in decode['frames']] == [168, 216]
    for value in decode['frames']:
        assert value['frozen_rgb_equal'] and value['raw_sha256'] == prior['frames'][value['frame']]['input_rgb8_sha256']
    expected_car = {v['frame']: v for v in prior['paired']['runs']['repaired_int8'] if v['round'] == 0}
    names = [f'{side}512.{i}.ffn' for side in ('encoder', 'decoder') for i in range(8)]
    calibration = ['car0', 'car5', 'car10', 'face48', 'face144', 'face242']
    diagnostic = ['car1', 'face96', 'face192']
    fresh = ['car3', 'car8', 'face72', 'face168', 'face216']
    heldout = diagnostic + fresh
    candidates = ['original', 'floor64', 'floor16']
    assert r['calibration_samples'] == calibration and r['heldout_samples'] == heldout
    assert r['diagnostic_samples'] == diagnostic and r['fresh_check_samples'] == fresh and r['candidates'] == candidates
    assert r['calibration_margin'] == 1.25
    assert not set(fresh).intersection(v['sample'] for v in screen['cases'])
    assert r['policies'] == dict(original='Frozen six-sample calibration',
        floor64='max(old_SH, group_training_absmax*1.25/64/127)', floor16='max(old_SH, group_training_absmax*1.25/16/127)')
    assert len(r['samples']) == 14 and {s['label'] for s in r['samples']} == set(calibration + heldout)
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
            by_sample[label, boundary['name']] = boundary
    assert len(r['calibration']) == 16 and [v['name'] for v in r['calibration']] == names
    scales_by_name = {}
    f32 = lambda v: struct.unpack('<f', struct.pack('<f', v))[0]
    for row in r['calibration']:
        assert row['fitted_only'] == calibration and row['original_reproduces_frozen'] is True
        assert list(row['variants']) == candidates
        old = next(v for v in screen['calibration'] if v['name'] == row['name'])
        assert row['variants']['original']['packed_metadata'] == old['packed_metadata']
        maximum = scale_array(row['hidden_training_absmax'], 2048, positive=False)
        loaded = {}
        for variant, packed in row['variants'].items():
            loaded[variant] = {}
            for key, width in (('sz', 512), ('sh', 2048), ('sg', 512)):
                loaded[variant][key] = scale_array(packed['scales'][key], width)
                assert packed['scales'][key]['raw_sha256'] == packed['packed_metadata'][key]['raw_sha256']
            folding = packed['folding']
            assert all(isinstance(v, int) and 0 <= v <= 2048 for v in folding.values())
            assert folding['lost_training_inactive_rows'] <= folding['lost_entire_rows']
        for variant, divisor in (('floor64', 64), ('floor16', 16)):
            original = row['variants']['original']['packed_metadata']
            revised = row['variants'][variant]['packed_metadata']
            assert all(revised[key] == original[key] for key in ('w0', 's0', 'sz', 'we', 'se', 'skip'))
            assert loaded[variant]['sz'] == loaded['original']['sz']
            expected_scales = [max(loaded['original']['sh'][c],
                f32(f32(f32(max(maximum[c//256*256:(c//256+1)*256])*1.25)/divisor)/127)) for c in range(2048)]
            assert loaded[variant]['sh'] == expected_scales
            assert all(a >= b for a, b in zip(loaded[variant]['sg'], loaded['original']['sg']))
        assert all(a >= b for a, b in zip(loaded['floor16']['sh'], loaded['floor64']['sh']))
        scales_by_name[row['name']] = loaded
    assert len(r['cases']) == 672
    assert [(v['name'], v['sample'], v['variant']) for v in r['cases']] == [(n, s, v) for n in names for s in calibration + heldout for v in candidates]
    by_case = {(v['name'], v['sample'], v['variant']): v for v in r['cases']}
    for row in r['cases']:
        boundary = by_sample[row['sample'], row['name']]
        assert row['input'] == boundary['input'] and row['reference'] == boundary['output']
        assert row['calibration'] == (row['sample'] in calibration)
        assert row['diagnostic'] == (row['sample'] in diagnostic) and row['fresh_check'] == (row['sample'] in fresh)
        for key in ('all_split_boundaries_match_cpu', 'nondebug_matches_cpu', 'debug_stores_disabled', 'hidden_before_quantization_unchanged'):
            assert row[key]
        old_key = row['name'], row['sample']
        reproduced = row['variant'] == 'original' and old_key in old_cases
        assert row['original_frozen_case_reproduced'] == reproduced
        if reproduced:
            assert row['cpu'] == old_cases[old_key]['cpu'] and row['input'] == old_cases[old_key]['input']
            assert row['error_vs_selected'] == old_cases[old_key]['error_vs_selected']
        for key, value in row['cpu'].items():
            shape = [144] if key == 'sx' else [144, 2048 if key in ('qh', 'hraw') else 512]
            dtype = '<f4' if key == 'sx' else '|i1' if key in ('qx', 'qz', 'qh', 'qg') else '<f2'
            metadata(value, shape, dtype)
        base = by_case[row['name'], row['sample'], 'original']
        assert all(row['cpu'][k] == base['cpu'][k] for k in ('qx', 'sx', 'zraw', 'qz', 'hraw'))
        for clip in row['clipping'].values():
            assert math.isfinite(clip['fraction']) and 0 <= clip['fraction'] <= 1
            assert math.isfinite(clip['maximum_range_multiple']) and clip['maximum_range_multiple'] >= 0
            if row['calibration']:
                assert clip['fraction'] == 0
        assert all(math.isfinite(v) and v >= 0 for v in row['error_vs_selected'].values())
        if row['sample'] in heldout:
            info = row['hidden_channel_diagnostics']
            assert info['total_values'] == 144*2048
            channels = info['channels']
            assert info['clipped_channels'] == len(channels) == len({v['channel'] for v in channels})
            assert sum(v['count'] for v in channels) == info['clipped_values']
            assert math.isclose(info['clipped_values']/info['total_values'], row['clipping']['cubic']['fraction'], abs_tol=1e-12)
            for key, flag in (('clipped_values_in_original_floor_channels', 'original_floor_limited'),
                              ('clipped_values_in_training_inactive_channels', 'training_inactive'),
                              ('clipped_values_in_lost_reduce_rows', 'lost_reduce_row')):
                assert info[key] == sum(v['count'] for v in channels if v[flag])
            for v in channels:
                assert 0 <= v['channel'] < 2048 and v['group'] == v['channel']//256 and 0 < v['count'] <= 144
                assert 0 <= v['original_reduce_nonzero'] <= 64 and 0 <= v['packed_reduce_nonzero'] <= 64
                assert v['lost_reduce_row'] == (v['original_reduce_nonzero'] > 0 and v['packed_reduce_nonzero'] == 0)
                assert v['training_inactive'] == (v['calibration_absmax'] == 0)
                assert all(math.isfinite(v[k]) and v[k] >= 0 for k in ('calibration_absmax', 'sample_absmax', 'capacity', 'maximum_overshoot', 'range_multiple'))
    assert len(r['counterfactuals']) == 128
    assert [(v['name'], v['sample']) for v in r['counterfactuals']] == [(n, s) for n in names for s in heldout]
    for row in r['counterfactuals']:
        assert row['diagnostic'] == (row['sample'] in diagnostic) and row['fresh_check'] == (row['sample'] in fresh)
        base = by_case[row['name'], row['sample'], 'original']
        assert row['original'] == base['cpu']['out']
        assert set(row['alternatives']) == {'no_hidden_clamp', 'original_reduce_weights_fp64'}
        for value in row['alternatives'].values():
            metadata(value['output'], [144, 512], '<f2')
            for key in ('error_vs_selected', 'error_vs_original_int8'):
                assert all(math.isfinite(v) and v >= 0 for v in value[key].values())
        if base['clipping']['cubic']['fraction'] == 0:
            assert row['alternatives']['no_hidden_clamp']['output'] == row['original']
    assert [v['name'] for v in r['timing']] == names
    labels = ['selected'] + candidates
    for row in r['timing']:
        for key in ('equal_output_copy', 'live_input_graph_checks', 'baseline_physical_sequence_and_fp8_elisions_equal',
                    'outputs_match_checked_eager', 'persistent_io_outside_pool', 'operands_unchanged',
                    'zero_entry_cpu_gpu_controls', 'all_candidates_same_launch_policies'):
            assert row[key]
        assert row['captured_complete_segments'] == 8 and row['replays_per_round'] == 20
        assert len(row['orders']) == 8
        for i, order in enumerate(row['orders']):
            expected_order = labels[i % 4:] + labels[:i % 4]
            assert order == (expected_order[::-1] if i >= 4 else expected_order)
        for label in labels:
            values = row['samples_ms'][label]
            assert len(values) == 8 and all(math.isfinite(v) and v > 0 for v in values)
            assert math.isclose(statistics.median(values), row['median_ms'][label], abs_tol=1e-9)
        for label in candidates:
            assert math.isclose(row['change_vs_selected_percent'][label], (row['median_ms'][label]/row['median_ms']['selected'] - 1)*100, abs_tol=1e-9)
        for label in candidates[1:]:
            assert math.isclose(row['change_vs_original_int8_percent'][label], (row['median_ms'][label]/row['median_ms']['original'] - 1)*100, abs_tol=1e-9)
    assert set(r['resources']) == set(names)
    for variants in r['resources'].values():
        assert list(variants) == candidates
        assert all(v['selections'] == variants['original']['selections'] for v in variants.values())
        for group in variants.values():
            assert len(group['kernels']) == len(group['selections']) == 11
            assert all(v['spills'] == 0 for v in group['kernels'].values())
    assert len(r['graphs']) == 2 and sum(v['replays'] for v in r['graphs']) == 19
    assert all(v['persistent_inputs_and_output_verified_outside_pool'] for v in r['graphs'])
    medians = {label: statistics.median(v['median_ms'][label] for v in r['timing']) for label in labels}
    assert medians == r['median_across_blocks_ms']
    final = json.loads(Path(m['log']).read_text(encoding='utf-8').splitlines()[-1])
    assert final == dict(passed=True, cases=672, timed_blocks=16, counterfactuals=128, median_across_blocks_ms=medians, full_candidate_model_test=False)
    summaries = {}
    for category, samples in (('diagnostic', diagnostic), ('fresh_check', fresh)):
        summaries[category] = {}
        for variant in candidates:
            rows = [v for v in r['cases'] if v['sample'] in samples and v['variant'] == variant]
            summaries[category][variant] = dict(cases=len(rows),
                median_feature_relative_rmse=statistics.median(v['error_vs_selected']['relative_rmse'] for v in rows),
                maximum_feature_relative_rmse=max(v['error_vs_selected']['relative_rmse'] for v in rows),
                hidden_out_of_range_values=sum(v['hidden_channel_diagnostics']['clipped_values'] for v in rows),
                hidden_out_of_range_values_in_floor_channels=sum(v['hidden_channel_diagnostics']['clipped_values_in_original_floor_channels'] for v in rows),
                hidden_out_of_range_values_in_lost_rows=sum(v['hidden_channel_diagnostics']['clipped_values_in_lost_reduce_rows'] for v in rows))
    audit_result.update(passed=True, phase='completed', primary={key: file(m[key]) for key in ('result', 'log', 'lease')},
        unique_files_checked=len(cache), cases=672, timing=r['timing'], median_across_blocks_ms=medians,
        probe_summary=summaries, constant_guard=dict(immutable_buffers=819, difference=guard['difference']),
        history_guard=history, gpu_kernel_code_unchanged=True, candidate_promoted=False,
        full_candidate_model_test=False, visual_approval=False,
        scope='Stored numerical/range/timing diagnostics audited; main decides next action. No full-model, image-quality or temporal acceptance.')
except BaseException:
    audit_result.update(passed=False, phase='failed', error=traceback.format_exc())
    raise
finally:
    write(m['handoff'], audit_result)
print(json.dumps(dict(passed=True, handoff=m['handoff'], cases=672)))
