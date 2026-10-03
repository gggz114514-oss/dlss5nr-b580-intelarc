"""Prewritten CPU stored-result audit; never executes the model or retries."""
import hashlib, json, math, statistics, sys, traceback
from datetime import datetime
from pathlib import Path
M = Path('D:/Codex-NR-Experiments/nr-b580/reference/results/c512-qkv-int8-full-v1-monitor-luna-v1/request-from-main.json')
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
m = js(M)
cache = {}


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def file(path, expected=None, size=None):
    p = Path(path).resolve()
    key = str(p).casefold()
    if key not in cache:
        digest, count = hashlib.sha256(), 0
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                digest.update(block)
                count += len(block)
        cache[key] = dict(path=str(p), sha256=digest.hexdigest(), bytes=count)
    value = cache[key]
    assert expected is None or value['sha256'] == expected, p
    assert size is None or value['bytes'] == size, p
    return value


def meta(value, shape, dtype):
    assert value['shape'] == shape and value['dtype'] == dtype
    assert value['bytes'] == math.prod(shape)*{'|i1': 1, '<f2': 2, '<f4': 4}[dtype]
    assert len(bytes.fromhex(value['raw_sha256'])) == 32


assert Path(m['log']).is_file(), 'Known-launched log missing'
if sys.argv[1:] == ['--startup']:
    write(m['startup'], dict(read_at=datetime.now().astimezone().isoformat(),
        manifest=str(M), log=m['log'], log_bytes=Path(m['log']).stat().st_size))
    print('Startup recorded')
    sys.exit(0)
assert not sys.argv[1:] and not Path(m['handoff']).exists()
audit_result = dict(passed=False, phase='auditing', no_model_execution=True, no_rerun=True)
try:
    r, lease = js(m['result']), js(m['lease'])
    assert lease['owner'] == m['owner'] and lease['returncode'] == 0
    assert r['passed'] and r['phase'] == 'completed' and not r.get('error') and not r.get('finalization_error')
    for key in ('arithmetic_changed', 'fp8_boundaries_preserved', 'default_unchanged', 'arithmetic_kernels_unchanged', 'own_history_for_every_frame',
                'donor_remained_fresh', 'reset_reproduces_first_frame', 'all_baseline_outputs_match_accepted',
                'inputs_constants_history_held_outputs_checked', 'all_constants_and_history_outside_pool',
                'no_extra_c512_debug_stores', 'body_inputs_and_private_history_unchanged'):
        assert r[key] is True, key
    for key in ('candidate_promoted', 'new_quality_approved', 'recalibration', 'saved_raw_frames_or_weights', 'exact_backend_changed', 'guards_removed'):
        assert r[key] is False, key
    assert r['human_review'] == 'pending' and r['range_policy'] == 'frozen_floor16_split'
    for path, digest in m['critical_sources'].items():
        file(path, digest)
    for group in ('sources', 'exact_gate'):
        assert r[group]
        for path, digest in r[group].items():
            file(path, digest)
    prior_path = M.parents[1] / 'int8-ffn-range-repair-v1/validation.json'
    file(prior_path, 'bd53a2e2b818886e79fed05e836646bdd89814a434aa932ae5bdd4f7944ce2aa')
    prior = js(prior_path)
    approved_path = M.parents[1]/'c512-int8-full-v1/validation.json'
    file(approved_path, 'd1d2b8e21b1c9e95e7d99b7e0bf5735237781e2412301295e7472aeff92ba7f2')
    approved = js(approved_path)
    screen_path = M.parents[1]/'c512-qkv-int8-v2/validation.json'
    file(screen_path, '0b8c5620e0260488ddc8654a2198de4416fe80612c8162ea6a8d1766574852eb')
    screen = js(screen_path)
    assert r['qkv_packed'] == screen['packed_constants_before']
    assert r['qkv_registry_swap_rejected_before_execution'] == ['qkv_signature','graph_signature']

    range_path = M.parents[1] / 'c512-int8-range-v1/validation.json'
    file(range_path, '97756e1d6c4af2c7d4cf97120ed0778820b9cf594232167ffe5ea451f018fda4')
    ranges = js(range_path)
    for row in ranges['calibration']:
        for value in row['variants']['floor16']['scales'].values():
            file(value['path'], value['sha256'], value['stored_bytes'])
    names = [f'{side}512.{i}' for side in ('encoder', 'decoder') for i in range(8)]
    expected_packs = {v['name'].removesuffix('.ffn'): v['variants']['floor16']['packed_metadata'] for v in ranges['calibration']}
    assert r['constant_guard']['baseline']['before'] == approved['constant_guard']['floor16']['before']
    assert {k:v for k,v in r['constant_guard']['qkv_int8']['before'].items()
            if not k.startswith('_c512_qkv_int8_buffers.')} == r['constant_guard']['baseline']['before']
    assert r['reconstructed_pack_metadata'] == expected_packs
    assert r['candidate']['c512_int8']['reconstructed_pack_metadata'] == expected_packs
    for name, count in (('baseline', 1011), ('qkv_int8', 1043)):
        guard = r['constant_guard'][name]
        assert guard['before'] == guard['after'] and len(guard['before']) == count
        assert '_previous' not in guard['before']
        assert len([n for n in guard['before'] if n.startswith('_int8_ffn_buffers.')]) == 40
        assert len([n for n in guard['before'] if n.startswith('_c512_int8_buffers.')]) == 192
        assert len([n for n in guard['before'] if n.startswith('_c512_qkv_int8_buffers.')]) == (32 if name=='qkv_int8' else 0)
        for value in guard['before'].values():
            assert value['bytes'] > 0 and len(bytes.fromhex(value['raw_sha256'])) == 32
        share = r['sharing'][name]
        assert share == dict(base_buffers=779, extras=count-779, base_objects_shared=True, packed_constants_disjoint=True)
    assert len(r['cpu_checks']) == 64
    labels = ['car1_reset', 'car1_temporal', 'face96', 'face192']
    assert {(v['label'], v['block']) for v in r['cpu_checks']} == {(label, n) for label in labels for n in names}
    for row in r['cpu_checks']:
        assert row['all_equal'] and row['input_unchanged'] and row['buffers'] == row['cpu']
        meta(row['input'], [144, 512], '<f2')
        assert set(row['buffers']) == {'qx', 'sx', 'qz', 'qh', 'qg', 'out'}
        for key in ('qx', 'qz', 'qg'):
            meta(row['buffers'][key], [144, 512], '|i1')
        meta(row['buffers']['sx'], [144], '<f4')
        meta(row['buffers']['qh'], [144, 2048], '|i1')
        meta(row['buffers']['out'], [144, 512], '<f2')
        for value in row['clipping'].values():
            assert 0 <= value['fraction'] <= 1 and math.isfinite(value['maximum_range_multiple'])
    checks = r['eager_graph_checks']
    expected_checks = {(name, label) for name in ('baseline', 'qkv_int8')
        for label in ('car1_reset', 'car1_temporal', 'live_reset', 'live_temporal')} | {('qkv_int8', 'face96'), ('qkv_int8', 'face192')}
    assert len(checks) == 10 and {(v['route'], v['label']) for v in checks} == expected_checks
    for row in checks:
        assert row['byte_equal'] and row['inputs_restored'] and row['private_history_and_seed_unchanged']
        live = row['label'].startswith('live_')
        assert row['live_input_changed'] == row['live_input_consumed'] == live
        assert row['cpu_blocks'] == (16 if row['route'] == 'qkv_int8' and not live else 0)
    expected = {v['frame']: v for v in approved['paired']['runs']['floor16'] if v['round'] == 0}
    for name in ('baseline', 'qkv_int8'):
        body = r['body'][name]
        assert len(body['samples_ms']) == 8 and all(math.isfinite(v) and v > 0 for v in body['samples_ms'])
        assert statistics.median(body['samples_ms']) == body['median_ms']
        builds = r['capture_build_counts'][name]
        assert len(builds) == 6
        if name == 'baseline':
            assert all(v == dict(triton_calls=643, standalone_fp8=140, quantization_calls=315, elided_fp8=175) for v in builds)
        if name == 'qkv_int8':
            assert all(v == dict(triton_calls=659, standalone_fp8=140, quantization_calls=315, elided_fp8=175) for v in builds)
        rows = r['paired']['runs'][name]
        assert len(rows) == 52 and r['paired']['passed']
        hashes = {}
        for index, row in enumerate(rows):
            assert (row['round'], row['frame']) == divmod(index, 13)
            order = ['baseline', 'qkv_int8'] if (row['round']+row['frame']) % 2 == 0 else ['qkv_int8', 'baseline']
            assert row['order'] == order and row['reset'] == expected[row['frame']]['reset']
            assert row['own_repeat_equal'] and row['private_history_matches_low']
            assert row['next_seed'] == expected[row['frame']]['next_seed']
            assert math.isfinite(row['host_ms']) and row['host_ms'] > 0
            pair = row['full_raw_sha256'], row['low_raw_sha256']
            if row['round'] == 0:
                hashes[row['frame']] = pair
            else:
                assert hashes[row['frame']] == pair
            if name == 'baseline':
                ref = expected[row['frame']]
                assert pair == (ref['full_raw_sha256'], ref['low_raw_sha256'])
        for mode in ('all', 'temporal'):
            summary = r['paired']['summary'][name][mode]
            selected = [v for v in rows if mode == 'all' or not v['reset']]
            assert statistics.mean(v['host_ms'] for v in selected) == summary['mean_host_ms']
            assert [statistics.mean(v['host_ms'] for v in selected if v['round'] == i) for i in range(4)] == summary['round_mean_host_ms']
        graphs = r['graphs'][name]
        assert len(graphs) == 2 and sum(v['replays'] for v in graphs) == 298
        assert all(v['persistent_inputs_and_output_verified_outside_pool'] for v in graphs)
        assert all(v['captured_dispatch']['int8_ffn_segment'] == 8 for v in graphs)
        assert all(v['captured_dispatch']['c512_int8_ffn_segment'] == 16 for v in graphs)
        if name == 'qkv_int8':
            assert all(v['captured_dispatch']['c512_int8_qkv'] == 16 for v in graphs)
    candidate = r['candidate']['c512_int8']
    assert candidate['policy'] == 'frozen_floor16_split' and candidate['registered_constants'] == 192
    assert candidate['calls'] == 192 and candidate['blocks'] == {n: 12 for n in names}
    assert candidate['runtime_calibration'] is False and candidate['per_invocation_scratch'] and candidate['arithmetic_kernels_unchanged']
    keys = {'entry', 'linear_debugFalse', 'expand_debugFalse', 'reduce_debugFalse', 'project_debugFalse'}
    assert set(candidate['resources']) == set(candidate['selections']) == keys
    frozen = ranges['resources']['encoder512.0.ffn']['floor16']
    for key in keys:
        assert candidate['resources'][key] == frozen['kernels'][key]
        assert candidate['selections'][key] == frozen['selections'][key]
        assert candidate['resources'][key]['spills'] == 0
    assert len(r['qkv_cpu_checks']) == 64
    assert [(v['label'],v['block']) for v in r['qkv_cpu_checks']] == [(label,n) for label in labels for n in names]
    for row in r['qkv_cpu_checks']:
        d = row['detail']
        assert d['name'] == row['block'] and d['input_quantization_byte_equal'] and d['dense_samples_byte_equal']
        assert d['samples'] == 48 and row['input_unchanged']
        assert all(len(bytes.fromhex(d[k]))==32 for k in ('qx_sha256','sx_sha256','sample_sha256'))
        assert len(row['qkv_hashes']) == 3 and all(len(bytes.fromhex(h))==32 for h in row['qkv_hashes'])
    qkv = r['candidate']['qkv_int8']
    keys = ['int8:quantize_rows','int8:dense(16, 64)']+['qkv:pack_'+str(s) for s in [(0,0),(0,4),(4,0),(4,4)]]
    assert qkv['resources'] == {k:screen['candidate_resources'][k] for k in keys}
    assert r['qkv_screen_hashes'] == {k:v['hash'] for k,v in qkv['resources'].items()}
    assert qkv['config']==[16,64] and qkv['registered_constants']==32
    assert qkv['calls']==192 and qkv['blocks']=={n:12 for n in names}
    for key in ('input_quantization_per_replay','weight_packing_at_construction','fp8_boundaries_preserved','per_invocation_scratch'):
        assert qkv[key] is True
    # Registered GPU bytes must be the screened immutable packs and scale arrays.
    guard = r['constant_guard']['qkv_int8']['before']
    for i,n in enumerate(names):
        for key in ('qw','sw'):
            assert guard[f'_c512_qkv_int8_buffers.{i}.{key}']['raw_sha256'] == r['qkv_packed'][n][key+'_sha256']
    assert len(r['frames']) == r['frames_completed'] == 243
    assert r['panel_order'] == ['input', 'baseline', 'qkv_int8'] and r['display_geometry'] == [2592, 1352]
    for i, row in enumerate(r['frames']):
        assert row['frame'] == i and row['reset'] == (i == 0) and row['inputs_unchanged'] and row['held_outputs_unchanged']
        assert row['input_rgb8_sha256'] == prior['frames'][i]['input_rgb8_sha256']
        assert row['motion'] == prior['frames'][i]['motion']
        file(row['motion']['path'], row['motion']['sha256'], row['motion']['stored_bytes'])
        assert set(row['runs']) == {'baseline', 'qkv_int8'}
        for name, value in row['runs'].items():
            assert value['finite'] and value['private_history_matches_low'] and value['next_seed'] == i+1
            assert all(len(bytes.fromhex(value[k])) == 32 for k in ('full_raw_sha256', 'low_raw_sha256'))
            if name == 'baseline':
                assert value['full_raw_sha256'] == approved['frames'][i]['runs']['floor16']['full_raw_sha256']
                assert value['low_raw_sha256'] == approved['frames'][i]['runs']['floor16']['low_raw_sha256']
        for region in ('full', 'roi'):
            value = row['metrics'][region]
            assert all(math.isfinite(value[k]) and value[k] >= 0 for k in ('rgb_rmse_8bit', 'mae_8bit'))
            assert len(value['mean_delta_RGB_8bit']) == 3 and all(math.isfinite(v) for v in value['mean_delta_RGB_8bit'])
        if i:
            assert math.isfinite(row['metrics']['temporal_error_mae_8bit']) and row['metrics']['temporal_error_mae_8bit'] >= 0
    qs = r['quality_summary']
    for region in ('full', 'roi'):
        assert qs[f'mean_{region}_rgb_rmse_8bit'] == statistics.mean(v['metrics'][region]['rgb_rmse_8bit'] for v in r['frames'])
    assert qs['mean_temporal_error_mae_8bit'] == statistics.mean(v['metrics']['temporal_error_mae_8bit'] for v in r['frames'][1:])
    assert len(r['images']) == 3
    for path, digest in r['images'].items():
        file(path, digest)
    vf = r['video_finalization']
    assert vf['passed'] and vf['video'] == r['video']
    assert vf['packet_timestamps_equal'] and vf['all_decoded_frames_byte_equal'] and vf['nclx'] == [1, 1, 1]
    probe = vf['probe']
    assert (probe['width'], probe['height'], probe['pix_fmt'], probe['avg_frame_rate'], int(probe['nb_frames'])) == (2592, 1352, 'yuv420p', '24/1', 243)
    assert abs(float(probe['duration']) - 10.125) < .001 and probe['color_range'] == 'tv'
    assert all(probe[k] == 'bt709' for k in ('color_space', 'color_transfer', 'color_primaries'))
    assert vf['vui'] == dict(colour_primaries=1, transfer_characteristics=1, matrix_coefficients=1, video_full_range_flag=0)
    assert len(vf['decoded_frames']) == 243
    for i, row in enumerate(vf['decoded_frames']):
        assert row['frame'] == i and row['byte_equal'] and row['bytes'] == 2592*1352*3//2
        assert len(bytes.fromhex(row['sha256'])) == 32
    file(r['video']['path'], r['video']['sha256'], r['video']['bytes'])
    final = json.loads(Path(m['log']).read_text(encoding='utf-8').splitlines()[-1])
    assert final == dict(passed=True, frames=243, cpu_checks=64, video=r['video'], body=r['body'], paired=r['paired']['summary'], human_review='pending')
    audit_result.update(passed=True, phase='completed', primary={k: file(m[k]) for k in ('result', 'log', 'lease')},
        unique_files_checked=len(cache), frames=243, cpu_checks=64, eager_graph_checks=10,
        body=r['body'], paired=r['paired']['summary'], qkv_cpu_checks=64, quality_summary=qs, video=r['video'], images=r['images'],
        candidate_promoted=False, human_review='pending', scope='Full candidate arithmetic and temporal execution audited; human image review remains pending.')
except BaseException:
    audit_result.update(passed=False, phase='failed', error=traceback.format_exc())
    raise
finally:
    write(m['handoff'], audit_result)
print(json.dumps(dict(passed=True, handoff=m['handoff'], frames=243)))
