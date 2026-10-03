"""Prewritten stored-result audit for compact C512 queries; no model execution."""
import hashlib, json, math, statistics, sys, traceback
from datetime import datetime
from pathlib import Path
M = Path('D:/Codex-NR-Experiments/nr-b580/reference/results/c512-compact-queries-v1-monitor-luna-v1/request-from-main.json')
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
        h, length = hashlib.sha256(), 0
        with p.open('rb') as stream:
            for block in iter(lambda: stream.read(1024*1024), b''):
                h.update(block); length += len(block)
        cache[key] = dict(path=str(p), sha256=h.hexdigest(), bytes=length)
    value = cache[key]
    assert expected is None or value['sha256'] == expected, p
    assert size is None or value['bytes'] == size, p
    return value


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
    for key in ('default_unchanged', 'padded_kv_influence_preserved', 'ignored_queries_have_no_influence',
                'all_frozen_outputs_equal', 'same_cpu_guards', 'inputs_constants_histories_and_pool_verified',
                'reset_reproduced', 'scopes_restored', 'donor_never_executed'):
        assert r[key] is True, key
    for key in ('candidate_promoted', 'arithmetic_changed', 'guards_removed', 'recalibration', 'new_video',
                'exact_backend_changed', 'nvidia_byte_parity_claimed', 'saved_raw_tensors'):
        assert r[key] is False, key
    for path, digest in m['critical_sources'].items():
        file(path, digest)
    for group in ('sources', 'exact_gate'):
        assert r[group]
        for path, digest in r[group].items():
            file(path, digest)
    fp = M.parents[1]/'c512-int8-full-v1/validation.json'
    file(fp, 'd1d2b8e21b1c9e95e7d99b7e0bf5735237781e2412301295e7472aeff92ba7f2')
    full = js(fp)
    review = fp.with_name('user-review-v1.json')
    file(review, '2f923c191f990960ea60ed28347d411057424adc57b38d99b4ac185042c6c9d6')
    assert js(review)['status'] == 'accepted_face_quality_for_c512_int8_full_v1'
    assert r['reused_approved_video'] == full['video'] == js(review)['video']
    v = r['reused_approved_video']; file(v['path'], v['sha256'], v['bytes'])
    # Authenticate static scales and all input motion artifacts, without importing Torch.
    rp = M.parents[1]/'c512-int8-range-v1/validation.json'
    file(rp, '97756e1d6c4af2c7d4cf97120ed0778820b9cf594232167ffe5ea451f018fda4')
    for row in js(rp)['calibration']:
        for v in row['variants']['floor16']['scales'].values():
            file(v['path'], v['sha256'], v['stored_bytes'])
    vp = M.parents[1]/'int8-ffn-range-repair-v1/validation.json'
    file(vp, 'bd53a2e2b818886e79fed05e836646bdd89814a434aa932ae5bdd4f7944ce2aa')
    for row in js(vp)['calibration']['layers']:
        v = row['hidden_scale']; file(v['path'], v['sha256'], v['stored_bytes'])
    shifts = [(0, 0), (0, 4), (4, 0), (4, 4)]
    assert len(r['geometry']) == 8
    assert {(tuple(v['shift']), v['bm']) for v in r['geometry']} == {(s, b) for s in shifts for b in (16, 32)}
    for row in r['geometry']:
        sy, sx = row['shift']; bm = row['bm']; coords = []
        assert len(row['geometry']) == 4
        for w, g in enumerate(row['geometry']):
            wy, wx = divmod(w, 2)
            y0, x0 = max(sy-wy*8, 0), max(sx-wx*8, 0)
            h, width = min(sy+12-wy*8, 8)-y0, min(sx+12-wx*8, 8)-x0
            assert g == dict(window=w, y0=y0, x0=x0, height=h, width=width, tiles=(h*width+bm-1)//bm)
            coords.extend((wy*8+y0+y-sy, wx*8+x0+x-sx) for y in range(h) for x in range(width))
        assert len(coords) == len(set(coords)) == 144 and set(coords) == {(y, x) for y in range(12) for x in range(12)}
        assert row['coverage_exact'] and row['valid_pixels'] == 144 and row['tiles'] == (5 if bm == 32 else 9)
    cases = ['zero', 'distinct', 'padded_kv_changed', 'ignored_q_changed']
    assert len(r['controls']) == 16
    assert {(tuple(v['shift']), v['case']) for v in r['controls']} == {(s, c) for s in shifts for c in cases}
    for row in r['controls']:
        assert row['attention_byte_equal'] and row['full_projection_byte_equal'] and row['inputs_unchanged']
        assert len(bytes.fromhex(row['compact_sha256'])) == len(bytes.fromhex(row['projection_sha256'])) == 32
        assert set(row['resources']) == {'attention', 'projection'}
        for label, res in row['resources'].items():
            assert res['spills'] == 0 and res['selection']['attempts'][-1]['spills'] == 0
            assert res['selection']['selected'] in ([[32], [16]] if label == 'attention' else [[16, 32]])
    for shift in shifts:
        values = {v['case']: v['compact_sha256'] for v in r['controls'] if tuple(v['shift']) == shift}
        assert values['distinct'] == values['ignored_q_changed'] != values['padded_kv_changed']
    names = [f'{side}512.{i}' for side in ('encoder', 'decoder') for i in range(8)]
    labels = ['reset', 'temporal', 'face96', 'face192']
    assert len(r['boundaries']) == 128
    assert {(v['label'], v['block'], v['kind']) for v in r['boundaries']} == {
        (label, name, kind) for label in labels for name in names for kind in ('attention', 'projection')}
    for row in r['boundaries']:
        assert row['byte_equal'] and row['inputs_unchanged'] and tuple(row['shift']) in shifts
        assert row['shape'] == ([16, 144, 32] if row['kind'] == 'attention' else [12, 12, 512])
        assert len(bytes.fromhex(row['raw_sha256'])) == 32
    checks = r['eager_checks']; routes = ['floor16', 'compact']
    assert len(checks) == 10
    assert {(v['route'], v['label']) for v in checks} == {
        (n, label) for n in routes for label in ('reset', 'temporal', 'live_reset', 'live_temporal')} | {('compact', 'face96'), ('compact', 'face192')}
    for row in checks:
        assert row['graph_eager_byte_equal'] and row['inputs_restored'] and row['history_unchanged']
        live = row['label'].startswith('live_')
        assert row['live'] == row['live_input_consumed'] == live
        assert row['probed_blocks'] == (16 if row['route'] == 'compact' and not live else 0)
    expected = {v['frame']: v for v in full['paired']['runs']['floor16'] if v['round'] == 0}
    frozen_hashes = {n: v['raw_sha256'] for n, v in full['constant_guard']['floor16']['before'].items()}
    for name in routes:
        samples = r['body'][name]['samples_ms']
        assert len(samples) == 8 and all(math.isfinite(x) and x > 0 for x in samples)
        assert r['body'][name]['median_ms'] == statistics.median(samples)
        rows = r['paired'][name]
        assert len(rows) == 52
        previous, seed = None, 0
        for j, row in enumerate(rows):
            repeat, frame = divmod(j, 13)
            ref = expected[frame]
            assert (row['round'], row['frame']) == (repeat, frame)
            assert row['order'] == (routes if (repeat+frame)%2 == 0 else routes[::-1])
            assert row['reset'] == ref['reset']
            assert row['previous_low_sha256'] == previous and row['seed_before'] == seed
            assert row['full_raw_sha256'] == ref['full_raw_sha256'] and row['low_raw_sha256'] == ref['low_raw_sha256']
            assert row['next_seed'] == ref['next_seed']
            previous, seed = row['low_raw_sha256'], row['next_seed']
            assert row['frozen_equal'] and row['history_private'] and row['other_history_unchanged']
            assert math.isfinite(row['host_ms']) and row['host_ms'] > 0
        for part in ('all', 'temporal'):
            subset = [v for v in rows if part == 'all' or not v['reset']]
            assert len(subset) == (52 if part == 'all' else 44)
            assert r['pipeline'][name][part] == dict(mean_ms=statistics.mean(v['host_ms'] for v in subset),
                round_mean_ms=[statistics.mean(v['host_ms'] for v in subset if v['round'] == k) for k in range(4)])
        guards = r['constant_guard'][name]
        assert guards['before'] == guards['after'] == frozen_hashes and len(guards['before']) == 1011
        assert r['capture_counts'][name] == full['capture_build_counts']['floor16']
        graphs = r['graphs'][name]
        assert len(graphs) == 2 and sum(v['replays'] for v in graphs) == 298
        assert all(v['persistent_inputs_and_output_verified_outside_pool'] for v in graphs)
        assert [v['captured_dispatch'] for v in graphs] == [v['captured_dispatch'] for v in full['graphs']['floor16']]
    assert len(r['frames']) == r['frames_completed'] == 243
    for i, row in enumerate(r['frames']):
        ref = full['frames'][i]
        assert row['frame'] == i and row['reset'] == (i == 0)
        assert row['input_rgb8_sha256'] == ref['input_rgb8_sha256'] and row['motion'] == ref['motion']
        v = row['motion']; file(v['path'], v['sha256'], v['stored_bytes'])
        assert row['inputs_unchanged'] and row['held_outputs_unchanged'] and set(row['runs']) == set(routes)
        for name in routes:
            v, f = row['runs'][name], ref['runs']['floor16']
            assert v['full_raw_sha256'] == f['full_raw_sha256'] and v['low_raw_sha256'] == f['low_raw_sha256']
            assert v['next_seed'] == i+1 and v['frozen_equal'] and v['private_history_matches_low']
    candidate = r['candidate']
    assert candidate['calls'] == 192 and candidate['blocks'] == {n: 12 for n in names}
    assert candidate['constants_added'] == 0 and candidate['full_kv_positions'] == 64
    assert candidate['per_block_output_bytes'] == 147456 and candidate['old_per_block_output_bytes'] == 262144
    for key in ('quantization_unchanged', 'unquantized_pool_input_preserved', 'public_boundary_api_unchanged', 'capture_only_layout_rewrite'):
        assert candidate[key] is True
    assert candidate['resources'] and all(v['spills'] == 0 for v in candidate['resources'].values())
    assert 'projection' in candidate['selections']
    for label, sel in candidate['selections'].items():
        assert sel['attempts'][-1]['spills'] == 0
        assert sel['selected'] in ([[16, 32]] if label == 'projection' else [[32], [16]])
    final = json.loads(Path(m['log']).read_text(encoding='utf-8').splitlines()[-1])
    assert final == dict(passed=True, body=r['body'], pipeline=r['pipeline'], frames=243,
        boundary_checks=128, controls=16, all_frozen_outputs_equal=True)
    audit_result.update(passed=True, phase='completed', primary={k: file(m[k]) for k in ('result', 'log', 'lease')},
        unique_files_checked=len(cache), controls=16, boundaries=128, eager_checks=10, frames=243,
        pipeline_rows=104, body=r['body'], pipeline=r['pipeline'], candidate=candidate,
        all_frozen_outputs_equal=True, same_cpu_guards=True, candidate_promoted=False,
        nvidia_byte_parity_claimed=False, reused_approved_video=r['reused_approved_video'])
except BaseException:
    audit_result.update(passed=False, phase='failed', error=traceback.format_exc())
    raise
finally:
    write(m['handoff'], audit_result)
print(json.dumps(dict(passed=True, handoff=m['handoff'], frames=243, boundaries=128)))
