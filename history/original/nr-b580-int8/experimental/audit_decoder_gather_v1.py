"""Fixed CPU stored-result audit for the byte-preserving decoder screen."""
import hashlib, json, math, statistics, sys, traceback
from datetime import datetime
from pathlib import Path
M = Path('D:/Codex-NR-Experiments/nr-b580/reference/results/decoder-gather-v1-monitor-luna-v1/request-from-main.json')
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
m, cache = js(M), {}


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def file(path, expected=None):
    p = Path(path).resolve()
    key = str(p).casefold()
    if key not in cache:
        h, length = hashlib.sha256(), 0
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024*1024), b''):
                h.update(block)
                length += len(block)
        cache[key] = dict(path=str(p), sha256=h.hexdigest(), bytes=length)
    v = cache[key]
    assert expected is None or v['sha256'] == expected, p
    return v


assert Path(m['log']).is_file()
if sys.argv[1:] == ['--startup']:
    write(m['startup'], dict(at=datetime.now().astimezone().isoformat(), manifest=str(M), log=m['log']))
    print('Startup recorded')
    sys.exit(0)
assert not sys.argv[1:] and not Path(m['handoff']).exists()
result = dict(passed=False, phase='auditing', no_model_execution=True, no_rerun=True)
try:
    r, lease = js(m['result']), js(m['lease'])
    assert lease['owner'] == m['owner'] and lease['returncode'] == 0
    assert r['passed'] and r['phase'] == 'completed' and not r.get('error') and not r.get('finalization_error')
    for k in ('arithmetic_changed', 'new_quantization', 'candidate_promoted', 'guards_removed',
              'nvidia_byte_parity_claimed', 'raw_tensor_files', 'new_video', 'end_to_end_speedup_claimed'):
        assert r[k] is False, k
    for k in ('default_unchanged', 'diagnostic_medians_are_not_additive', 'all_frozen_outputs_equal',
              'constants_unchanged', 'fixture_inputs_unchanged', 'history_unchanged', 'persistent_io_outside_pool', 'scopes_restored'):
        assert r[k] is True, k
    for group in (m['critical_sources'], r['sources'], r['exact_gate']):
        assert group
        for path, h in group.items():
            file(path, h)
    fullpath = M.parents[1]/'c512-quad-full-v1/validation.json'
    file(fullpath, 'dc739e5351b38a091a5392fe535d1a04158e6c064428848367441b424e8153e0')
    full = js(fullpath)
    expected = {v['frame']: v for v in full['paired']['compact'] if v['round'] == 0}
    assert len(r['frames']) == len(expected) == 13
    assert r['public_graph_replays'] == 13
    for i, row in enumerate(r['frames']):
        assert row == dict(frame=i, reset=expected[i]['reset'], byte_equal=True, inputs_unchanged=True,
            history_unchanged=True, low_raw_sha256=expected[i]['low_raw_sha256'], full_raw_sha256=expected[i]['full_raw_sha256'])
    assert r['constant_hashes_before'] == r['constant_hashes_after'] == full['constant_guard']['compact']['before']
    assert len(r['constant_hashes_before']) == 1011
    names = ['decoder_input']+[f'decoder.{i}.0' for i in range(4)]
    assert [v['name'] for v in r['boundaries']] == names
    for row, c in zip(r['boundaries'], (512, 256, 128, 64, 32)):
        assert row['byte_equal'] and row['actual_body_provenance']
        assert row['projected_shape'][2] == row['skip_shape'][2] == row['output_shape'][2] == c
        assert row['skip_shape'][0] <= row['projected_shape'][0]*2 and row['skip_shape'][1] <= row['projected_shape'][1]*2
        assert set(row['hashes']) == {'p', 's', 'scale', 'out'}
        assert all(len(bytes.fromhex(h)) == 32 for h in row['hashes'].values())
        if c == 32:
            assert len(row['shift']) == 2
            assert row['output_shape'][:2] == [((v+s+7)//8)*8 for v, s in zip(row['skip_shape'], row['shift'])]
        else:
            assert row['shift'] is None and row['output_shape'] == row['skip_shape']
    shifts = [None, [0, 0], [0, 4], [4, 0], [4, 4]]
    assert [(v['case'], v['shift']) for v in r['controls']] == [(case, shift) for case in ('finite_patterns', 'strided_odd_crop') for shift in shifts]
    for v in r['controls']:
        assert v['byte_equal'] and v['scale_stride'] == [2] and len(bytes.fromhex(v['output_sha256'])) == 32
        assert v['projected_shape'] == ([32, 64, 32] if v['case'] == 'finite_patterns' else [16, 32, 32])
        assert v['skip_shape'] == ([64, 128, 32] if v['case'] == 'finite_patterns' else [29, 61, 32])
        if v['case'] == 'strided_odd_crop':
            assert v['projected_stride'] == [4096, 64, 1] and v['skip_stride'] == [8192, 64, 1]
    assert r['resources'] and r['decoder_resources']
    for group in (r['resources'], r['decoder_resources']):
        for k, v in group.items():
            assert v['spills'] == 0 and k.endswith(':'+v['hash']) and len(bytes.fromhex(v['hash'])) == 32
            assert v['selection']['selected'] == [256]
            assert all(a['spills'] == 0 for a in v['selection']['attempts'])
    assert set(r['decoder_calls']) == set(names) and len(set(r['decoder_calls'].values())) == 1
    assert next(iter(r['decoder_calls'].values())) >= 18
    labels = ['body_original', 'body_gather', 'merges_original', 'merges_gather']
    assert [v['name'] for v in r['workloads']] == labels
    assert (r['rounds'], r['replays_per_sample'], r['warmup_replays']) == (12, 32, 16)
    assert r['clock'] == 'perf_counter_ns_with_device_completion'
    for v in r['workloads']:
        assert v['captured_matches_eager'] and len(v['elapsed_ns']) == len(v['samples_ms']) == 12
        assert all(isinstance(n, int) and n > 0 for n in v['elapsed_ns'])
        assert v['samples_ms'] == [n/32e6 for n in v['elapsed_ns']]
        assert all(math.isfinite(t) and t > 0 for t in v['samples_ms'])
        assert v['median_ms'] == statistics.median(v['samples_ms'])
        want = [expected[1]['low_raw_sha256']] if v['name'].startswith('body') else [b['hashes']['out'] for b in r['boundaries']]
        assert v['output_hashes'] == want
        if v['name'].startswith('body'):
            counts = v['capture_counts']
            assert len(counts) == 3 and counts == [counts[0]]*3
            assert all(isinstance(x, int) and x > 0 for row in counts for x in row.values())
    assert len(r['orders']) == 12
    for i, order in enumerate(r['orders']):
        off = i % 4
        want = labels[off:]+labels[:off]
        assert order == (want[::-1] if i % 2 else want)
    assert [v['case'] for v in r['live_checks']] == ['zero', 'negated']
    original_hashes = {v['name']: v['output_hashes'] for v in r['workloads']}
    for row in r['live_checks']:
        assert all(row[k] for k in ('all_changed', 'byte_equal', 'eager_equal', 'restored'))
        h = row['hashes']
        assert set(h) == set(labels)
        assert h['body_original'] == h['body_gather'] and h['merges_original'] == h['merges_gather']
        assert all(h[k] != original_hashes[k] for k in labels)
    rows = {v['name']: v for v in r['workloads']}
    comparisons = []
    for family in ('body', 'merges'):
        a, b = rows[family+'_original'], rows[family+'_gather']
        comparisons.append(dict(family=family, baseline_ms=a['median_ms'], candidate_ms=b['median_ms'],
            saving_ms=a['median_ms']-b['median_ms'], saving_percent=(a['median_ms']-b['median_ms'])/a['median_ms']*100,
            faster_rounds=sum(y<x for x, y in zip(a['samples_ms'], b['samples_ms']))))
    assert r['comparisons'] == comparisons
    assert json.loads(Path(m['log']).read_text(encoding='utf-8').splitlines()[-1]) == dict(passed=True, comparisons=comparisons, diagnostic_only=True)
    result.update(passed=True, phase='completed', primary={k: file(m[k]) for k in ('result', 'log', 'lease')},
        unique_files_checked=len(cache), frames=13, real_boundaries=5, numeric_controls=10, live_checks=2,
        comparisons=comparisons, arithmetic_changed=False, candidate_promoted=False,
        all_frozen_outputs_equal=True, diagnostic_medians_are_not_additive=True)
except BaseException:
    result.update(passed=False, phase='failed', error=traceback.format_exc())
    raise
finally:
    write(m['handoff'], result)
print(json.dumps(dict(passed=True, handoff=m['handoff'])))
