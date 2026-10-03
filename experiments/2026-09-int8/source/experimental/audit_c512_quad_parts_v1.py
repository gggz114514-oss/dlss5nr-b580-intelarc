"""Fixed stored-result audit of frozen-kernel C512 stage diagnostics."""
import hashlib, json, math, statistics, sys, traceback
from pathlib import Path
from datetime import datetime
M = Path('D:/Codex-NR-Experiments/nr-b580/reference/results/c512-quad-parts-v1-monitor-luna-v1/request-from-main.json')
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
m = js(M)
cache = {}


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(value, f, indent=2, allow_nan=False); f.write('\n')


def file(path, expected=None):
    p = Path(path).resolve(); key = str(p).casefold()
    if key not in cache:
        h, length = hashlib.sha256(), 0
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024*1024), b''):
                h.update(block); length += len(block)
        cache[key] = dict(path=str(p), sha256=h.hexdigest(), bytes=length)
    v = cache[key]
    assert expected is None or v['sha256'] == expected, p
    return v


assert Path(m['log']).is_file()
if sys.argv[1:] == ['--startup']:
    write(m['startup'], dict(at=datetime.now().astimezone().isoformat(), manifest=str(M), log=m['log']))
    print('Startup recorded'); sys.exit(0)
assert not sys.argv[1:] and not Path(m['handoff']).exists()
audit_result = dict(passed=False, phase='auditing', no_model_execution=True, no_rerun=True)
try:
    r, lease = js(m['result']), js(m['lease'])
    assert lease['owner'] == m['owner'] and lease['returncode'] == 0
    assert r['passed'] and r['phase'] == 'completed' and not r.get('error') and not r.get('finalization_error')
    for key in ('candidate_promoted', 'arithmetic_changed', 'profiler_used', 'raw_tensor_files', 'new_video', 'end_to_end_speedup_claimed'):
        assert r[key] is False, key
    for key in ('default_unchanged', 'diagnostic_medians_are_not_additive', 'real_body_byte_equal', 'fixture_inputs_unchanged',
                'history_seed_and_body_inputs_unchanged', 'constants_unchanged', 'persistent_io_outside_pool', 'scopes_restored'):
        assert r[key] is True, key
    for group in (m['critical_sources'], r['sources'], r['exact_gate']):
        assert group
        for path, h in group.items():
            file(path, h)
    fp = M.parents[1]/'c512-quad-full-v1/validation.json'
    file(fp, 'dc739e5351b38a091a5392fe535d1a04158e6c064428848367441b424e8153e0')
    full = js(fp)
    expected = {v['frame']: v for v in full['paired']['compact'] if v['round'] == 0}
    assert len(r['warmup']) == 2
    for i, v in enumerate(r['warmup']):
        assert v == dict(frame=i, byte_equal=True, low_raw_sha256=expected[i]['low_raw_sha256'], full_raw_sha256=expected[i]['full_raw_sha256'])
    assert r['public_graph_replays'] == 2 and r['rewrite_scopes'] == 7
    assert r['constant_hashes_before'] == r['constant_hashes_after'] == full['constant_guard']['compact']['before']
    assert len(r['constant_hashes_before']) == 1011
    names = [f'{side}512.{i}' for side in ('encoder', 'decoder') for i in range(8)]
    assert [v['name'] for v in r['fixtures']] == names
    prior_boundaries = {(v['block'], v['kind']): v for v in full['boundaries'] if v['label'] == 'temporal'}
    assert len(prior_boundaries) == 32
    for v in r['fixtures']:
        assert v['attention_byte_equal'] and v['projection_byte_equal']
        assert v['shift'] == prior_boundaries[v['name'], 'attention']['shift']
        assert v['hashes']['new'] == prior_boundaries[v['name'], 'attention']['raw_sha256']
        assert v['hashes']['full'] == prior_boundaries[v['name'], 'projection']['raw_sha256']
        assert set(v['hashes']) == {'q', 'k', 'v', 'residual', 'old', 'new', 'full'}
        assert all(len(bytes.fromhex(h)) == 32 for h in v['hashes'].values())
    sp = M.parents[1]/'c512-quad-queries-v2/validation.json'
    file(sp, '546d2ee7b20478fdacaa7a6bde4bc547b02e4dd8b490685669578b405ce5bb89')
    screened = {v['label']: v for v in js(sp)['resources']}
    assert len(r['resources']) == 10 and set(r['resources']) == set(screened)
    for label, v in r['resources'].items():
        assert v == {k: screened[label][k] for k in ('hash', 'spills', 'registers', 'shared_bytes', 'options')}
        assert v['spills'] == 0
    work_names = [stage+'/'+route for stage in ('attention', 'projection', 'pair') for route in ('old', 'quad')]
    assert [w['name'] for w in r['workloads']] == work_names
    assert r['rounds'] == 12 and r['replays_per_sample'] == 32
    for w in r['workloads']:
        assert w['name'] == w['stage']+'/'+w['route'] and w['captured_output_byte_equal']
        assert w['kernels_per_replay'] == (32 if w['stage'] == 'pair' else 16)
        key = ('old' if w['route'] == 'old' else 'new') if w['stage'] == 'attention' else 'full'
        assert w['output_hashes'] == [v['hashes'][key] for v in r['fixtures']]
        values = w['samples_ms']
        assert len(values) == 12 and all(math.isfinite(x) and x > 0 for x in values)
        assert statistics.median(values) == w['median_ms']
    assert len(r['orders']) == 12
    for i, order in enumerate(r['orders']):
        off = i%6; want = work_names[off:]+work_names[:off]
        assert order == (want[::-1] if i%2 else want)
    assert r['live_checks'] == [dict(stage=s, both_routes_changed=True, byte_equal=True, restored=True) for s in ('attention', 'projection', 'pair')]
    medians = {w['name']: w['median_ms'] for w in r['workloads']}
    final = json.loads(Path(m['log']).read_text(encoding='utf-8').splitlines()[-1])
    assert final == dict(passed=True, medians_ms=medians, diagnostic_only=True)
    audit_result.update(passed=True, phase='completed', primary={k: file(m[k]) for k in ('result', 'log', 'lease')},
        unique_files_checked=len(cache), medians_ms=medians, workloads=r['workloads'],
        fixtures=16, live_checks=3, diagnostic_medians_are_not_additive=True, candidate_promoted=False)
except BaseException:
    audit_result.update(passed=False, phase='failed', error=traceback.format_exc())
    raise
finally:
    write(m['handoff'], audit_result)
print(json.dumps(dict(passed=True, handoff=m['handoff'])))
