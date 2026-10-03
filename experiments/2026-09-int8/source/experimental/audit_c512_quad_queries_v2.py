"""Fixed stdlib audit of the completed local V2 screen; never executes kernels."""
import hashlib, json, sys, traceback
from datetime import datetime
from pathlib import Path
M = Path('D:/Codex-NR-Experiments/nr-b580/reference/results/c512-quad-queries-v2-monitor-luna-v1/request-from-main.json')
js = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
m = js(M)


def write(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


cache = {}
def file(path, expected=None):
    p = Path(path).resolve()
    if str(p) not in cache:
        h, length = hashlib.sha256(), 0
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024*1024), b''):
                h.update(block); length += len(block)
        cache[str(p)] = dict(path=str(p), sha256=h.hexdigest(), bytes=length)
    v = cache[str(p)]
    assert expected is None or v['sha256'] == expected, p
    return v


assert Path(m['log']).is_file(), 'Known-launched log missing'
if sys.argv[1:] == ['--startup']:
    write(m['startup'], dict(at=datetime.now().astimezone().isoformat(),
                            manifest=str(M), log=m['log']))
    print('Startup recorded')
    sys.exit(0)
assert not sys.argv[1:] and not Path(m['handoff']).exists()
audit_result = dict(passed=False, phase='auditing', no_model_execution=True, no_rerun=True)
try:
    r, lease = js(m['result']), js(m['lease'])
    assert lease['owner'] == m['owner'] and lease['returncode'] == 0
    assert r['passed'] and r['phase'] == 'completed' and not r.get('error') and not r.get('finalization_error')
    for key in ('all_resources_passed_before_candidate_dispatch', 'constants_unchanged',
                'padded_kv_influence_preserved', 'ignored_queries_have_no_influence', 'fork_restored'):
        assert r[key] is True, key
    for key in ('candidate_promoted', 'full_model_run', 'performance_claimed', 'nvidia_parity_claimed', 'saved_raw_tensors'):
        assert r[key] is False, key
    for group in (m['critical_sources'], r['sources'], r['exact_gate']):
        assert group
        for p, h in group.items():
            file(p, h)
    shifts = [(0, 0), (0, 4), (4, 0), (4, 4)]
    assert len(r['geometry']) == 4
    assert {tuple(v['shift']) for v in r['geometry']} == set(shifts)
    order = [base+g%4+8*(g//4)+16*word for base in (0, 4, 32, 36)
             for word in range(2) for g in range(8)]
    for v in r['geometry']:
        assert v['valid_queries'] == 144 and v['unique'] and len(v['mapping']) == 9
        coords = []
        sy, sx = v['shift']
        for tile, g in enumerate(v['mapping']):
            qy, qx = tile//3+sy//4, tile%3+sx//4
            window, start = qy//2*2+qx//2, ((qy%2)*2+qx%2)*16
            assert g == dict(tile=tile, window=window, query_start=start)
            for i in range(16):
                local = order[start+i]
                got = (window//2*8+local//8-sy, window%2*8+local%8-sx)
                assert got == (tile//3*4+i//4, tile%3*4+i%4)
                coords.append(got)
        assert len(coords) == len(set(coords)) == 144 and set(coords) == {(y, x) for y in range(12) for x in range(12)}
    expected_labels = {'reference_attention', 'candidate_projection'} | {
        prefix+str(s) for prefix in ('candidate_attention_', 'reference_projection_') for s in shifts}
    assert len(r['resources']) == 10 and {v['label'] for v in r['resources']} == expected_labels
    for v in r['resources']:
        assert v['spills'] == 0 and len(bytes.fromhex(v['hash'])) == 32
        assert v['grid'] == ([2, 64] if v['label'] == 'reference_attention' else [9, 16])
        assert v['options'] == dict(num_warps=4, enable_fp_fusion=False,
            **({'num_stages': 1} if 'attention' in v['label'] else {}))
    cases = ('zero', 'distinct', 'padded_kv_changed', 'ignored_q_changed')
    assert len(r['controls']) == 16 and {(tuple(v['shift']), v['case']) for v in r['controls']} == {(s, c) for s in shifts for c in cases}
    assert len(r['graphs']) == 8 and {(tuple(v['shift']), v['case']) for v in r['graphs']} == {(s, c) for s in shifts for c in ('zero', 'distinct')}
    by_case = {(tuple(v['shift']), v['case']): v for v in r['controls']}
    for v in r['controls']:
        assert v['attention_byte_equal'] and v['projection_byte_equal'] and v['inputs_unchanged']
        assert len(bytes.fromhex(v['attention_sha256'])) == len(bytes.fromhex(v['projection_sha256'])) == 32
    for s in shifts:
        hashes = {c: by_case[s, c]['attention_sha256'] for c in cases}
        assert hashes['distinct'] == hashes['ignored_q_changed'] != hashes['padded_kv_changed']
    for v in r['graphs']:
        assert v['live_input_byte_equal']
        want = by_case[tuple(v['shift']), v['case']]
        assert all(v[k] == want[k] for k in ('attention_sha256', 'projection_sha256'))
    assert r['candidate_dispatches'] == 40  # 32 eager + 8 capture submissions; replays separate.
    assert set(r['constant_hashes']) == {'bias', 'weight', 'residual', 'scale', 'inverse'}
    assert all(len(bytes.fromhex(h)) == 32 for h in r['constant_hashes'].values())
    for p in (m['result'], m['log'], m['lease'], M):
        file(p)
    audit_result.update(passed=True, phase='completed', result=m['result'],
        local_screen_only=True, resources=10, controls=16, graph_checks=8,
        files=list(cache.values()))
except BaseException:
    audit_result['error'] = traceback.format_exc()
    raise
finally:
    write(m['handoff'], audit_result)
    print(json.dumps(dict(passed=audit_result['passed'], handoff=m['handoff'])))
