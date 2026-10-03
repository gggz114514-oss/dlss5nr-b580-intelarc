"""Prewritten stored-result audit for host diagnostics; no model execution."""
import hashlib, json, math, statistics, sys, traceback
from datetime import datetime
from pathlib import Path
M = Path('D:/Codex-NR-Experiments/nr-b580/reference/results/c512-host-cost-v1-monitor-luna-v1/request-from-main.json')
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
                h.update(block)
                length += len(block)
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
    for key in ('default_unchanged', 'independent_history_per_scenario', 'trace_ast_identity_verified',
                'cpu_preserved_history_and_replay_counts', 'all_plain_and_traced_outputs_frozen_equal',
                'trace_bindings_restored', 'steady_no_recapture', 'guards_constants_histories_preserved'):
        assert r[key] is True, key
    for key in ('candidate_promoted', 'arithmetic_changed', 'guards_removed', 'new_video'):
        assert r[key] is False, key
    assert r['extra_gpu_fences_in_trace'] == 0
    assert sorted(r['annotations']) == sorted(['rgb_validation', 'motion_value_checks',
        'front', 'history_warp', 'front', 'graph_call', 'history_commit'])
    for path, digest in m['critical_sources'].items():
        file(path, digest)
    for group in ('sources', 'exact_gate'):
        assert r[group]
        for path, digest in r[group].items():
            file(path, digest)
    fp = M.parents[1] / 'c512-int8-full-v1/validation.json'
    file(fp, 'd1d2b8e21b1c9e95e7d99b7e0bf5735237781e2412301295e7472aeff92ba7f2')
    full = js(fp)
    review = fp.with_name('user-review-v1.json')
    file(review, '2f923c191f990960ea60ed28347d411057424adc57b38d99b4ac185042c6c9d6')
    assert js(review)['status'] == 'accepted_face_quality_for_c512_int8_full_v1'
    assert r['reused_approved_video'] == full['video'] == js(review)['video']
    v = r['reused_approved_video']
    file(v['path'], v['sha256'], v['bytes'])
    routes = ['baseline', 'floor16']
    scenarios = ['baseline_plain', 'baseline_traced', 'floor16_plain', 'floor16_traced']
    expected = {n: {v['frame']: v for v in full['paired']['runs'][n] if v['round'] == 0} for n in routes}
    assert len(r['rows']) == 208
    history = {}
    seen = set()
    for index, row in enumerate(r['rows']):
        repeat, offset = divmod(index, 52)
        frame, position = divmod(offset, 4)
        shift = (repeat + frame) % 4
        order = scenarios[shift:] + scenarios[:shift]
        assert (row['round'], row['frame'], row['scenario']) == (repeat, frame, order[position])
        assert row['order'] == order
        case = row['scenario']
        name, mode = case.rsplit('_', 1)
        assert row['route'] == name and row['mode'] == mode
        ref = expected[name][frame]
        assert row['reset'] == ref['reset']
        assert row['full_raw_sha256'] == ref['full_raw_sha256'] and row['low_raw_sha256'] == ref['low_raw_sha256']
        assert row['next_seed'] == ref['next_seed']
        key = repeat, case
        before = history.get(key, (None, 0))
        assert (row['previous_low_sha256'], row['seed_before']) == before
        history[key] = row['low_raw_sha256'], row['next_seed']
        assert all(row[k] is True for k in ('frozen_output_equal', 'private_history_matches_low',
            'other_histories_unchanged', 'inputs_unchanged'))
        assert math.isfinite(row['host_ms']) and row['host_ms'] > 0
        assert (repeat, frame, case) not in seen
        seen.add((repeat, frame, case))
        if mode == 'plain':
            assert row['trace'] is None
            continue
        trace = row['trace']
        assert trace['ast_identity_verified'] and trace['bindings_restored'] and trace['extra_gpu_fences'] == 0
        assert trace['host_ms'] == row['host_ms']
        parents = dict(pipeline=None, prepare='pipeline', model='pipeline', composite='pipeline', final_wait='pipeline',
            call_guard='model', vit_constants='call_guard', rgb_validation='model', motion_value_checks='model',
            front='model', graph_call='model', graph_validate='graph_call', graph_constants='graph_validate',
            graph_completion_wait='graph_call', history_commit='model')
        if not row['reset']:
            parents['history_warp'] = 'model'
        if name == 'floor16':
            parents['c512_constants'] = 'call_guard'
        spans = trace['spans']
        assert len(spans) == len(parents) and {v['name'] for v in spans} == set(parents)
        assert spans[0]['name'] == 'pipeline' and spans[0]['start_ns'] == 0
        for i, span in enumerate(spans):
            assert span['id'] == i
            assert isinstance(span['inclusive_ns'], int) and isinstance(span['exclusive_ns'], int)
            assert span['inclusive_ns'] == span['end_ns']-span['start_ns'] > 0
            if span['parent'] is None:
                assert i == 0 and parents[span['name']] is None
            else:
                assert 0 <= span['parent'] < i
                parent = spans[span['parent']]
                assert parent['name'] == parents[span['name']]
                assert parent['start_ns'] <= span['start_ns'] < span['end_ns'] <= parent['end_ns']
            children = [v for v in spans if v['parent'] == i]
            assert span['exclusive_ns'] == span['inclusive_ns']-sum(v['inclusive_ns'] for v in children)
            assert span['exclusive_ns'] >= 0
        assert sum(v['exclusive_ns'] for v in spans) == spans[0]['inclusive_ns']
        assert row['host_ms'] == spans[0]['inclusive_ns']/1e6
    assert len(seen) == 208
    for case in scenarios:
        for mode in ('all', 'temporal'):
            rows = [v for v in r['rows'] if v['scenario'] == case and (mode == 'all' or not v['reset'])]
            assert len(rows) == (52 if mode == 'all' else 44)
            summary = r['pipeline'][case][mode]
            assert summary['mean_ms'] == statistics.mean(v['host_ms'] for v in rows)
            assert summary['round_mean_ms'] == [statistics.mean(v['host_ms'] for v in rows if v['round'] == i) for i in range(4)]
    for name in routes:
        groups = {}
        for row in r['rows']:
            if row['route'] == name and row['mode'] == 'traced' and not row['reset']:
                for span in row['trace']['spans']:
                    groups.setdefault(span['name'], []).append(span)
        assert set(groups) == set(r['span_summary'][name])
        for label, rows in groups.items():
            v = r['span_summary'][name][label]
            assert v['count'] == len(rows) == 44
            assert v['mean_inclusive_ms'] == statistics.mean(x['inclusive_ns'] for x in rows)/1e6
            assert v['mean_exclusive_ms'] == statistics.mean(x['exclusive_ns'] for x in rows)/1e6
        for mode in ('all', 'temporal'):
            assert r['instrumentation_overhead'][name][mode] == r['pipeline'][name+'_traced'][mode]['mean_ms']-r['pipeline'][name+'_plain'][mode]['mean_ms']
    labels = ['call_guard', 'vit_constants', 'graph_validate', 'graph_constants', 'buffer_enumeration']
    assert len(r['cpu']) == 96
    assert {(v['route'], v['round'], v['label']) for v in r['cpu']} == {
        (name, repeat, label) for name in routes for repeat in range(8)
        for label in labels + (['c512_constants', 'c512_registry'] if name == 'floor16' else [])}
    for row in r['cpu']:
        assert row['iterations'] == 32 and row['no_model_execution']
        assert math.isfinite(row['per_call_ms']) and row['per_call_ms'] > 0
    for name in routes:
        own_labels = labels + (['c512_constants', 'c512_registry'] if name == 'floor16' else [])
        assert set(r['cpu_summary'][name]) == set(own_labels)
        for label in own_labels:
            samples = [v['per_call_ms'] for v in r['cpu'] if v['route'] == name and v['label'] == label]
            assert len(samples) == 8
            assert r['cpu_summary'][name][label] == dict(samples_ms=samples, median_ms=statistics.median(samples))
        guard = r['constant_guard'][name]
        assert guard['before'] == guard['after'] and len(guard['before']) == (819 if name == 'baseline' else 1011)
        assert '_previous' not in guard['before']
        for digest in guard['before'].values():
            assert len(bytes.fromhex(digest)) == 32
        assert guard['before'] == {k: v['raw_sha256'] for k, v in full['constant_guard'][name]['before'].items()}
        graphs = r['graphs'][name]
        assert len(graphs) == 2 and sum(v['replays'] for v in graphs) == 107
        assert all(v['persistent_inputs_and_output_verified_outside_pool'] for v in graphs)
        assert [v['captured_dispatch'] for v in graphs] == [v['captured_dispatch'] for v in full['graphs'][name]]
    assert r['capture_counts'] == full['capture_build_counts']
    final = json.loads(Path(m['log']).read_text(encoding='utf-8').splitlines()[-1])
    assert final == dict(passed=True, pipeline_rows=208, cpu_cases=96,
        pipeline=r['pipeline'], cpu_summary=r['cpu_summary'], no_arithmetic_change=True)
    audit_result.update(passed=True, phase='completed', primary={k: file(m[k]) for k in ('result', 'log', 'lease')},
        unique_files_checked=len(cache), pipeline_rows=208, traced_rows=104, cpu_cases=96,
        pipeline=r['pipeline'], span_summary=r['span_summary'], cpu_summary=r['cpu_summary'],
        instrumentation_overhead=r['instrumentation_overhead'], all_frozen_output_bytes_equal=True,
        guards_removed=False, candidate_promoted=False, reused_approved_video=r['reused_approved_video'],
        scope='Host diagnostics only. Main must interpret overlap and probe overhead before proposing an optimization.')
except BaseException:
    audit_result.update(passed=False, phase='failed', error=traceback.format_exc())
    raise
finally:
    write(m['handoff'], audit_result)
print(json.dumps(dict(passed=True, handoff=m['handoff'], pipeline_rows=208)))
