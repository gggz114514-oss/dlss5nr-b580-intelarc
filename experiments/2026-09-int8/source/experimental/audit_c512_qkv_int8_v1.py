"""Fixed stored-result audit of approximate INT8 QKV integer oracle, error and local timing screen."""
import hashlib, json, math, statistics, sys, traceback
from pathlib import Path
from datetime import datetime
M = Path('D:/Codex-NR-Experiments/nr-b580/reference/results/c512-qkv-int8-v1-monitor-luna-v1/request-from-main.json')
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
    for key in ('candidate_promoted', 'candidate_integrated_into_model', 'quality_accepted', 'weight_quantization_timed', 'profiler_used', 'raw_tensor_files', 'new_video', 'end_to_end_speedup_claimed'):
        assert r[key] is False, key
    for key in ('arithmetic_changed', 'fp8_boundaries_preserved', 'input_quantization_timed', 'int8_configs_byte_equal', 'all_resources_checked_before_candidate_dispatch', 'default_unchanged', 'diagnostic_medians_are_not_additive', 'fixture_source_body_byte_equal', 'fixture_inputs_unchanged',
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
    pp = M.parents[1]/'c512-quad-parts-v1/validation.json'
    file(pp, '8972a47416ae82717f32453adbf5ab11924c9666631db5d36059515d26c652dd')
    prior = {v['name']: v for v in js(pp)['fixtures']}
    for v in r['fixtures']:
        assert v['actual_body_provenance'] and v['prior_boundary_hashes_equal']
        assert v['shift'] == prior[v['name']]['shift']
        assert set(v['hashes']) == {'x', 'mlp', 'q', 'k', 'v'}
        assert all(len(bytes.fromhex(h)) == 32 for h in v['hashes'].values())
        assert all(v['hashes'][k] == prior[v['name']]['hashes'][k] for k in ('q', 'k', 'v'))
        assert v['hashes']['mlp'] == prior[v['name']]['hashes']['residual']
    up = M.parents[1]/'c512-upstream-parts-v1/validation.json'
    file(up, '59768dbfabeafe6bb98ec20eb9b6bcca9d8253c1367b309b99b0bbb4c5205f3a')
    upstream = js(up)
    assert r['source_resources'] == upstream['source_resources']
    assert r['source_selections'] == upstream['source_selections']
    assert r['fixtures'] == upstream['fixtures']
    qkv_labels = ['dense']+['pack_'+str(s) for s in ((0,0),(0,4),(4,0),(4,4))]
    assert set(r['resources']) == {'qkv:'+k for k in qkv_labels}
    assert all(v == r['source_resources'][k] and v['spills']==0 for k,v in r['resources'].items())
    shifts = [(0,0),(0,4),(4,0),(4,4)]
    configs = [(32,64),(16,64),(16,32)]
    labels = ['int8:quantize_rows']+['int8:dense'+str(c) for c in configs]+['qkv:pack_'+str(s) for s in shifts]
    assert set(r['candidate_resources']) == set(labels)
    for k,v in r['candidate_resources'].items():
        assert len(bytes.fromhex(v['hash']))==32 and isinstance(v['spills'],int) and v['spills']>=0
        assert set(v)=={'hash','spills','registers','shared_bytes'}
        if k.startswith('qkv:'): assert v==r['source_resources'][k]
    accepted=[]
    assert [tuple(v['config']) for v in r['configs']]==configs
    for cfg in r['configs']:
        c=tuple(cfg['config'])
        keys=['int8:quantize_rows','int8:dense'+str(c)]+['qkv:pack_'+str(s) for s in shifts]
        good=all(r['candidate_resources'][k]['spills']==0 for k in keys)
        assert cfg['accepted'] is good
        assert cfg['reason']==('zero_spill' if good else 'spill_rejected_before_dispatch')
        if good:accepted.append(f'int8_{c[0]}x{c[1]}')
    assert accepted and r['candidate_dispatches']>0
    assert r['packed_constants_before']==r['packed_constants_after']
    assert list(r['packed_constants_before'])==names
    for v in r['packed_constants_before'].values():
        assert set(v)=={'source_sha256','qw_sha256','sw_sha256'}
        assert all(len(bytes.fromhex(h))==32 for h in v.values())
    work_names=['qkv']+accepted
    assert [w['name'] for w in r['workloads']]==work_names
    assert r['rounds']==12 and r['replays_per_sample']==32
    want_hashes=[f['hashes'][k] for f in r['fixtures'] for k in ('q','k','v')]
    assert r['workloads'][0]['output_hashes']==want_hashes
    candidate_hashes=r['workloads'][1]['output_hashes']
    assert len(candidate_hashes)==48 and all(len(bytes.fromhex(h))==32 for h in candidate_hashes)
    for w in r['workloads']:
        assert w['captured_matches_eager'] and w['kernels_per_replay']==(32 if w['name']=='qkv' else 48)
        if w['name']!='qkv':assert w['output_hashes']==candidate_hashes
        values=w['samples_ms']
        assert len(values)==12 and all(math.isfinite(x) and x>0 for x in values)
        assert statistics.median(values)==w['median_ms']
    assert len(r['orders'])==12
    for i,order in enumerate(r['orders']):
        off=i%len(work_names);want=work_names[off:]+work_names[:off]
        assert order==(want[::-1] if i%2 else want)
    assert [(v['case'],v['route']) for v in r['controls']]==[(case,route) for case in ('zero','negated') for route in work_names]
    for case in ('zero','negated'):
        rows=[v for v in r['controls'] if v['case']==case]
        values=rows[1]['output_hashes']
        assert len(values)==48 and values!=candidate_hashes
        assert all(v['output_hashes']==values for v in rows[1:])
        for v in rows:
            assert v['changed'] and v['eager_byte_equal']
            assert v['zero_reference_byte_equal'] is (True if case=='zero' else None)
        if case=='zero':assert rows[0]['output_hashes']==values
    assert r['live_checks']==[dict(case=c,all_routes_changed=True,int8_configs_byte_equal=True,restored=True) for c in ('zero','negated')]
    assert [(v['case'],v['route']) for v in r['oracle_checks']]==[(case,route) for case in ('actual','zero','negated') for route in accepted]
    oracle_by_case={}
    for row in r['oracle_checks']:
        assert [v['name'] for v in row['blocks']]==names
        for v in row['blocks']:
            assert v['input_quantization_byte_equal'] and v['dense_samples_byte_equal'] and v['samples']==48
            assert all(len(bytes.fromhex(v[k]))==32 for k in ('qx_sha256','sx_sha256','sample_sha256'))
        if row['case'] in oracle_by_case:assert row['blocks']==oracle_by_case[row['case']]
        oracle_by_case[row['case']]=row['blocks']
    assert [(v['route'],v['name'],v['family']) for v in r['numerical_errors']]==[(route,name,key) for route in accepted for name in names for key in ('q','k','v')]
    errors={}
    for row in r['numerical_errors']:
        e=row['metrics'];key=(row['name'],row['family'])
        assert e['count']==73728 and e['padding_values']==57344 and e['padding_byte_equal']
        assert e['finite'] and e['output_fp8_boundary']
        assert all(math.isfinite(e[k]) for k in ('max_abs','mean_abs','rms','mean_signed','byte_equal_fraction'))
        assert 0<=e['byte_equal_fraction']<=1 and 0<=e['mean_abs']<=e['rms']+1e-12
        assert e['rms']<=e['max_abs']+1e-12 and abs(e['mean_signed'])<=e['mean_abs']+1e-12
        if key in errors:assert e==errors[key]
        errors[key]=e
    baseline=r['workloads'][0]
    want_comparisons=[]
    for row in r['workloads'][1:]:
        a,b=baseline['median_ms'],row['median_ms']
        want_comparisons.append(dict(route=row['name'],baseline_ms=a,candidate_ms=b,saving_ms=a-b,
            saving_percent=(a-b)/a*100,faster_rounds=sum(y<x for x,y in zip(baseline['samples_ms'],row['samples_ms']))))
    assert r['local_comparisons']==want_comparisons
    medians = {w['name']: w['median_ms'] for w in r['workloads']}
    final = json.loads(Path(m['log']).read_text(encoding='utf-8').splitlines()[-1])
    assert final == dict(passed=True, medians_ms=medians, diagnostic_only=True)
    audit_result.update(passed=True, phase='completed', primary={k: file(m[k]) for k in ('result', 'log', 'lease')},
        unique_files_checked=len(cache), medians_ms=medians, workloads=r['workloads'],
        fixtures=16, live_checks=2, quality_accepted=False, numerical_errors=r['numerical_errors'], local_comparisons=r['local_comparisons'], configs=r['configs'], diagnostic_medians_are_not_additive=True, candidate_promoted=False)
except BaseException:
    audit_result.update(passed=False, phase='failed', error=traceback.format_exc())
    raise
finally:
    write(m['handoff'], audit_result)
print(json.dumps(dict(passed=True, handoff=m['handoff'])))
