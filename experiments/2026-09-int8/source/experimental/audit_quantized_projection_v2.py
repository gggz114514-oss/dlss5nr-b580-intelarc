"""Luna's bounded stored-result audit; no model execution or self-written checks."""
import hashlib,json,math,statistics,sys,traceback
from datetime import datetime
from pathlib import Path
M=Path('D:/Codex-NR-Experiments/nr-b580/reference/results/quantized-projection-store-v2-monitor-luna-v1/request-from-main.json')
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
m=js(M)
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')
assert Path(m['log']).is_file(),'Known-launched log missing'
if sys.argv[1:]==['--startup']:
    assert not Path(m['startup']).exists()
    write(m['startup'],dict(read_at=datetime.now().astimezone().isoformat(),manifest=str(M),log=m['log'],log_bytes=Path(m['log']).stat().st_size))
    print('Startup recorded');sys.exit(0)
assert not sys.argv[1:] and not Path(m['handoff']).exists()
a=dict(passed=False,phase='auditing',no_model_execution=True,no_rerun=True);cache={}
def file(path,expected=None,size=None):
    p=Path(path).resolve();key=str(p).casefold()
    if key not in cache:
        h=hashlib.sha256();n=0
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b);n+=len(b)
        cache[key]=dict(path=str(p),sha256=h.hexdigest(),bytes=n)
    v=cache[key]
    assert expected is None or v['sha256']==expected,p
    assert size is None or v['bytes']==size,p
    return v
try:
    r,l=js(m['result']),js(m['lease'])
    assert l['owner']==m['owner'] and l['returncode']==0
    assert r['passed'] and r['phase']=='completed' and not r.get('error') and not r.get('finalization_error')
    for key in ('default_unchanged','fewer_physical_calls_same_logical_rounding','body_inputs_and_private_history_unchanged',
                'all_frozen_car_and_face_outputs_byte_equal','reset_reproduces_first_frame','inputs_constants_history_and_held_outputs_unchanged'):assert r[key]
    for key in ('candidate_promoted','new_quantization','recalibration','new_video'):assert r[key] is False
    assert r['previous_failure']['setup_only'] and r['previous_failure']['returncode']==1
    file(r['previous_failure']['report'],r['previous_failure']['sha256'])
    assert r['sharing_setup']['donor_never_executed'] and r['sharing_setup']['unchanged_topology_guard']
    assert r['donor_remained_fresh_and_unchanged']
    assert set(r['sharing_setup']['routes'])=={'repaired','branched','combined'}
    for row in r['sharing_setup']['routes'].values():
        assert row['owned_packed_buffers']==40 and row['base_buffers_shared']>0
        assert row['base_buffers_identical_objects'] and row['packed_buffers_disjoint']
        assert row['receipt']['all_contents_verified_as_raw_bytes'] and not row['receipt']['history_shared']
    for p,h in m['critical_sources'].items():file(p,h)
    for group in ('sources','exact_gate'):
        assert r[group]
        for p,h in r[group].items():file(p,h)
    prior_path=Path('D:/Codex-NR-Experiments/nr-b580/reference/results/int8-ffn-range-repair-v1/validation.json')
    file(prior_path,'bd53a2e2b818886e79fed05e836646bdd89814a434aa932ae5bdd4f7944ce2aa');prior=js(prior_path)
    expected={v['frame']:v for v in prior['paired']['runs']['repaired_int8'] if v['round']==0}
    names=('repaired','branched','combined')
    assert set(r['body'])==set(names) and set(r['paired']['runs'])==set(names) and r['paired']['passed']
    assert len(r['eager_graph_checks'])==6 and all(v['byte_equal'] and v['inputs_unchanged'] for v in r['eager_graph_checks'])
    for name in names:
        samples=r['body'][name]['samples_ms'];assert len(samples)==7 and all(math.isfinite(x) and x>0 for x in samples)
        assert math.isclose(statistics.median(samples),r['body'][name]['median_ms'],abs_tol=1e-9)
        builds=r['capture_build_counts'][name];assert len(builds)==6
        rows=r['paired']['runs'][name];assert len(rows)==39
        for index,v in enumerate(rows):
            assert (v['round'],v['frame'])==divmod(index,13)
            ref=expected[v['frame']]
            assert v['frozen_byte_equal'] and v['private_history_matches_low'] and v['next_seed']==ref['next_seed']
            assert v['full_raw_sha256']==ref['full_raw_sha256'] and v['low_raw_sha256']==ref['low_raw_sha256']
            assert math.isfinite(v['host_ms']) and v['host_ms']>0
        for mode in ('all','temporal'):
            value=statistics.mean(v['host_ms'] for v in rows if mode=='all' or not v['reset'])
            assert math.isclose(value,r['paired']['summary'][name][mode]['mean_host_ms'],abs_tol=1e-9)
            for k in range(3):
                value=statistics.mean(v['host_ms'] for v in rows if v['round']==k and (mode=='all' or not v['reset']))
                assert math.isclose(value,r['paired']['summary'][name][mode]['round_mean_host_ms'][k],abs_tol=1e-9)
    for i in range(6):
        b,p,c=[r['capture_build_counts'][n][i] for n in names]
        assert c['standalone_fp8']<p['standalone_fp8']<b['standalone_fp8']
        assert c['triton_calls']<p['triton_calls']<b['triton_calls']
        assert c['quantization_calls']==p['quantization_calls']==b['quantization_calls']
    assert len(r['frames'])==r['frames_completed']==243
    for i,row in enumerate(r['frames']):
        assert row['frame']==i and row['reset']==(i==0) and row['inputs_unchanged'] and row['held_outputs_unchanged']
        ref=prior['frames'][i];assert set(row['runs'])==set(names)
        for v in row['runs'].values():
            assert v['byte_equal'] and v['private_history_matches_low'] and v['next_seed']==i+1
            assert v['full_raw_sha256']==ref['full_raw_sha256'] and v['low_raw_sha256']==ref['low']['raw_sha256']
    for graphs in r['graphs'].values():
        assert len(graphs)==2 and sum(v['replays'] for v in graphs)==285
        assert all(v['persistent_inputs_and_output_verified_outside_pool'] for v in graphs)
    for candidate in r['candidates'].values():
        assert candidate['range_calibration_unchanged']
        resources=candidate['projection_stores']['resources'];assert resources and all(v['spills']==0 for v in resources.values())
    video=r['reused_approved_video'];assert video==prior['video'];file(video['path'],video['sha256'],video['bytes'])
    final=json.loads(Path(m['log']).read_text(encoding='utf-8').splitlines()[-1])
    assert final['passed'] and final['frames']==243 and final['all_byte_equal']
    changes={n:{mode:(r['paired']['summary'][n][mode]['mean_host_ms']/r['paired']['summary']['repaired'][mode]['mean_host_ms']-1)*100
                for mode in ('all','temporal')} for n in ('branched','combined')}
    a.update(passed=True,phase='completed',primary={k:file(m[k]) for k in ('result','log','lease')},unique_files_checked=len(cache),
        body=r['body'],timing=r['paired']['summary'],change_percent=changes,build_counts=r['capture_build_counts'],
        all243_face_bytes_equal=True,candidate_promoted=False,
        scope='Numerical/source audit passed; main decides whether measured performance warrants adoption')
except BaseException:
    a.update(passed=False,phase='failed',error=traceback.format_exc());raise
finally:write(m['handoff'],a)
print(json.dumps(dict(passed=True,handoff=m['handoff'],all243_face_bytes_equal=True)))
