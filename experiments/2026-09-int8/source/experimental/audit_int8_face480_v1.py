"""Prescribed finite Luna checks. No GPU/video execution or improvised audits."""
import hashlib,json,math,sys,traceback
from datetime import datetime
from pathlib import Path

M=Path('D:/Codex-NR-Experiments/nr-b580/reference/results/int8-ffn-face480-review-v1-monitor-luna-v1/request-from-main.json')
js=lambda p:json.loads(Path(p).read_text(encoding='utf-8-sig'))
m=js(M);log=Path(m['log']);lease=Path(m['lease']);result=Path(m['result'])
assert log.is_file(),'Known-launched log missing: stop and report to main'
def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+chr(10),encoding='utf-8')

if sys.argv[1:]==['--startup']:
    assert not Path(m['startup']).exists()
    write(m['startup'],dict(read_at_local=datetime.now().astimezone().isoformat(),manifest=str(M),
        actual_log=str(log),log_bytes=log.stat().st_size,result_exists=result.is_file(),lease_exists=lease.is_file()))
    print('Startup recorded; monitor only these manifest paths');sys.exit(0)
assert not sys.argv[1:]
assert not Path(m['handoff']).exists(),'Do not rerun completed or failed audit'
assert lease.is_file(),'Lease not complete: do not audit yet'
audit=dict(passed=False,phase='auditing',manifest=str(M),no_model_execution=True,no_video_execution=True,
    human_review='pending',new_quality_approved=False,candidate_promoted=False)
cache={}
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
    l=js(lease);r=js(result)
    assert l['owner']==m['owner'] and l['returncode']==0
    assert 0<=l['finished_unix']-l['started_unix']<=1200
    assert r['passed'] and r['phase']=='completed' and not r.get('error') and not r.get('finalization_error')
    assert r['frames_completed']==243 and len(r['frames'])==243 and len(r['images'])==7
    assert r['model_input_face_crop'] is False and r['performance_benchmark'] is False
    assert r['panel_order']==['source','selected_NR256_FP16','continuous_INT8_FFN_NR256']
    assert r['face_roi']==[256,16,640,400] and r['face_enlargement']==2
    assert r['scaler_identity_and_zero_motion_passed'] and r['independent_histories']
    assert r['reset_reproduces_first_frame'] and r['inputs_constants_held_outputs_unchanged']
    for name in ('human_review','new_quality_approved','candidate_promoted'):assert r[name]==audit[name]
    audit['primary']={name:file(path) for name,path in [('report',result),('log',log),('lease',lease),('manifest',M)]}
    for p,h in m['critical_sources'].items():file(p,h)
    audit['source_counts']={}
    for key in ('sources','exact_gate'):
        assert isinstance(r[key],dict) and r[key]
        for p,h in r[key].items():file(p,h)
        audit['source_counts'][key]=len(r[key])
    arrays=set()
    for i,row in enumerate(r['frames']):
        assert row['frame']==i and row['reset']==(i==0)
        assert row['inputs_unchanged'] and row['held_outputs_unchanged']
        for name in ('selected','int8_p4'):
            run=row['runs'][name];meta=run['low']
            assert run['next_seed']==i+1 and run['private_history_matches_low']
            assert meta['shape']==[256,256,3] and meta['dtype']=='<f2'
            file(meta['path'],meta['sha256'],meta['stored_bytes']);arrays.add(str(Path(meta['path']).resolve()))
        meta=row['motion'];file(meta['path'],meta['sha256'],meta['stored_bytes'])
        assert all(math.isfinite(row[k]) and row[k]>=0 for k in ('full_mae','face_roi_mae'))
    audit['low_array_files_checked']=len(arrays)
    audit['images']=[file(p,h) for p,h in r['images'].items()]
    p=r['probe'];assert (p['width'],p['height'],p['avg_frame_rate'],int(p['nb_read_frames']))==(2592,1352,'24/1',243)
    assert abs(float(p['duration'])-10.125)<.001
    assert all(p.get(k)=='bt709' for k in ('color_space','color_transfer','color_primaries')) and p['color_range']=='tv'
    v=r['video'];assert Path(v['path']).samefile(m['video']);audit['video']=file(v['path'],v['sha256'],v['bytes'])
    for name,graphs in r['graphs'].items():
        assert len(graphs)==2 and sum(g['replays'] for g in graphs)==246
        assert all(g['persistent_inputs_and_output_verified_outside_pool'] for g in graphs)
    resources=[r['candidate']['ffn_resources'],r['candidate']['capture']['resources'],r['selected_resources']]
    assert all(group and all(v['spills']==0 for v in group.values()) for group in resources)
    assert r['candidate']['registered_packed_tensors']==40
    final=json.loads(log.read_text(encoding='utf-8-sig').splitlines()[-1])
    assert final['passed'] and final['frames']==243 and final['video']==r['video']
    audit.update(passed=True,phase='completed',frames=243,graph_replays_per_route=246,
        probe=p,independent_histories=True,reset_reproduces_first_frame=True,
        scope='Audit of stored completed experiment evidence; video/model not rerun',
        note='Three columns source/selected FP16/new continuous INT8; full-frame processing, fixed output crop enlarged for face inspection')
except BaseException:
    audit.update(passed=False,phase='failed',error=traceback.format_exc());raise
finally:write(m['handoff'],audit)
print(json.dumps(dict(passed=True,frames=243,handoff=m['handoff'],human_review='pending'),ensure_ascii=False))
